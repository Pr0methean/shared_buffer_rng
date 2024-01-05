#![feature(sync_unsafe_cell)]
#![feature(iter_array_chunks)]
#![feature(lazy_cell)]
#![feature(iterator_try_collect)]

use core::hint::spin_loop;
use core::cell::SyncUnsafeCell;
use std::iter::repeat_with;
use std::pin::pin;
use std::sync::{Arc}; // FIXME: Change to loom::sync::Arc once https://github.com/tokio-rs/loom/issues/156 is fixed
#[cfg(not(loom))]
use std::{sync::{Mutex, atomic::{AtomicUsize, Ordering}}, thread::{spawn, yield_now}};
use std::slice;
#[cfg(loom)]
use loom::{sync::{Mutex, atomic::{AtomicUsize, Ordering}}, thread::{spawn, yield_now}};
use log::info;
use rand::Rng;
use rand::rngs::OsRng;
use rand_core::{CryptoRng, Error, RngCore};
use rand_core::block::{BlockRng64, BlockRngCore};

struct SharedBufferRngInner<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng> {
    started_reading: AtomicUsize,
    started_writing: AtomicUsize,
    finished_writing: AtomicUsize,
    buffer: [Mutex<[u64; WORDS_PER_SEED]>; SEEDS_CAPACITY],
    // SAFETY: only accessed by one thread
    inner: SyncUnsafeCell<T>,
}

unsafe impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send> Sync
for SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T> {}

pub struct DefaultableArray<const N: usize, T>([T; N]);

impl <const N: usize, T: Default + Copy> Default for DefaultableArray<N, T> {
    fn default() -> Self {
        DefaultableArray([T::default(); N])
    }
}

impl <const N: usize, T> AsMut<[T]> for DefaultableArray<N, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl <const N: usize, T> AsRef<[T]> for DefaultableArray<N, T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send>
SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T> {
    fn fill(&self) -> bool {
        let write_start = self.started_writing.fetch_add(SEEDS_CAPACITY, Ordering::SeqCst);
        let read = self.started_reading.load(Ordering::SeqCst);
        if write_start >= read {
            // Buffer is full
            self.started_writing.fetch_sub(SEEDS_CAPACITY, Ordering::SeqCst);
            return false;
        }
        let mut write_end = write_start + SEEDS_CAPACITY;
        let mut actual_size = SEEDS_CAPACITY;
        if write_end > read + SEEDS_CAPACITY {
            // Short write
            actual_size -= write_end - (read + SEEDS_CAPACITY);
            self.started_writing.fetch_sub(SEEDS_CAPACITY - actual_size, Ordering::SeqCst);
            write_end = read + SEEDS_CAPACITY;
        }
        let write_start_index = write_start % SEEDS_CAPACITY;
        let mut write_end_index = write_end % SEEDS_CAPACITY;
        if write_end_index == 0 {
            write_end_index = SEEDS_CAPACITY;
        }
        let inner = unsafe { self.inner.get().as_mut() }.unwrap();
        // SAFETY: above bounds checks prevent reading bytes while they're still being written
        if write_end_index > write_start_index {
            self.fill_range(write_start_index, write_end_index, inner);
        } else {
            self.fill_range(write_start_index, SEEDS_CAPACITY, inner);
            self.fill_range(0, write_end_index, inner);
        }
        let cmpex = self.finished_writing.compare_exchange(write_start, write_end, Ordering::SeqCst, Ordering::SeqCst);
        if !cmpex.is_ok() {
            panic!("compare_exchange result: {:?}", cmpex)
        }
        true
    }

    fn fill_range(&self, write_start_index: usize, write_end_index: usize, inner: &mut T) {
        self.buffer[write_start_index..write_end_index].iter().for_each(
            |seed| seed.lock().unwrap().iter_mut().for_each(|word| *word
                = inner.next_u64())
        );
    }

    fn read_range(&self, write_start_index: usize, write_end_index: usize, out: &mut [[u64; WORDS_PER_SEED]]) {
        out.iter_mut().zip(&self.buffer[write_start_index..write_end_index]).for_each(
            |(out, seed)| out.copy_from_slice(seed.lock().unwrap().as_slice()));
    }

    fn poll(&self, out: &mut [[u64; WORDS_PER_SEED]]) -> usize {
        let desired_size = out.len().min(SEEDS_CAPACITY);
        if desired_size == 0 {
            return 0;
        }
        let read_start = self.started_reading.fetch_add(desired_size, Ordering::SeqCst);
        let written = self.finished_writing.load(Ordering::SeqCst);
        if read_start >= written {
            // Buffer is empty
            self.started_reading.fetch_sub(desired_size, Ordering::SeqCst);
            return 0;
        }
        let mut read_end = read_start + desired_size;
        let mut actual_size = desired_size;
        if read_end > written {
            // Short read
            actual_size -= read_end - written;
            read_end = written;
            self.started_reading.fetch_sub(desired_size - actual_size, Ordering::SeqCst);
        }
        let read_start = read_start % SEEDS_CAPACITY;
        let mut read_end = read_end % SEEDS_CAPACITY;
        if read_end == 0 {
            read_end = SEEDS_CAPACITY;
        }
        if read_end > read_start {
            self.read_range(read_start, read_end, out);
        } else {
            let bytes_before_wrap = SEEDS_CAPACITY - read_start;
            self.read_range(read_start, SEEDS_CAPACITY, &mut out[0..bytes_before_wrap]);
            self.read_range(0, read_end, &mut out[bytes_before_wrap..]);
        }
        return actual_size;
    }
}

#[derive(Debug)]
pub struct SharedBufferRng<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send>
(Arc<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>>);

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send> Clone for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, T> {
    fn clone(&self) -> Self {
        SharedBufferRng(self.0.clone())
    }
}

pub type SharedBufferRngStd = SharedBufferRng<8, 16, OsRng>;

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send + 'static>
SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, T> {

    pub fn new_master_rng(inner: T) -> BlockRng64<Self> {
        BlockRng64::new(Self::new(inner))
    }

    pub fn new(inner: T) -> Self {
        let inner = Arc::new(SyncUnsafeCell::new(SharedBufferRngInner {
            started_reading: 0.into(),
            started_writing: 0.into(),
            finished_writing: 0.into(),
            buffer: repeat_with(|| Mutex::new([0; WORDS_PER_SEED])).take(SEEDS_CAPACITY).collect::<Vec<_>>().try_into().ok().unwrap(),
            inner: inner.into()
        }));
        let inner_weak = Arc::downgrade(&inner);
        spawn(move || {
            info!("Starting the writer thread for {:?}", inner_weak);
            loop {
                match inner_weak.upgrade() {
                    // SAFETY: This thread is the only one that mutates inner
                    Some(inner) => unsafe {
                        let pinned = pin!(inner.as_ref()).get();
                        if !pinned.as_ref().unwrap().fill() {
                            spin_loop();
                        }
                    },
                    None => {
                        info!("Writer thread exiting");
                        return
                    }
                }
            }
        });
        SharedBufferRng(inner)
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send> BlockRngCore
for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, T> {
    type Item = u64;
    type Results = DefaultableArray<WORDS_PER_SEED, u64>;

    fn generate(&mut self, results: &mut Self::Results) {
        let results: &mut [[u64; WORDS_PER_SEED]] = slice::from_mut(&mut results.0).into();
        let inner = unsafe { self.0.get().as_ref().unwrap() };
        loop {
            match inner.poll(results) {
                0 => yield_now(),
                _ => return
            }
        }
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send> CryptoRng for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, T> {}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering::SeqCst;
    use std::sync::{LazyLock};
    use loom::model::Builder;
    use loom::sync::atomic::{AtomicUsize, fence};
    use rand::rngs::StdRng;
    use rand_core::block::{BlockRng64, BlockRngCore};
    use rand_core::SeedableRng;
    use scc::{Bag};
    use super::*;

    const U8_VALUES: usize = u8::MAX as usize + 1;

    struct ByteValuesInOrderRng {
        words_written: AtomicUsize
    }

    impl BlockRngCore for ByteValuesInOrderRng {
        type Item = u64;
        type Results = DefaultableArray<1,u64>;

        fn generate(&mut self, results: &mut Self::Results) {
            let first_word = self.words_written.fetch_add(U8_VALUES, SeqCst);

            results.0.iter_mut().zip(first_word..first_word + U8_VALUES)
                                         .for_each(|(result_word, word_num)| *result_word = word_num as u64);
        }
    }

    #[test]
    fn basic_test() -> Result<(), Error>{
        let shared_seeder = SharedBufferRngStd::new(OsRng::default());
        let client_prng: StdRng = StdRng::from_rng(&mut BlockRng64::new(shared_seeder))?;
        let zero_seed_prng = StdRng::from_seed([0; 32]);
        assert_ne!(client_prng, zero_seed_prng);
        Ok(())
    }

    const WORDS: LazyLock<Bag<u64>> = LazyLock::new(Bag::new);

    #[test_log::test]
    fn loom_test_at_most_once_delivery() {
        const THREADS: usize = 2;
        const ITERS_PER_THREAD: usize = 2;
        let mut builder = Builder::default();
        builder.max_threads = THREADS + 1; // include filler thread
        builder.check(|| {
            let shared_seeder = SharedBufferRng::<8,4,_>::new(BlockRng64::new(
                ByteValuesInOrderRng { words_written: AtomicUsize::new(0)}));
            fence(SeqCst);
            let ths: Vec<_> = (0..THREADS)
                .map(|_| {
                    let mut seeder_clone = BlockRng64::new(shared_seeder.clone());
                    loom::thread::spawn(move || {
                        for _ in 0..ITERS_PER_THREAD {
                            WORDS.push(seeder_clone.next_u64());
                        }
                    })
                })
                .collect();
            for th in ths {
                th.join().unwrap();
            }
            fence(SeqCst);
            let mut words_sorted = Vec::new();
            WORDS.pop_all((), |(), word| words_sorted.push(word));
            let mut words_dedup = words_sorted.clone();
            words_dedup.dedup();
            assert_eq!(words_dedup.len(), words_sorted.len());
        });
    }
}
