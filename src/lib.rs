use std::iter::repeat_with;
#[cfg(not(loom))]
use std::{sync::{Arc, Weak, Mutex, atomic::{AtomicUsize, Ordering}}, thread::{spawn, yield_now}, cell::UnsafeCell};
use std::fmt::Debug;
use std::slice;
use std::sync::atomic::Ordering::SeqCst;
#[cfg(loom)]
use loom::{sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}}, thread::{spawn, yield_now}, cell::UnsafeCell};
use log::info;
use rand::Rng;
use rand::rngs::OsRng;
use rand_core::{CryptoRng};
use rand_core::block::{BlockRng64, BlockRngCore};

#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct SyncUnsafeCell<T>(UnsafeCell<T>);

// Reimplemented so that we can use same API in loom
impl<T> SyncUnsafeCell<T> {
    #[allow(unused)]
    #[cfg(not(loom))]
    pub(crate) unsafe fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        f(self.0.get().as_ref().unwrap())
    }

    #[allow(unused)]
    #[cfg(loom)]
    pub(crate) unsafe fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        f(self.0.get().deref())
    }

    #[cfg(not(loom))]
    pub(crate) unsafe fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(self.0.get().as_mut().unwrap())
    }

    #[cfg(loom)]
    pub(crate) unsafe fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(self.0.get_mut().deref())
    }
}

impl <T> From<T> for SyncUnsafeCell<T> {
    fn from(value: T) -> Self {
        SyncUnsafeCell(UnsafeCell::new(value))
    }
}

unsafe impl <T> Sync for SyncUnsafeCell<T> {}

#[derive(Debug)]
struct SharedBufferRngInner<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng> {
    started_reading: AtomicUsize,
    started_writing: AtomicUsize,
    finished_writing: AtomicUsize,
    buffer: [Mutex<[u64; WORDS_PER_SEED]>; SEEDS_CAPACITY],
    // SAFETY: only accessed by one thread
    inner: SyncUnsafeCell<T>,
    #[cfg(loom)]
    closed: AtomicBool
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
        // SAFETY: above bounds checks prevent reading bytes while they're still being written
        unsafe {
            self.inner.with_mut(|inner|
                if write_end_index > write_start_index {
                    self.fill_range(write_start_index, write_end_index, inner);
                } else {
                    self.fill_range(write_start_index, SEEDS_CAPACITY, inner);
                    self.fill_range(0, write_end_index, inner);
                });
        }
        true
    }

    fn fill_range(&self, write_start_index: usize, write_end_index: usize, inner: &mut T) {
        self.buffer[write_start_index..write_end_index].iter().for_each(|seed| {
            seed.lock().unwrap().iter_mut().for_each(|word| *word
                = inner.next_u64());
            self.finished_writing.fetch_add(1, Ordering::SeqCst);
        });
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
    #[cfg(loom)]
    #[inline]
    fn downgrade(arc: &Arc<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>>)
                 -> Arc<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>> {
        arc.clone()
    }

    #[cfg(not(loom))]
    #[inline]
    fn downgrade(arc: &Arc<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>>)
                 -> Weak<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>> {
        Arc::downgrade(arc)
    }

    #[cfg(loom)]
    #[inline]
    fn upgrade(arc: &Arc<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>>)
               -> Option<&Arc<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>>> {
        if unsafe { arc.0.with(|inner| (*inner).closed.load(SeqCst)) } {
            None
        } else {
            Some(arc)
        }
    }

    #[cfg(all(loom,test))]
    fn close(&self) {
        unsafe {
            self.0.with(|inner| (*inner).closed.fetch_or(true, SeqCst));
        }
    }

    #[cfg(not(loom))]
    #[inline]
    fn upgrade(weak: &Weak<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>>)
               -> Option<Arc<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY, T>>>> {
        weak.upgrade()
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send + Debug + 'static>
SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, T> {
    pub fn new_master_rng(inner: T) -> BlockRng64<Self> {
        BlockRng64::new(Self::new(inner))
    }

    pub fn new(inner: T) -> Self {
        let inner = Arc::new(SyncUnsafeCell::from(SharedBufferRngInner {
            started_reading: 0.into(),
            started_writing: 0.into(),
            finished_writing: 0.into(),
            buffer: repeat_with(|| Mutex::new([0; WORDS_PER_SEED])).take(SEEDS_CAPACITY).collect::<Vec<_>>().try_into().ok().unwrap(),
            inner: inner.into(),
            #[cfg(loom)]
            closed: false.into()
        }));
        let inner_weak = Self::downgrade(&inner);
        spawn(move || {
            info!("Starting the writer thread for {:?}", inner_weak);
            loop {
                match Self::upgrade(&inner_weak) {
                    // SAFETY: This thread is the only one that mutates inner
                    Some(inner) => unsafe {
                        inner.with(|inner| {
                            if !inner.fill() {
                                yield_now();
                            }
                        })
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
        unsafe {
            loop {
                self.0.with(|inner|
                    match inner.poll(results) {
                        0 => yield_now(),
                        _ => return
                    }
                );
            }
        }
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: Rng + Send> CryptoRng for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, T> {}

#[cfg(test)]
mod tests {
    #[cfg(not(loom))]
    use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    #[cfg(loom)]
    use loom::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use rand_core::block::{BlockRng64, BlockRngCore};
    #[cfg(not(loom))]
    use rand_core::{Error};
    use super::*;

    const U8_VALUES: usize = u8::MAX as usize + 1;

    #[derive(Debug)]
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

    #[cfg(not(loom))]
    #[test]
    fn basic_test() -> Result<(), Error>{
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let shared_seeder = SharedBufferRngStd::new(OsRng::default());
        let client_prng: StdRng = StdRng::from_rng(&mut BlockRng64::new(shared_seeder))?;
        let zero_seed_prng = StdRng::from_seed([0; 32]);
        assert_ne!(client_prng, zero_seed_prng);
        Ok(())
    }

    #[cfg(loom)]
    loom::lazy_static! {
        static ref WORDS: scc::Bag<u64> = scc::Bag::new();
    }
    #[cfg(loom)]
    #[test_log::test]
    fn loom_test_at_most_once_delivery() {
        use loom::model::Builder;
        use rand::RngCore;
        const THREADS: usize = 2;
        const ITERS_PER_THREAD: usize = 1;
        let mut builder = Builder::default();
        builder.max_threads = THREADS + 2; // include filler thread and test thread
        builder.max_branches = 300_000;
        builder.check(|| {
            let seeder: SharedBufferRng::<8,4,_> = SharedBufferRng::new(BlockRng64::new(
                ByteValuesInOrderRng { words_written: AtomicUsize::new(0)}));
            let ths: Vec<_> = (0..THREADS)
                .map(|_| {
                    let mut seeder_clone = BlockRng64::new(seeder.clone());
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
            let mut words_sorted = Vec::new();
            WORDS.pop_all((), |(), word| words_sorted.push(word));
            let mut words_dedup = words_sorted.clone();
            words_dedup.dedup();
            assert_eq!(words_dedup.len(), words_sorted.len());
            seeder.close();
        });
    }
}
