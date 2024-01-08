use std::sync::Arc;
use core::fmt::Debug;
use core::marker::PhantomData;
use std::sync::OnceLock;
use std::thread::Builder;
use aligned::{A64, Aligned};
use async_channel::{Receiver, Sender};
use log::{error, info};
use rand::Rng;
use rand::rngs::{OsRng};
use rand_core::{CryptoRng, Error, RngCore, SeedableRng};
use rand_core::block::{BlockRng64, BlockRngCore};
use std::cell::{UnsafeCell};
use std::rc::Rc;
use rand::rngs::adapter::ReseedingRng;
use rand_chacha::ChaCha12Core;

pub struct DefaultableAlignedArray<const N: usize, T>(Aligned<A64, [T; N]>);

impl <const N: usize, T: Default + Copy> Default for DefaultableAlignedArray<N, T> {
    fn default() -> Self {
        DefaultableAlignedArray(Aligned([T::default(); N]))
    }
}

impl <const N: usize, T> AsMut<[T]> for DefaultableAlignedArray<N, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl <const N: usize, T> AsRef<[T]> for DefaultableAlignedArray<N, T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}

#[derive(Debug)]
pub struct SharedBufferRng<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType> {
    // Needed to keep the weak sender reachable as long as the receiver is strongly reachable
    _sender: Arc<Sender<Aligned<A64, [u64; WORDS_PER_SEED]>>>,
    receiver: Receiver<Aligned<A64, [u64; WORDS_PER_SEED]>>,
    // Used to determine whether to implement CryptoRng
    _source: PhantomData<SourceType>
}

// Can't derive Clone because that would only work for SourceType: Clone but we don't actually clone the source
impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType> Clone
for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType> {
    fn clone(&self) -> Self {
        SharedBufferRng {
            _sender: self._sender.clone(),
            receiver: self.receiver.clone(),
            _source: self._source
        }
    }
}

pub type SharedBufferRngStd = SharedBufferRng<8, 16, OsRng>;

static DEFAULT_ROOT: OnceLock<SharedBufferRngStd> = OnceLock::new();

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct ThreadLocalSeeder<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType>
(Rc<UnsafeCell<BlockRng64<SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType>>>>);

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType>
ThreadLocalSeeder<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType> {
    fn new(source: SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType>) -> Self {
        ThreadLocalSeeder(Rc::new(UnsafeCell::new(BlockRng64::new(source))))
    }

    fn get_mut(&self) -> &mut BlockRng64<SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType>> {
        // SAFETY: Same as impl RngCore for ThreadRng: https://rust-random.github.io/rand/src/rand/rngs/thread.rs.html
        unsafe {
            self.0.get().as_mut().unwrap()
        }
    }
}

thread_local! {
    static DEFAULT_FOR_THREAD: ThreadLocalSeeder<8, 16, OsRng>
        = ThreadLocalSeeder::new(DEFAULT_ROOT.get_or_init(|| SharedBufferRngStd::new(OsRng::default())).clone());
}


impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType>
RngCore for ThreadLocalSeeder<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType> {
    fn next_u32(&mut self) -> u32 {
        self.get_mut().next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.get_mut().next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.get_mut().fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.get_mut().try_fill_bytes(dest)
    }
}

pub fn thread_seeder() -> ThreadLocalSeeder<8, 16, OsRng> {
    DEFAULT_FOR_THREAD.with(ThreadLocalSeeder::clone)
}

pub fn thread_rng() -> ReseedingRng<ChaCha12Core, ThreadLocalSeeder<8, 16, OsRng>> {
    let mut reseeder = thread_seeder();
    let mut seed = <ChaCha12Core as SeedableRng>::Seed::default();
    reseeder.fill_bytes(&mut seed);
    ReseedingRng::new(ChaCha12Core::from_seed(seed), 1 << 16, reseeder)
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize>
SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, OsRng> {
    pub fn new_master_rng() -> BlockRng64<Self> {
        BlockRng64::new(Self::new(OsRng::default()))
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType: Rng + Send + Debug + 'static>
    SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType> {
    pub fn new(mut source: SourceType) -> Self {
        let (sender, receiver) = async_channel::bounded(SEEDS_CAPACITY);
        info!("Creating a SharedBufferRngInner for {:?}", source);
        let inner = receiver.clone();
        let weak_sender = sender.clone().downgrade();
        Builder::new().name(format!("Load seed from {:?} into shared buffer", source)).spawn(move || {
            let mut seed_from_source = Aligned([0; WORDS_PER_SEED]);
            'outer: loop {
                match weak_sender.upgrade() {
                    None => {
                        info!("Detected that a seed channel is no longer open for receiving");
                        return
                    },
                    Some(sender) => {
                        seed_from_source.iter_mut().for_each(
                                |word| *word = source.next_u64());
                        loop {
                            let result = sender.send_blocking(seed_from_source);
                            if result.is_ok() {
                                continue 'outer;
                            } else {
                                if weak_sender.upgrade().is_none() {
                                    info!("Detected (with seed already fetched) that a seed channel is no longer open \
                                    for receiving");
                                } else {
                                    error!("Error writing to shared buffer: {:?}", result);
                                }
                                sender.close();
                                return;
                            }
                        }
                    }
                }
            }
        }).unwrap();
        SharedBufferRng {
            receiver: inner,
            _sender: sender.into(),
            _source: PhantomData::default()
        }
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType> BlockRngCore
for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType> {
    type Item = u64;
    type Results = DefaultableAlignedArray<WORDS_PER_SEED, u64>;

    fn generate(&mut self, results: &mut Self::Results) {
        match self.receiver.recv_blocking() {
            Ok(seed) => {
                *results.0 = *seed;
                return;
            },
            Err(e) => panic!("Error from recv_blocking(): {}", e)
        }
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: CryptoRng> CryptoRng
for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, T> {}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::sync::{OnceLock};
    use rand_core::block::{BlockRng64, BlockRngCore};
    use rand_core::{Error};
    use scc::Bag;
    use std::thread::spawn;
    use super::*;

    const U8_VALUES: usize = u8::MAX as usize + 1;

    #[derive(Debug)]
    struct ByteValuesInOrderRng {
        words_written: AtomicUsize
    }

    impl BlockRngCore for ByteValuesInOrderRng {
        type Item = u64;
        type Results = DefaultableAlignedArray<1,u64>;

        fn generate(&mut self, results: &mut Self::Results) {
            let first_word = self.words_written.fetch_add(U8_VALUES, SeqCst);

            results.0.iter_mut().zip(first_word..first_word + U8_VALUES)
                                         .for_each(|(result_word, word_num)| *result_word = word_num as u64);
        }
    }

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

    static WORDS: OnceLock<Bag<u64>> = OnceLock::new();

    #[test]
    fn test_at_most_once_delivery() {
        use rand_core::RngCore;
        WORDS.get_or_init(Bag::new);
        const THREADS: usize = 2;
        const ITERS_PER_THREAD: usize = 1;
        let seeder: SharedBufferRng<8,4,_> = SharedBufferRng::new(BlockRng64::new(
            ByteValuesInOrderRng { words_written: AtomicUsize::new(0)}));
        let ths: Vec<_> = (0..THREADS)
            .map(|_| {
                let mut seeder_clone = BlockRng64::new(seeder.clone());
                spawn(move || {
                    for _ in 0..ITERS_PER_THREAD {
                        WORDS.get().unwrap().push(seeder_clone.next_u64());
                    }
                })
            })
            .collect();
        for th in ths {
            th.join().unwrap();
        }
        let mut words_sorted = Vec::new();
        WORDS.get().unwrap().pop_all((), |(), word| words_sorted.push(word));
        let mut words_dedup = words_sorted.clone();
        words_dedup.dedup();
        assert_eq!(words_dedup.len(), words_sorted.len());
    }
}
