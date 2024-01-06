use std::{sync::{Arc, atomic::{AtomicUsize}}, thread::{spawn, yield_now}};
use std::fmt::Debug;
use std::slice;
use async_channel::{Receiver, Sender};
use log::{info};
use rand::Rng;
use rand::rngs::OsRng;
use rand_core::{CryptoRng, RngCore};
use rand_core::block::{BlockRng64, BlockRngCore};

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

#[derive(Clone, Debug)]
pub struct SharedBufferRng<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize> {
    // Needed to keep the weak sender reachable as long as the receiver is strongly reachable
    sender: Arc<Sender<[u64; WORDS_PER_SEED]>>,
    receiver: Receiver<[u64; WORDS_PER_SEED]>
}

pub type SharedBufferRngStd = SharedBufferRng<8, 16>;

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize>
SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY> {
    pub fn new_master_rng<T: Rng + Send + Debug + 'static>(inner: T) -> BlockRng64<Self> {
        BlockRng64::new(Self::new(inner))
    }

    pub fn new<T: Rng + Send + Debug + 'static>(mut inner_inner: T) -> Self {
        let (sender, receiver) = async_channel::bounded(SEEDS_CAPACITY);
        info!("Creating a SharedBufferRngInner for {:?}", inner_inner);
        let inner = receiver.clone();
        let weak_sender = sender.clone().downgrade();
        spawn(move || {
            let mut seed_from_source = [0; WORDS_PER_SEED];
            loop {
                match weak_sender.upgrade() {
                    None => return,
                    Some(sender) => unsafe {
                        seed_from_source.iter_mut().for_each(
                                |word| *word = inner_inner.next_u64());
                        while !sender.send_blocking(seed_from_source).is_ok() {
                            if weak_sender.upgrade().is_none() {
                                return;
                            }
                            yield_now();
                        }
                    }
                }
            }
        });
        while (receiver.is_empty()) {
            yield_now();
        }
        SharedBufferRng {
            receiver: inner,
            sender: sender.into()
        }
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize> BlockRngCore
for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY> {
    type Item = u64;
    type Results = DefaultableArray<WORDS_PER_SEED, u64>;

    fn generate(&mut self, results: &mut Self::Results) {
        let results: &mut [[u64; WORDS_PER_SEED]] = slice::from_mut(&mut results.0).into();
        match self.receiver.recv_blocking() {
            Ok(seed) => {
                results[0].copy_from_slice(&seed);
                return;
            },
            Err(e) => panic!("Error from recv_blocking(): {}", e)
        }
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize> CryptoRng for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY> {}

#[cfg(test)]
mod tests {
    use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::sync::OnceLock;
    use rand_core::block::{BlockRng64, BlockRngCore};
    use rand_core::{Error};
    use scc::Bag;
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

    static WORDS: OnceLock<Bag<u64>> = OnceLock::new();

    #[test_log::test]
    fn loom_test_at_most_once_delivery() {
        WORDS.get_or_init(Bag::new);
        use rand::RngCore;
        const THREADS: usize = 2;
        const ITERS_PER_THREAD: usize = 1;
        let seeder: SharedBufferRng::<8,4> = SharedBufferRng::new(BlockRng64::new(
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
