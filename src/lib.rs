#[cfg(not(loom))]
use std::{sync::{Arc, atomic::{AtomicUsize}}, thread::{spawn, yield_now}, cell::UnsafeCell};
use std::fmt::Debug;
use std::slice;
use async_channel::{Receiver};
use log::{error, info};
use rand::Rng;
use rand::rngs::OsRng;
use rand_core::{CryptoRng, RngCore};
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

    pub(crate) unsafe fn with_mut<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
        f(self.0.get().as_mut().unwrap())
    }

}

impl <T> From<T> for SyncUnsafeCell<T> {
    fn from(value: T) -> Self {
        SyncUnsafeCell(UnsafeCell::new(value))
    }
}

unsafe impl <T> Sync for SyncUnsafeCell<T> {}

#[derive(Debug)]
struct SharedBufferRngInner<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize> {
    receiver: Receiver<[u64; WORDS_PER_SEED]>,
}

unsafe impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize> Sync
for SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY> {}

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

#[derive(Debug)]
pub struct SharedBufferRng<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize>
(Arc<SyncUnsafeCell<SharedBufferRngInner<WORDS_PER_SEED, SEEDS_CAPACITY>>>);

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize> Clone for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY> {
    fn clone(&self) -> Self {
        SharedBufferRng(self.0.clone())
    }
}

pub type SharedBufferRngStd = SharedBufferRng<8, 16>;

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize>
SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY> {
    pub fn new_master_rng<T: Rng + Send + Debug + 'static>(inner: T) -> BlockRng64<Self> {
        BlockRng64::new(Self::new(inner))
    }

    pub fn new<T: Rng + Send + Debug + 'static>(inner_inner: T) -> Self {
        let (sender, receiver) = async_channel::bounded(SEEDS_CAPACITY);
        info!("Creating a SharedBufferRngInner for {:?}", inner_inner);
        let inner_inner = SyncUnsafeCell::from(inner_inner);
        let inner = Arc::new(SyncUnsafeCell::from(SharedBufferRngInner {
            receiver
        }));
        let weak_sender = sender.downgrade();
        spawn(move || {
            let mut seed_from_source = [0; WORDS_PER_SEED];
            loop {
                match weak_sender.upgrade() {
                    None => return,
                    Some(sender) => unsafe {
                        inner_inner.with_mut(
                            |inner_inner| seed_from_source.iter_mut().for_each(
                                |word| *word = inner_inner.next_u64()));
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
        SharedBufferRng(inner)
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize> BlockRngCore
for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY> {
    type Item = u64;
    type Results = DefaultableArray<WORDS_PER_SEED, u64>;

    fn generate(&mut self, results: &mut Self::Results) {
        let results: &mut [[u64; WORDS_PER_SEED]] = slice::from_mut(&mut results.0).into();
        unsafe {
            loop {
                self.0.with(|inner|
                    match inner.receiver.recv_blocking() {
                        Ok(seed) => {
                            results[0].copy_from_slice(&seed);
                            return;
                        },
                        Err(e) => {
                            error!("Error from recv_blocking(): {}", e);
                            yield_now();
                        }
                    }
                );
            }
        }
    }
}

impl <const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize> CryptoRng for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY> {}

#[cfg(test)]
mod tests {
    #[cfg(not(loom))]
    use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
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
}
