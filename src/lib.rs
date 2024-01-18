#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(array_chunks)]

use aligned::{Aligned, A64};
use bytemuck::{cast_slice_mut, Pod, Zeroable};
use core::fmt::Debug;
use core::marker::PhantomData;
use core::mem::size_of;
use crossbeam_channel::{bounded, Receiver};
use log::info;
use rand::rngs::adapter::ReseedingRng;
use rand::rngs::OsRng;
use rand::Rng;
use rand_chacha::ChaCha12Core;
use rand_core::block::{BlockRng64, BlockRngCore};
use rand_core::{CryptoRng, RngCore, SeedableRng};
use std::sync::OnceLock;
use std::thread::Builder;

// Alignment is chosen to prevent "false sharing" (i.e. instance A and instance B being part of or straddling the same
// cache line, which would prevent &mut A from being used concurrently with &B or &mut B because only one CPU core can
// have a given cache line in the modified state). All modern x86, ARM, x86-64 and Aarch64 CPUs have 64-byte cache
// lines. TODO: Find a future-proof way to choose the right alignment for obscure architectures.
#[derive(Copy, Clone)]
#[repr(transparent)] // may be necessary to make Bytemuck transmutation safe
pub struct DefaultableAlignedArray<const N: usize, T>(Aligned<A64, [T; N]>);

impl<const N: usize, T: Default + Copy> Default for DefaultableAlignedArray<N, T> {
    fn default() -> Self {
        DefaultableAlignedArray(Aligned([T::default(); N]))
    }
}

impl<const N: usize, T> AsMut<[T; N]> for DefaultableAlignedArray<N, T> {
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<const N: usize, T> AsMut<[T]> for DefaultableAlignedArray<N, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<const N: usize, T> AsRef<[T; N]> for DefaultableAlignedArray<N, T> {
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<const N: usize, T> AsRef<[T]> for DefaultableAlignedArray<N, T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_slice()
    }
}

unsafe impl<const N: usize, T: Zeroable> Zeroable for DefaultableAlignedArray<N, T> {}

unsafe impl<const N: usize, T: Pod> Pod for DefaultableAlignedArray<N, T> {}

/// An RNG that reads from a shared buffer, to which only one thread per buffer will read from a seed source. It will
/// share the buffer with all of its clones. Once this and all clones have been dropped, the source-reading thread will
/// detect this using a [std::sync::Weak] reference and terminate. Since this RNG is used to implement [BlockRngCore]
/// for instances of [BlockRng64], it can produce seeds of any desired size, but a `[u64; [WORDS_PER_SEED]]` will be
/// fastest.
///
/// # Type parameters
/// * [WORDS_PER_SEED] is the seed size to optimize for.
/// * [SEEDS_CAPACITY] is the maximum number of `[u64; [WORDS_PER_SEED]]` instances to keep in memory for future use.
/// * [SourceType] is the type of the seed source; currently it's only used to ensure the [SharedBufferRng] implements
///   [CryptoRng] if and only if the seed source does so.
#[derive(Debug)]
pub struct SharedBufferRng<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType> {
    receiver: Receiver<DefaultableAlignedArray<WORDS_PER_SEED, u64>>,
    // Used to determine whether to implement CryptoRng
    _source: PhantomData<SourceType>,
}

// Can't derive Clone because that would only work for SourceType: Clone but we don't actually clone the source
impl<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType> Clone
    for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType>
{
    /// Returns a new SharedBufferRng view on the same buffer.
    fn clone(&self) -> Self {
        SharedBufferRng {
            receiver: self.receiver.clone(),
            _source: self._source,
        }
    }
}

impl<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType>
    SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType>
{
    pub fn new_seeder(&self) -> BlockRng64<Self> {
        BlockRng64::new(self.clone())
    }

    pub fn new_standard_rng(
        &self,
        reseeding_threshold: u64,
    ) -> ReseedingRng<ChaCha12Core, BlockRng64<SharedBufferRngStd>> {
        let mut reseeder = seeder_from_default_buffer();
        let mut seed = <ChaCha12Core as SeedableRng>::Seed::default();
        reseeder.fill_bytes(&mut seed);
        ReseedingRng::new(ChaCha12Core::from_seed(seed), reseeding_threshold, reseeder)
    }

    pub fn new_default_rng(&self) -> ReseedingRng<ChaCha12Core, BlockRng64<SharedBufferRngStd>> {
        self.new_standard_rng(1 << 16)
    }
}

pub const WORDS_PER_STD_RNG: usize = 4;
pub const SEEDS_PER_STD_BUFFER: usize = 128;

pub type SharedBufferRngStd = SharedBufferRng<WORDS_PER_STD_RNG, SEEDS_PER_STD_BUFFER, OsRng>;

pub type ReseedingRngStd = ReseedingRng<ChaCha12Core, BlockRng64<SharedBufferRngStd>>;

static DEFAULT_ROOT: OnceLock<SharedBufferRngStd> = OnceLock::new();

fn get_default_root() -> &'static SharedBufferRngStd {
    DEFAULT_ROOT.get_or_init(|| SharedBufferRngStd::new(OsRng::default()))
}

/// Gets a seed generator backed by the default instance of [SharedBufferRng].
pub fn seeder_from_default_buffer() -> BlockRng64<SharedBufferRngStd> {
    get_default_root().new_seeder()
}

/// Creates a PRNG that's identical to [rand::thread_rng]() except that it uses the default instance of
/// [SharedBufferRng]. Intended as a drop-in replacement for [rand::thread_rng](). Note that once this has been called,
/// the seed-reading thread will run until it panics or the program exits, because the underlying buffer will be
/// reachable from a static variable.
pub fn rng_from_default_buffer(reseeding_threshold: u64) -> ReseedingRngStd {
    get_default_root().new_standard_rng(reseeding_threshold)
}

pub fn default_rng() -> ReseedingRngStd {
    get_default_root().new_default_rng()
}

impl<
        const WORDS_PER_SEED: usize,
        const SEEDS_CAPACITY: usize,
        SourceType: Rng + Send + Debug + 'static,
    > SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType>
    where [(); WORDS_PER_SEED * size_of::<u64>()]:, [(); WORDS_PER_SEED * SEEDS_CAPACITY * size_of::<u64>()]:
{
    /// Creates an RNG that will have a new dedicated thread reading from [source] into a new buffer that's shared with
    /// all clones of this [SharedBufferRng].
    pub fn new(mut source: SourceType) -> Self {
        let (sender, receiver) = bounded(SEEDS_CAPACITY);
        info!("Creating a SharedBufferRngInner for {:?}", source);
        Builder::new().name(format!("Load seed from {:?} into shared buffer", source)).spawn(move || {
            let mut seeds_from_source = [0u8; WORDS_PER_SEED * SEEDS_CAPACITY * size_of::<u64>()];
            let mut aligned_seed: DefaultableAlignedArray<WORDS_PER_SEED, u64> = DefaultableAlignedArray::default();
            loop {
                source.fill_bytes(&mut seeds_from_source);
                for seed in seeds_from_source.array_chunks::<{ WORDS_PER_SEED * size_of::<u64>() }>() {
                    cast_slice_mut(aligned_seed.as_mut()).copy_from_slice(&*seed);
                    let result = sender.send(aligned_seed);
                    if !result.is_ok() {
                        info!("Detected (with seed already fetched) that a seed channel is no longer open for receiving");
                        return;
                    }
                }
            }
        }).unwrap();
        SharedBufferRng {
            receiver: receiver.into(),
            _source: PhantomData::default(),
        }
    }
}

impl<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, SourceType> BlockRngCore
    for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, SourceType>
{
    type Item = u64;
    type Results = DefaultableAlignedArray<WORDS_PER_SEED, u64>;

    fn generate(&mut self, results: &mut Self::Results) {
        match self.receiver.recv() {
            Ok(seed) => {
                *results = seed;
                return;
            }
            Err(e) => panic!("Error from recv(): {}", e),
        }
    }
}

impl<const WORDS_PER_SEED: usize, const SEEDS_CAPACITY: usize, T: CryptoRng> CryptoRng
    for SharedBufferRng<WORDS_PER_SEED, SEEDS_CAPACITY, T>
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use rand_core::block::{BlockRng64, BlockRngCore};
    use rand_core::Error;
    use scc::Bag;
    use std::sync::OnceLock;
    use std::thread::spawn;

    const U8_VALUES: usize = u8::MAX as usize + 1;

    #[derive(Debug)]
    struct ByteValuesInOrderRng {
        words_written: AtomicUsize,
    }

    impl BlockRngCore for ByteValuesInOrderRng {
        type Item = u64;
        type Results = DefaultableAlignedArray<1, u64>;

        fn generate(&mut self, results: &mut Self::Results) {
            let first_word = self.words_written.fetch_add(U8_VALUES, SeqCst);

            results
                .0
                .iter_mut()
                .zip(first_word..first_word + U8_VALUES)
                .for_each(|(result_word, word_num)| *result_word = word_num as u64);
        }
    }

    #[test]
    fn basic_test() -> Result<(), Error> {
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
        let seeder: SharedBufferRng<8, 4, _> =
            SharedBufferRng::new(BlockRng64::new(ByteValuesInOrderRng {
                words_written: AtomicUsize::new(0),
            }));
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
        WORDS
            .get()
            .unwrap()
            .pop_all((), |(), word| words_sorted.push(word));
        let mut words_dedup = words_sorted.clone();
        words_dedup.dedup();
        assert_eq!(words_dedup.len(), words_sorted.len());
    }
}
