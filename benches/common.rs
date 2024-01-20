use bytemuck::cast_slice_mut;
use rand::Rng;
use rand_core::block::BlockRngCore;
use rand_core::OsRng;
use shared_buffer_rng::{SharedBufferRng, WORDS_PER_STD_RNG};

pub const RESEEDING_THRESHOLD: u64 = 1024;

#[derive(Copy, Clone)]
pub struct DefaultableUnalignedArray<const N: usize, T>([T; N]);

impl <const N: usize, T: Default + Copy> Default for DefaultableUnalignedArray<N, T> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

impl<const N: usize, T> AsMut<[T; N]> for DefaultableUnalignedArray<N, T> {
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

impl<const N: usize, T> AsRef<[T; N]> for DefaultableUnalignedArray<N, T> {
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<const N: usize, T> AsRef<[T]> for DefaultableUnalignedArray<N, T> {
    fn as_ref(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<const N: usize, T> AsMut<[T]> for DefaultableUnalignedArray<N, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

pub struct RngBufferCore<const N: usize, T: Rng>(pub T);

impl <const N: usize, T: Rng> BlockRngCore for RngBufferCore<N, T> where [(); WORDS_PER_STD_RNG * N]: {
    type Item = u64;
    type Results = DefaultableUnalignedArray<{ WORDS_PER_STD_RNG * N }, u64>;

    fn generate(&mut self, results: &mut Self::Results) {
        self.0.fill_bytes(cast_slice_mut(results.as_mut()));
    }
}

pub type BenchmarkSharedBufferRng<const N: usize> = SharedBufferRng<WORDS_PER_STD_RNG, N, OsRng>;
