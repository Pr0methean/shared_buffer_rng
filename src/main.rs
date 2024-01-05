#![feature(lazy_cell)]

use std::sync::atomic::Ordering::SeqCst;
use std::sync::LazyLock;
use core::sync::atomic::{AtomicUsize, fence};
use rand_core::RngCore;
use rand_core::block::{BlockRng, BlockRngCore};
use scc::Bag;
use shared_buffer_rng::SharedBufferRng;
use std::thread;

const U8_VALUES: usize = u8::MAX as usize + 1;

struct ByteValuesInOrderRng {
    words_written: AtomicUsize
}

struct DefaultableByteArray([u32; U8_VALUES]);

impl Default for DefaultableByteArray {
    fn default() -> Self {
        DefaultableByteArray([0; U8_VALUES])
    }
}

impl AsMut<[u32]> for DefaultableByteArray {
    fn as_mut(&mut self) -> &mut [u32] {
        self.0.as_mut()
    }
}

impl AsRef<[u32]> for DefaultableByteArray {
    fn as_ref(&self) -> &[u32] {
        self.0.as_ref()
    }
}

impl BlockRngCore for ByteValuesInOrderRng {
    type Item = u32;
    type Results = DefaultableByteArray;

    fn generate(&mut self, results: &mut Self::Results) {
        let first_word = self.words_written.fetch_add(U8_VALUES, SeqCst);

        results.0.iter_mut().zip(first_word..first_word + U8_VALUES)
            .for_each(|(result_word, word_num)| *result_word = word_num as u32);
    }
}

const WORDS: LazyLock<Bag<u32>> = LazyLock::new(Bag::new);
fn main() {
    const THREADS: usize = 4;
    const ITERS_PER_THREAD: usize = 256;
    let shared_seeder = SharedBufferRng::<128,_>::new(BlockRng::new(
        ByteValuesInOrderRng { words_written: AtomicUsize::new(0)}));
    fence(SeqCst);
    let ths: Vec<_> = (0..THREADS)
        .map(|_| {
            let mut seeder_clone = shared_seeder.clone();
            thread::spawn(move || {
                for _ in 0..ITERS_PER_THREAD {
                    let next_word = seeder_clone.next_u32();
                    WORDS.push(next_word);
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
}