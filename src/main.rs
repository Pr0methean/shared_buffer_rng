#![feature(lazy_cell)]

use std::sync::atomic::Ordering::SeqCst;
use core::sync::atomic::{AtomicUsize};
use std::sync::{LazyLock};
use rand_core::RngCore;
use rand_core::block::{BlockRng64, BlockRngCore};
use scc::Bag;
use shared_buffer_rng::SharedBufferRng;
use std::thread::spawn;

const U8_VALUES: usize = u8::MAX as usize + 1;

#[derive(Debug)]
struct ByteValuesInOrderRng {
    words_written: AtomicUsize
}

struct DefaultableByteArray([u64; U8_VALUES]);

impl Default for DefaultableByteArray {
    fn default() -> Self {
        DefaultableByteArray([0; U8_VALUES])
    }
}

impl AsMut<[u64]> for DefaultableByteArray {
    fn as_mut(&mut self) -> &mut [u64] {
        self.0.as_mut()
    }
}

impl AsRef<[u64]> for DefaultableByteArray {
    fn as_ref(&self) -> &[u64] {
        self.0.as_ref()
    }
}

impl BlockRngCore for ByteValuesInOrderRng {
    type Item = u64;
    type Results = DefaultableByteArray;

    fn generate(&mut self, results: &mut Self::Results) {
        let first_word = self.words_written.fetch_add(U8_VALUES, SeqCst);

        results.0.iter_mut().zip(first_word..first_word + U8_VALUES)
            .for_each(|(result_word, word_num)| *result_word = word_num as u64);
    }
}

const WORDS: LazyLock<Bag<u64>> = LazyLock::new(Bag::new);
fn main() {
    const THREADS: usize = 2;
    const ITERS_PER_THREAD: usize = 1;
    let seeder: SharedBufferRng::<8,4> = SharedBufferRng::new(BlockRng64::new(
        ByteValuesInOrderRng { words_written: AtomicUsize::new(0)}));
    let ths: Vec<_> = (0..THREADS)
        .map(|_| {
            let mut seeder_clone = BlockRng64::new(seeder.clone());
            spawn(move || {
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
}