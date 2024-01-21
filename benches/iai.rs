#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::sync::atomic::{AtomicU64};
use iai::black_box;
use paste::paste;
use rand::rngs::adapter::ReseedingRng;
use rand_chacha::{ChaCha12Core};
use rand_core::{OsRng, RngCore, SeedableRng};
use std::sync::atomic::Ordering::SeqCst;
use std::thread::spawn;
use rand_core::block::BlockRng64;
mod common;
use common::{BenchmarkSharedBufferRng, RngBufferCore, RESEEDING_THRESHOLD};

macro_rules! single_thread_bench_iai {
    ($n:expr) => {
        paste! {
            fn [< single_thread_bench_ $n _shared_buffer >]() {
                let mut reseeding_from_shared =
                    BenchmarkSharedBufferRng::<$n>::new(OsRng::default()).new_standard_rng(RESEEDING_THRESHOLD);
                (0..(2 * RESEEDING_THRESHOLD * $n.max(1))).for_each(|_| {
                    let _ = black_box(reseeding_from_shared.next_u64());
                })
            }

            fn [< single_thread_bench_ $n _local_buffer >]() {
                let mut buffer = BlockRng64::new(RngBufferCore::<$n, OsRng>(OsRng::default()));
                let mut seed = [0u8; 32];
                buffer.fill_bytes(&mut seed);
                let mut reseeding_from_buffer = ReseedingRng::new(ChaCha12Core::from_seed(seed), RESEEDING_THRESHOLD, buffer);
                (0..(2 * RESEEDING_THRESHOLD * $n.max(1))).for_each(|_| {
                    let _ = black_box(reseeding_from_buffer.next_u64());
                })
            }
        }
    }
}

macro_rules! contended_bench_iai {
    ($n:expr) => {
        contended_bench_iai!($n, num_cpus::get_physical());
    };
    ($n:expr, $threads:expr) => {
        paste! {
            fn [< contended_bench_ $n _shared_buffer >]() {
                let iterations_left = Arc::new(AtomicU64::new(2 * RESEEDING_THRESHOLD * $n.max(1)));
                let root = BenchmarkSharedBufferRng::<$n>::new(OsRng::default());
                let rngs: Vec<_> = (0..$threads)
                    .map(|_| root.new_standard_rng(RESEEDING_THRESHOLD))
                    .collect();
                let main_thread_rng = rngs.pop().unwrap();
                drop(root);
                let background_threads: Vec<_> = rngs.into_iter()
                    .map(|mut rng| {
                        let iterations_left = iterations_left.clone();
                        spawn(move || {
                            while iterations_left.fetch_sub(1, SeqCst) > 0 {
                                black_box(rng.next_u64());
                            }
                        })
                    })
                    .collect();
                while iterations_left.fetch_sub(1, SeqCst) > 0 {
                    black_box(main_thread_rng.next_u64());
                }
            }

            fn [< contended_bench_ $n _local_buffer >]() {
                let iterations_left = Arc::new(AtomicU64::new(2 * RESEEDING_THRESHOLD * $n.max(1)));
                let rngs: Vec<_> = (0..$threads)
                    .map(|_| {
                        let mut buffer = BlockRng64::new(RngBufferCore::<$n, OsRng>(OsRng::default()));
                        let mut seed = [0u8; 32];
                        buffer.fill_bytes(&mut seed);
                        ReseedingRng::new(ChaCha12Core::from_seed(seed), RESEEDING_THRESHOLD, buffer)
                    })
                    .collect();
                let main_thread_rng = rngs.pop().unwrap();
                let background_threads: Vec<_> = rngs.into_iter()
                    .map(|mut rng| {
                        let iterations_left = iterations_left.clone();
                        spawn(move || {
                            while ITERATIONS_LEFT.fetch_sub(1, SeqCst) > 0 {
                                black_box(rng.next_u64());
                            }
                        })
                    })
                    .collect();
                while iterations_left.fetch_sub(1, SeqCst) > 0 {
                    black_box(main_thread_rng.next_u64());
                }
            }
        }
    };
}

single_thread_bench_iai!(0);
single_thread_bench_iai!(1);
single_thread_bench_iai!(2);
single_thread_bench_iai!(4);
single_thread_bench_iai!(8);
single_thread_bench_iai!(16);
single_thread_bench_iai!(32);
single_thread_bench_iai!(64);
single_thread_bench_iai!(128);

contended_bench_iai!(0);
contended_bench_iai!(1);
contended_bench_iai!(2);
contended_bench_iai!(4);
contended_bench_iai!(8);
contended_bench_iai!(16);
contended_bench_iai!(32);
contended_bench_iai!(64);
contended_bench_iai!(128);

iai::main!(single_thread_bench_0_shared_buffer,
    single_thread_bench_1_shared_buffer,
    single_thread_bench_2_shared_buffer, single_thread_bench_2_local_buffer,
    single_thread_bench_4_shared_buffer, single_thread_bench_4_local_buffer,
    single_thread_bench_8_shared_buffer, single_thread_bench_8_local_buffer,
    contended_bench_0_shared_buffer,
    contended_bench_1_shared_buffer, contended_bench_1_local_buffer,
    contended_bench_2_shared_buffer, contended_bench_2_local_buffer,
    contended_bench_4_shared_buffer, contended_bench_4_local_buffer,
    contended_bench_8_shared_buffer, contended_bench_8_local_buffer,
);