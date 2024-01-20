#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use core::mem::size_of;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::adapter::ReseedingRng;
use rand_chacha::{ChaCha12Core};
use rand_core::{OsRng, RngCore, SeedableRng};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;
use std::thread::spawn;
use rand_core::block::BlockRng64;
mod lib;
use lib::{BenchmarkSharedBufferRng, RngBufferCore, RESEEDING_THRESHOLD};

macro_rules! single_thread_bench {
    ($group:expr, $n:expr) => {
        let mut reseeding_from_shared =
            BenchmarkSharedBufferRng::<$n>::new(OsRng::default()).new_standard_rng(RESEEDING_THRESHOLD);
        $group.bench_with_input(BenchmarkId::new("SharedBufferRng", $n),
        &$n, |b, _| b.iter(|| black_box(reseeding_from_shared.next_u64())));
        drop(reseeding_from_shared);
        if $n > 1 {
            let mut buffer = BlockRng64::new(RngBufferCore::<$n, OsRng>(OsRng::default()));
            let mut seed = [0u8; 32];
            buffer.fill_bytes(&mut seed);
            let mut reseeding_from_buffer = ReseedingRng::new(ChaCha12Core::from_seed(seed), RESEEDING_THRESHOLD, buffer);
            $group.bench_with_input(BenchmarkId::new("RngBufferCore", $n),
            &$n, |b, _| b.iter(|| black_box(reseeding_from_buffer.next_u64())));
        }
    };
}

fn benchmark_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Thread");
    group.throughput(Throughput::Bytes(size_of::<u64>() as u64));
    single_thread_bench!(group, 0);
    single_thread_bench!(group, 1);
    single_thread_bench!(group, 2);
    single_thread_bench!(group, 4);
    single_thread_bench!(group, 8);
    single_thread_bench!(group, 16);
    single_thread_bench!(group, 32);
    single_thread_bench!(group, 64);
    single_thread_bench!(group, 128);
    single_thread_bench!(group, 256);
    single_thread_bench!(group, 512);
    single_thread_bench!(group, 1024);
    let mut reseeding_from_os = ReseedingRng::new(
        ChaCha12Core::from_rng(OsRng::default()).unwrap(),
        RESEEDING_THRESHOLD,
        OsRng::default(),
    );
    group.bench_function("OsRng", |b| {
        b.iter(|| black_box(reseeding_from_os.next_u64()))
    });
    group.finish();
}

static FINISHED: AtomicBool = AtomicBool::new(false);

macro_rules! benchmark_contended {
    ($group:expr, $n:expr) => {
        benchmark_contended!($group, $n, num_cpus::get_physical());
    };

    ($group:expr, $n:expr, $threads:expr) => {
        let root = BenchmarkSharedBufferRng::<$n>::new(OsRng::default());
        let rngs: Vec<_> = (0..($threads - 1))
            .map(|_| root.new_standard_rng(RESEEDING_THRESHOLD))
            .collect();
        let background_threads: Vec<_> = rngs.into_iter()
            .map(|mut rng| {
                spawn(move || {
                    while !FINISHED.load(SeqCst) {
                        black_box(rng.next_u64());
                    }
                })
            })
            .collect();
        let mut reseeding_from_shared = root.new_standard_rng(RESEEDING_THRESHOLD);
        drop(root);
        $group.bench_with_input(BenchmarkId::new(format!("SharedBufferRng, {:02} threads", $threads), $n),
            &$n, |b, _| b.iter(|| black_box(reseeding_from_shared.next_u64())));
        FINISHED.store(true, SeqCst);
        background_threads
            .into_iter()
            .for_each(|handle| handle.join().unwrap());
        FINISHED.store(false, SeqCst);
        if $n > 1 {
            let rngs: Vec<_> = (0..($threads - 1))
                .map(|_| {
                    let mut buffer = BlockRng64::new(RngBufferCore::<$n, OsRng>(OsRng::default()));
                    let mut seed = [0u8; 32];
                    buffer.fill_bytes(&mut seed);
                    ReseedingRng::new(ChaCha12Core::from_seed(seed), RESEEDING_THRESHOLD, buffer)
                })
                .collect();
            let background_threads: Vec<_> = rngs.into_iter()
                .map(|mut rng| {
                    spawn(move || {
                        while !FINISHED.load(SeqCst) {
                            black_box(rng.next_u64());
                        }
                    })
                })
                .collect();
            let mut buffer = BlockRng64::new(RngBufferCore::<$n, OsRng>(OsRng::default()));
            let mut seed = [0u8; 32];
            buffer.fill_bytes(&mut seed);
            let mut reseeding_from_buffer = ReseedingRng::new(ChaCha12Core::from_seed(seed), RESEEDING_THRESHOLD, buffer);
            $group.bench_with_input(BenchmarkId::new(format!("RngBufferCore, {:02} threads", $threads), $n),
            &$n, |b, _| b.iter(|| black_box(reseeding_from_buffer.next_u64())));
            FINISHED.store(true, SeqCst);
            background_threads
                .into_iter()
                .for_each(|handle| handle.join().unwrap());
            FINISHED.store(false, SeqCst);
        }
    };
}

fn benchmark_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("Contended");
    group.throughput(Throughput::Bytes(size_of::<u64>() as u64));
    benchmark_contended!(group, 0);
    benchmark_contended!(group, 1);
    benchmark_contended!(group, 2);
    benchmark_contended!(group, 4);
    benchmark_contended!(group, 8);
    benchmark_contended!(group, 16);
    benchmark_contended!(group, 32);
    benchmark_contended!(group, 64);
    benchmark_contended!(group, 128);
    benchmark_contended!(group, 256);
    benchmark_contended!(group, 512);
    benchmark_contended!(group, 1024);
    let num_threads = num_cpus::get_physical();
    let background_threads: Vec<_> = (0..(num_threads - 1))
        .map(|_| {
            let mut rng = ReseedingRng::new(
                ChaCha12Core::from_rng(OsRng::default()).unwrap(),
                RESEEDING_THRESHOLD,
                OsRng::default(),
            );
            spawn(move || {
                while !FINISHED.load(SeqCst) {
                    black_box(rng.next_u64());
                }
            })
        })
        .collect();
    let mut reseeding_from_os = ReseedingRng::new(
        ChaCha12Core::from_rng(OsRng::default()).unwrap(),
        RESEEDING_THRESHOLD,
        OsRng::default(),
    );
    group.bench_function(format!("OsRng, {:02} threads", num_threads), |b| {
        b.iter(|| black_box(reseeding_from_os.next_u64()))
    });
    FINISHED.store(true, SeqCst);
    background_threads
        .into_iter()
        .for_each(|handle| handle.join().unwrap());
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().confidence_level(0.99).sample_size(4096);
    targets = benchmark_single_thread, benchmark_contended
}
criterion_main!(benches);
