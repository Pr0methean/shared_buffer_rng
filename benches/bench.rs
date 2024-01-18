use core::mem::size_of;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::rngs::adapter::ReseedingRng;
use rand_chacha::ChaCha12Core;
use rand_core::{OsRng, RngCore, SeedableRng};
use shared_buffer_rng::{rng_from_default_buffer, SharedBufferRngStd};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;
use std::thread::spawn;

const RESEEDING_THRESHOLD: u64 = 1024;

fn benchmark_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Thread");
    group.throughput(Throughput::Bytes(size_of::<u64>() as u64));
    let mut reseeding_from_shared =
        SharedBufferRngStd::new(OsRng::default()).new_standard_rng(RESEEDING_THRESHOLD);
    group.bench_function("With SharedBufferRngStd", |b| {
        b.iter(|| black_box(reseeding_from_shared.next_u64()))
    });
    drop(reseeding_from_shared);
    let mut reseeding_from_os = ReseedingRng::new(
        ChaCha12Core::from_rng(OsRng::default()).unwrap(),
        RESEEDING_THRESHOLD,
        OsRng::default(),
    );
    group.bench_function("With OsRng", |b| {
        b.iter(|| black_box(reseeding_from_os.next_u64()))
    });
    group.finish();
}

static FINISHED: AtomicBool = AtomicBool::new(false);

fn benchmark_contended(c: &mut Criterion) {
    let root = SharedBufferRngStd::new(OsRng::default());
    let mut group = c.benchmark_group("Contended");
    group.throughput(Throughput::Bytes(size_of::<u64>() as u64));
    let background_threads: Vec<_> = (0..(num_cpus::get() - 1))
        .map(|_| {
            let mut rng = root.new_standard_rng(RESEEDING_THRESHOLD);
            spawn(move || {
                while !FINISHED.load(SeqCst) {
                    black_box(rng.next_u64());
                }
            })
        })
        .collect();
    drop(root);
    let mut reseeding_from_shared = rng_from_default_buffer(RESEEDING_THRESHOLD);
    group.bench_function("With SharedBufferRngStd", |b| {
        b.iter(|| black_box(reseeding_from_shared.next_u64()))
    });
    FINISHED.store(true, SeqCst);
    background_threads
        .into_iter()
        .for_each(|handle| handle.join().unwrap());
    FINISHED.store(false, SeqCst);
    let background_threads: Vec<_> = (0..(num_cpus::get() - 1))
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
    group.bench_function("With OsRng", |b| {
        b.iter(|| black_box(reseeding_from_os.next_u64()))
    });
    FINISHED.store(true, SeqCst);
    group.finish();
    background_threads
        .into_iter()
        .for_each(|handle| handle.join().unwrap());
}

criterion_group! {
    name = benches;
    config = Criterion::default().confidence_level(0.99).sample_size(1000);
    targets = benchmark_single_thread, benchmark_contended
}
criterion_main!(benches);
