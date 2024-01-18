use core::mem::size_of;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput, PlotConfiguration, AxisScale};
use rand::rngs::adapter::ReseedingRng;
use rand_chacha::ChaCha12Core;
use rand_core::{OsRng, RngCore, SeedableRng};
use shared_buffer_rng::{SharedBufferRng, WORDS_PER_STD_RNG};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;
use std::thread::spawn;
use std::time::Duration;

const RESEEDING_THRESHOLD: u64 = 1024;

type BenchmarkSharedBufferRng<const N: usize> = SharedBufferRng<WORDS_PER_STD_RNG, N, OsRng>;

macro_rules! single_thread_bench {
    ($group:expr, $n:expr) => {
        let mut reseeding_from_shared =
            BenchmarkSharedBufferRng::<$n>::new(OsRng::default()).new_standard_rng(RESEEDING_THRESHOLD);
        $group.bench_function(BenchmarkId::new("With SharedBufferRng", format!("buffer size {:04}", $n)), |b| {
            b.iter(|| black_box(reseeding_from_shared.next_u64()))
        });
        drop(reseeding_from_shared);
    };
}

fn benchmark_single_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Thread");
    group.throughput(Throughput::Bytes(size_of::<u64>() as u64));
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
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
    group.bench_function("With OsRng", |b| {
        b.iter(|| black_box(reseeding_from_os.next_u64()))
    });
    group.finish();
}

static FINISHED: AtomicBool = AtomicBool::new(false);

macro_rules! benchmark_contended {
    ($group:expr, $n:expr) => {
        let cpus = num_cpus::get();
        benchmark_contended!($group, $n, cpus - 1);
        benchmark_contended!($group, $n, cpus);
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
        $group.bench_function(BenchmarkId::new("With SharedBufferRng", format!("{:02} threads, buffer size {:04}", $threads, $n)), |b| {
            b.iter(|| black_box(reseeding_from_shared.next_u64()))
        });
        FINISHED.store(true, SeqCst);
        background_threads
            .into_iter()
            .for_each(|handle| handle.join().unwrap());
        FINISHED.store(false, SeqCst);
    };
}

fn benchmark_contended(c: &mut Criterion) {
    let mut group = c.benchmark_group("Contended");
    group.throughput(Throughput::Bytes(size_of::<u64>() as u64));
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
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
    let num_cpus = num_cpus::get();
    vec![num_cpus, num_cpus - 1].into_iter().for_each(|num_threads| {
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
        group.bench_function(BenchmarkId::new("With SharedBufferRng", format!("{:02} threads", num_threads)), |b| {
            b.iter(|| black_box(reseeding_from_os.next_u64()))
        });
        FINISHED.store(true, SeqCst);
        background_threads
            .into_iter()
            .for_each(|handle| handle.join().unwrap());
    });
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().confidence_level(0.99).sample_size(2048).warm_up_time(Duration::from_secs(10));
    targets = benchmark_single_thread, benchmark_contended
}
criterion_main!(benches);
