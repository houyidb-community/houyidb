//! 冷加载基准：构建一次快照，再分别计时 eager / view 模式的加载与首次查询。
//! 真机 A/B 方法与结果见 `docs/HOUYIDB_MOBILE_BENCH.md`。
//! EN: Cold-load benchmark: build a snapshot once, then time eager and view
//! loads plus the first query. The on-device A/B method and results are in
//! `docs/HOUYIDB_MOBILE_BENCH.md`.

use houyi_mobile::db::MobileDB;
use houyi_vector::cluster::FlatStorageDtype;
use houyi_vector::DistanceMetric;
use rand::Rng;
use rand::SeedableRng;
use std::time::Instant;

const N: usize = 100_000;
const DIMS: usize = 128;

fn synth() -> (Vec<u64>, Vec<f32>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let clusters = 256usize;
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..DIMS).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let mut ids = Vec::with_capacity(N);
    let mut vecs = Vec::with_capacity(N * DIMS);
    for i in 0..N {
        ids.push(i as u64);
        let c = &centers[i % clusters];
        for &v in c.iter() {
            vecs.push(v + rng.gen_range(-0.05f32..0.05));
        }
    }
    (ids, vecs)
}

fn dtype_of(s: &str) -> FlatStorageDtype {
    match s {
        "f16" => FlatStorageDtype::F16,
        "i8" => FlatStorageDtype::I8,
        other => panic!("unknown dtype {other}"),
    }
}

fn vm_hwm_mb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") => {
            let path = &args[2];
            let dtype = dtype_of(&args[3]);
            let (ids, vectors) = synth();
            let db = MobileDB::new();
            db.get_or_create_with_config(
                "faces",
                DIMS,
                "flat-clustered",
                DistanceMetric::Cosine,
                dtype,
            )
            .unwrap();
            let t = Instant::now();
            db.insert_batch("faces", &ids, &vectors, DIMS).unwrap();
            let insert_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            db.build("faces").unwrap();
            let build_ms = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            db.save_to(path).unwrap();
            let save_ms = t.elapsed().as_secs_f64() * 1e3;
            let bytes = std::fs::metadata(path).unwrap().len();
            println!(
                "insert_ms={insert_ms:.1},build_ms={build_ms:.1},save_ms={save_ms:.1},bytes={bytes}"
            );
        }
        Some("dump") => {
            let path = &args[2];
            let (_, vectors) = synth();
            use std::io::Write;
            let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
            f.write_all(&(N as u64).to_le_bytes()).unwrap();
            f.write_all(&(DIMS as u64).to_le_bytes()).unwrap();
            for v in &vectors {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
            f.flush().unwrap();
            println!("dumped n={N} dims={DIMS} bytes={}", 16 + vectors.len() * 4);
        }
        Some("load") => {
            let path = &args[2];
            let iters: usize = args[3].parse().unwrap();
            let view = args.get(4).is_some_and(|s| s == "view");
            let query = vec![0.013f32; DIMS];
            for _ in 0..iters {
                let t = Instant::now();
                let db = if view {
                    MobileDB::load_view_from(path).unwrap()
                } else {
                    MobileDB::load_from(path).unwrap()
                };
                let load_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = Instant::now();
                let top = db.search("faces", &query, 3).unwrap();
                let first_us = t.elapsed().as_secs_f64() * 1e6;
                println!(
                    "mode={},load_ms={load_ms:.1},first_us={first_us:.0},total_ms={:.1},count={},hwm_mb={:.1},top={:?}",
                    if view { "view" } else { "eager" },
                    load_ms + first_us / 1000.0,
                    db.count("faces").unwrap(),
                    vm_hwm_mb().unwrap_or(0.0),
                    top
                );
            }
        }
        _ => {
            eprintln!(
                "usage: loadbench build <path> <f16|i8> | dump <base.bin> | load <path> <iters>"
            );
            std::process::exit(2);
        }
    }
}
