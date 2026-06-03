use rayon::prelude::*;
use std::collections::HashSet;
use std::env;
use std::fs::{create_dir_all, write, File};
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};

const WINDOW_RADIUS: usize = 4096;

const MIN_ASCII_RUN: usize = 6;
const MAX_ENTROPY: f64 = 7.6;
const MIN_ASCII_RATIO: f64 = 0.20;
const PROGRESS_EVERY: usize = 100;

const THREADS: usize = 24;

struct Pipeline {
    name: &'static str,
    use_rot: bool,
    func: fn(u8, u8, u8) -> u8,
}

struct Beacon {
    name: &'static str,
    bytes: &'static [u8],
    weight: i32,
    primary: bool,
}

fn read_bytes(path: &str) -> Vec<u8> {
    let mut f = File::open(path).expect("failed to open file");

    let mut buf = Vec::new();

    f.read_to_end(&mut buf).expect("failed to read file");

    buf
}

fn deinterleave(
a: &[u8],
b: &[u8],
block: usize,
shift_a: i32,
shift_b: i32,
rev: bool,
) -> Vec<u8> {
    let (mut a, mut b) = if rev { (b, a) } else { (a, b) };

    let mut start_a = 0i32;
    let mut start_b = 0i32;

    if shift_a >= 0 {
        start_a += shift_a;
    } else {
        start_b += -shift_a;
    }

    if shift_b >= 0 {
        start_b += shift_b;
    } else {
        start_a += -shift_b;
    }

    if start_a as usize >= a.len() || start_b as usize >= b.len() {
        return Vec::new();
    }

    a = &a[start_a as usize..];
    b = &b[start_b as usize..];

    let len = a.len().min(b.len());

    let mut out = Vec::with_capacity(len * 2);

    let mut i = 0;

    while i < len {
        let end = (i + block).min(len);

        out.extend_from_slice(&a[i..end]);
        out.extend_from_slice(&b[i..end]);

        i += block;
    }

    out
}

#[inline(always)]
fn nibble_swap(x: u8) -> u8 {
    x.rotate_left(4)
}

#[inline(always)]
fn xor_rot(b: u8, key: u8, rot: u8) -> u8 {
    (b ^ key).rotate_left(rot as u32)
}

#[inline(always)]
fn xor_nib_rot(b: u8, key: u8, rot: u8) -> u8 {
    nibble_swap(b ^ key).rotate_left(rot as u32)
}

#[inline(always)]
fn xor_rot_nib(b: u8, key: u8, rot: u8) -> u8 {
    nibble_swap((b ^ key).rotate_left(rot as u32))
}

#[inline(always)]
fn rot_xor(b: u8, key: u8, rot: u8) -> u8 {
    b.rotate_left(rot as u32) ^ key
}

#[inline(always)]
fn nib_xor(b: u8, key: u8, _: u8) -> u8 {
    nibble_swap(b) ^ key
}

#[inline(always)]
fn add_rot(b: u8, key: u8, rot: u8) -> u8 {
    b.wrapping_add(key).rotate_left(rot as u32)
}

#[inline(always)]
fn sub_rot(b: u8, key: u8, rot: u8) -> u8 {
    b.wrapping_sub(key).rotate_left(rot as u32)
}

#[inline(always)]
fn rot_add(b: u8, key: u8, rot: u8) -> u8 {
    b.rotate_left(rot as u32).wrapping_add(key)
}

#[inline(always)]
fn rot_sub(b: u8, key: u8, rot: u8) -> u8 {
    b.rotate_left(rot as u32).wrapping_sub(key)
}

fn delta_xor(blob: &[u8]) -> Vec<u8> {
    if blob.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(blob.len());

    out.push(blob[0]);

    for i in 1..blob.len() {
        out.push(blob[i] ^ blob[i - 1]);
    }

    out
}

fn delta_sub(blob: &[u8]) -> Vec<u8> {
    if blob.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(blob.len());

    out.push(blob[0]);

    for i in 1..blob.len() {
        out.push(blob[i].wrapping_sub(blob[i - 1]));
    }

    out
}

fn ascii_ratio(win: &[u8]) -> f64 {
    let printable = win.iter().filter(|&&b| (0x20..0x7f).contains(&b)).count();

    printable as f64 / win.len() as f64
}

fn longest_ascii_run(win: &[u8]) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;

    for &b in win {
        if (0x20..0x7f).contains(&b) {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }

    best
}

fn entropy(win: &[u8]) -> f64 {
    let mut counts = [0u32; 256];

    for &b in win {
        counts[b as usize] += 1;
    }

    let n = win.len() as f64;

    let mut ent = 0.0;

    for &c in &counts {
        if c > 0 {
            let p = c as f64 / n;
            ent -= p * p.log2();
        }
    }

    ent
}

fn extract_strings(win: &[u8], min_len: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = Vec::new();

    for &b in win {
        if (0x20..0x7f).contains(&b) {
            cur.push(b);
        } else {
            if cur.len() >= min_len {
                out.push(String::from_utf8_lossy(&cur).to_string());
            }

            cur.clear();
        }
    }

    if cur.len() >= min_len {
        out.push(String::from_utf8_lossy(&cur).to_string());
    }

    out
}

fn portugueseness(strings_lower: &[String]) -> i32 {
    const TERMS: &[(&str, i32)] = &[
    (" de ", 2),
    (" a ", 2),
    (" o ", 2),
    (" que ", 3),
    (" carne", 15),
    (" alma", 15),
    (" desejos", 15),
    (" ideia", 12),
    (" calma", 12),
    (" braços", 15),
    (" presente", 10),
    ];

    let mut score = 0;

    for s in strings_lower {
        for &(needle, weight) in TERMS {
            if s.contains(needle) {
                score += weight;
            }
        }
    }

    score
}

fn streaming_find(
blob: &[u8],
needle: &[u8],
key: u8,
rot: u8,
func: fn(u8, u8, u8) -> u8,
) -> Vec<usize> {
    let mut hits = Vec::new();

    if blob.len() < needle.len() {
        return hits;
    }

    'outer: for i in 0..=(blob.len() - needle.len()) {
        for j in 0..needle.len() {
            let b = func(blob[i + j], key, rot);

            if b != needle[j] {
                continue 'outer;
            }
        }

        hits.push(i);
    }

    hits
}

fn materialize_window(
blob: &[u8],
start: usize,
end: usize,
key: u8,
rot: u8,
func: fn(u8, u8, u8) -> u8,
) -> Vec<u8> {
    blob[start..end]
    .iter()
    .map(|&b| func(b, key, rot))
    .collect()
}

fn main() {
    rayon::ThreadPoolBuilder::new()
    .num_threads(THREADS)
    .build_global()
    .unwrap();

    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!(
        "Usage: arcus_scan \
        <raw6.bin> <raw7.bin>"
        );
        return;
    }

    let raw6 = read_bytes(&args[1]);

    let raw7 = read_bytes(&args[2]);

    create_dir_all("out").expect("failed creating out");

    let pipelines = vec![
    Pipeline {
        name: "xor_rot",
        use_rot: true,
        func: xor_rot,
    },
    Pipeline {
        name: "xor_nib_rot",
        use_rot: true,
        func: xor_nib_rot,
    },
    Pipeline {
        name: "xor_rot_nib",
        use_rot: true,
        func: xor_rot_nib,
    },
    Pipeline {
        name: "rot_xor",
        use_rot: true,
        func: rot_xor,
    },
    Pipeline {
        name: "nib_xor",
        use_rot: false,
        func: nib_xor,
    },
    Pipeline {
        name: "add_rot",
        use_rot: true,
        func: add_rot,
    },
    Pipeline {
        name: "sub_rot",
        use_rot: true,
        func: sub_rot,
    },
    Pipeline {
        name: "rot_add",
        use_rot: true,
        func: rot_add,
    },
    Pipeline {
        name: "rot_sub",
        use_rot: true,
        func: rot_sub,
    },
    ];

    let primary_beacons = vec![
    Beacon {
        name: "flag",
        bytes: b"flag{",
            weight: 500,
            primary: true,
        },
        Beacon {
            name: "ctf",
            bytes: b"ctf=",
            weight: 150,
            primary: true,
        },
        ];

        let secondary_beacons = vec![
        Beacon {
            name: "arcus",
            bytes: b"arcus",
            weight: 120,
            primary: false,
        },
        Beacon {
            name: "augusta",
            bytes: b"augusta",
            weight: 120,
            primary: false,
        },
        Beacon {
            name: "carne",
            bytes: b"carne",
            weight: 60,
            primary: false,
        },
        Beacon {
            name: "alma",
            bytes: b"alma",
            weight: 60,
            primary: false,
        },
        Beacon {
            name: "desejos",
            bytes: b"desejos",
            weight: 60,
            primary: false,
        },
        Beacon {
            name: "ideia",
            bytes: b"ideia",
            weight: 60,
            primary: false,
        },
        Beacon {
            name: "calma",
            bytes: b"calma",
            weight: 60,
            primary: false,
        },
        Beacon {
            name: "PK",
            bytes: b"PK",
            weight: 20,
            primary: false,
        },
        Beacon {
            name: "gzip",
            bytes: b"\x1f\x8b",
            weight: 20,
            primary: false,
        },
        Beacon {
            name: "MZ",
            bytes: b"MZ",
            weight: 20,
            primary: false,
        },
        ];

        let mut shift_pairs = Vec::<(i32, i32)>::new();

        for s in 4..=7 {
            shift_pairs.push((s, 0));
            shift_pairs.push((0, s));
            shift_pairs.push((s, s));
            shift_pairs.push((s, -s));
        }

        let total_hits = AtomicUsize::new(0);

        for (shift_a, shift_b) in shift_pairs {
            let base_blob = deinterleave(&raw6, &raw7, 16, shift_a, shift_b, true);

            if base_blob.is_empty() {
                continue;
            }

            let blobs = vec![
            ("raw", base_blob.clone()),
            ("delta_xor", delta_xor(&base_blob)),
            ("delta_sub", delta_sub(&base_blob)),
            ];

            for (delta_name, blob) in blobs {
                pipelines
                .par_iter()
                .for_each(
                |pipe| {
                    let rot_range =
                    if pipe.use_rot
                    {
                        0u8..=7u8
                    } else {
                        0u8..=0u8
                    };

                    for key
                    in 0u8..=255
                    {
                        for rot in
                        rot_range
                        .clone()
                        {
                            let mut positions =
                            HashSet::<(
                            &str,
                            usize,
                            )>::new();

                            for beacon in
                            &primary_beacons
                            {
                                let hits =
                                streaming_find(
                                &blob,
                                beacon.bytes,
                                key,
                                rot,
                                pipe
                                .func,
                                );

                                for pos in
                                hits
                                {
                                    positions
                                    .insert(
                                    (
                                    beacon
                                    .name,
                                    pos,
                                    ),
                                    );
                                }
                            }

                            if positions
                            .is_empty()
                            {
                                continue;
                            }

                            for (
                            anchor,
                            pos,
                            ) in positions
                            {
                                let start =
                                pos.saturating_sub(
                                WINDOW_RADIUS,
                                );

                                let end =
                                (pos
                                + WINDOW_RADIUS)
                                .min(
                                blob
                                .len(),
                                );

                                let window =
                                materialize_window(
                                &blob,
                                start,
                                end,
                                key,
                                rot,
                                pipe.func,
                                );

                                let run =
                                longest_ascii_run(
                                &window,
                                );

                                if run
                                < MIN_ASCII_RUN
                                {
                                    continue;
                                }

                                let ascii =
                                ascii_ratio(
                                &window,
                                );

                                if ascii
                                < MIN_ASCII_RATIO
                                {
                                    continue;
                                }

                                let ent =
                                entropy(
                                &window,
                                );

                                if ent
                                > MAX_ENTROPY
                                {
                                    continue;
                                }

                                let strings =
                                extract_strings(
                                &window,
                                4,
                                );

                                if strings
                                .is_empty()
                                {
                                    continue;
                                }

                                let strings_lower: Vec<String> =
                                strings
                                .iter()
                                .map(
                                |s| {
                                    s.to_lowercase()
                                },
                                )
                                .collect();

                                let mut score =
                                portugueseness(
                                &strings_lower,
                                );

                                let mut marker_count =
                                0;

                                for beacon in
                                &primary_beacons
                                {
                                    let needle =
                                    String::from_utf8_lossy(
                                    beacon.bytes,
                                    )
                                    .to_lowercase();

                                    let found =
                                    strings_lower
                                    .iter()
                                    .any(
                                    |s| {
                                        s.contains(
                                        &needle,
                                        )
                                    },
                                    );

                                    if found {
                                        marker_count +=
                                        1;

                                        score +=
                                        beacon
                                        .weight;
                                    }
                                }

                                for beacon in
                                &secondary_beacons
                                {
                                    let needle =
                                    String::from_utf8_lossy(
                                    beacon.bytes,
                                    )
                                    .to_lowercase();

                                    let found =
                                    strings_lower
                                    .iter()
                                    .any(
                                    |s| {
                                        s.contains(
                                        &needle,
                                        )
                                    },
                                    );

                                    if found {
                                        marker_count +=
                                        1;

                                        score +=
                                        beacon
                                        .weight;
                                    }
                                }

                                if marker_count
                                < 2
                                {
                                    continue;
                                }

                                let file_base =
                                format!(
                                "out/score{}_{}_sa{}_sb{}_{}_{}_k{:02x}_r{}_{}",
                                score,
                                delta_name,
                                shift_a,
                                shift_b,
                                pipe.name,
                                anchor,
                                key,
                                rot,
                                pos
                                );

                                write(
                                format!(
                                "{}.bin",
                                file_base
                                ),
                                &window,
                                )
                                .ok();

                                let meta =
                                format!(
                                "score={}\ndelta={}\nshift_a={}\nshift_b={}\npipeline={}\nkey=0x{:02x}\nrot={}\nanchor={}\npos={}\nascii_ratio={:.4}\nentropy={:.4}\nlongest_ascii_run={}\nmarker_count={}\nstrings:\n{}\n",
                                score,
                                delta_name,
                                shift_a,
                                shift_b,
                                pipe.name,
                                key,
                                rot,
                                anchor,
                                pos,
                                ascii,
                                ent,
                                run,
                                marker_count,
                                strings.join("\n")
                                );

                                write(
                                format!(
                                "{}.txt",
                                file_base
                                ),
                                meta,
                                )
                                .ok();

                                let hits = total_hits.fetch_add(
                                1,
                                Ordering::Relaxed,
                                ) + 1;

                                if hits % PROGRESS_EVERY == 0 {
                                    println!(
                                    "[{} hits] best_recent=score:{} delta:{} pipe:{} key:{:02x} rot:{} pos:{}",
                                    hits,
                                    score,
                                    delta_name,
                                    pipe.name,
                                    key,
                                    rot,
                                    pos
                                    );
                                }
                            }
                        }
                    }
                },
                );
            }
        }

        println!("total_hits={}", total_hits.load(Ordering::Relaxed));
    }
