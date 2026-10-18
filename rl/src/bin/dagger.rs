//! DAgger data collector: a Halite bot that plays the neural network's moves while a shadow expert
//! (DexBot by default) sees exactly the same frames; the expert's answers are recorded as labels
//! for the positions *our* policy reaches. Std only. Lives in the rl crate (not the starter kit)
//! so the arena never picks it up.
//!
//!     DAGGER_EXPERT="starter-kit/rust/target/release/dex_bot" DAGGER_OUT=rl/data/dagger1 \
//!     NEURO_WEIGHTS=rl/runs/bc4/best.hnn rl/target/release/dagger
//!
//! DAGGER_BETA (default 0): each turn, play the expert's moves instead with this probability.
//! Output, one file per game and seat, `DAGGER_OUT/<pid>-<nanos>.dag`, little endian:
//!     "DAG1", u16 width, u16 height, u8 tag, productions (W*H u8), then per turn:
//!     owners (W*H u8), strengths (W*H u8), expert moves (W*H u8, 0-4; 0 where not ours)
//! Turn k of the file is frame k of the game (frame 0 = the initial map).

#[allow(dead_code)]
#[path = "../../../starter-kit/rust/src/nn.rs"]
mod nn;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn read_line(r: &mut impl BufRead) -> Option<String> {
    let mut s = String::new();
    match r.read_line(&mut s) {
        Ok(0) | Err(_) => None,
        Ok(_) => Some(s.trim_end().to_string()),
    }
}

/// Original Halite I map format: run-length (count, owner) pairs, then W*H strengths.
fn parse_map(line: &str, n: usize) -> (Vec<u8>, Vec<u8>) {
    let mut it = line.split_ascii_whitespace().map(|t| t.parse::<u32>().unwrap());
    let mut owners = Vec::with_capacity(n);
    while owners.len() < n {
        let count = it.next().unwrap() as usize;
        let owner = it.next().unwrap() as u8;
        owners.extend(std::iter::repeat(owner).take(count));
    }
    let strengths = (0..n).map(|_| it.next().unwrap() as u8).collect();
    (owners, strengths)
}

fn send(out: &mut impl Write, s: &str) -> bool {
    out.write_all(s.as_bytes()).is_ok() && out.write_all(b"\n").is_ok() && out.flush().is_ok()
}

fn main() {
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut stdout = std::io::stdout().lock();
    let tag_line = read_line(&mut input).unwrap();
    let size_line = read_line(&mut input).unwrap();
    let prod_line = read_line(&mut input).unwrap();
    let map_line = read_line(&mut input).unwrap();
    let tag: u8 = tag_line.trim().parse().unwrap();
    let dims: Vec<usize> = size_line.split_ascii_whitespace().map(|t| t.parse().unwrap()).collect();
    let (w, h) = (dims[0], dims[1]);
    let n = w * h;
    let prod: Vec<u8> = prod_line.split_ascii_whitespace().map(|t| t.parse().unwrap()).collect();

    let weights = std::env::var("NEURO_WEIGHTS").expect("NEURO_WEIGHTS");
    let net = nn::Net::load(&std::fs::read(&weights).expect("weights")).expect("bad weights");
    let beta: f64 = std::env::var("DAGGER_BETA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let expert_cmd = std::env::var("DAGGER_EXPERT").unwrap_or("starter-kit/rust/target/release/dex_bot".into());
    let out_dir = std::env::var("DAGGER_OUT").unwrap_or("rl/data/dagger".into());
    std::fs::create_dir_all(&out_dir).unwrap();

    // the shadow expert gets the same init lines; its stderr is discarded
    let mut expert = Command::new("sh").arg("-c").arg(&expert_cmd)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().expect("expert");
    let mut ein = expert.stdin.take().unwrap();
    let mut eout = BufReader::new(expert.stdout.take().unwrap());
    for l in [&tag_line, &size_line, &prod_line, &map_line] {
        send(&mut ein, l);
    }
    let _expert_name = read_line(&mut eout);
    send(&mut stdout, "Dagger");

    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let path = format!("{}/{}-{}.dag", out_dir, std::process::id(), nanos);
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    let mut header = b"DAG1".to_vec();
    header.extend((w as u16).to_le_bytes());
    header.extend((h as u16).to_le_bytes());
    header.push(tag);
    header.extend(&prod);
    file.write_all(&header).unwrap();

    let mut rng: u64 = 0x9E3779B97F4A7C15 ^ nanos as u64;
    let mut turn = 0usize;
    loop {
        let Some(line) = read_line(&mut input) else { break };
        let (owners, strengths) = parse_map(&line, n);
        // the expert answers this frame while we compute ours
        let expert_alive = send(&mut ein, &line);
        let logits = net.forward(w, h, &net.features(w, h, &owners, &strengths, &prod, tag, turn));
        let mut expert_dirs = vec![0u8; n];
        if expert_alive {
            if let Some(reply) = read_line(&mut eout) {
                let t: Vec<i64> = reply.split_ascii_whitespace().filter_map(|t| t.parse().ok()).collect();
                for c in t.chunks_exact(3) {
                    let (x, y, d) = (c[0], c[1], c[2]);
                    if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && (0..5).contains(&d) {
                        let i = y as usize * w + x as usize;
                        if owners[i] == tag {
                            expert_dirs[i] = d as u8;
                        }
                    }
                }
            }
        }
        file.write_all(&owners).unwrap();
        file.write_all(&strengths).unwrap();
        file.write_all(&expert_dirs).unwrap();

        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let use_expert = ((rng >> 11) as f64 / (1u64 << 53) as f64) < beta;
        let mut out = String::new();
        for i in 0..n {
            if owners[i] != tag {
                continue;
            }
            let d = if use_expert {
                expert_dirs[i] as usize
            } else {
                let l = &logits[i * 5..i * 5 + 5];
                (0..5).max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a))).unwrap()
            };
            if d != 0 {
                out.push_str(&format!("{} {} {} ", i % w, i / w, d));
            }
        }
        if !send(&mut stdout, out.trim_end()) {
            break;
        }
        turn += 1;
    }
    let _ = file.flush();
    let _ = expert.kill();
}
