//! NeuroAlpha: a convolutional policy network cloned from DexBot's moves (behaviour cloning).
//!
//! Each turn the network (src/nn.rs) reads the whole map and gives 5 move logits per site; every
//! site we own plays its most likely move. No search, no hand-written rules. Trained by the rl/
//! crate; the weights are compiled into the binary (weights/neuro_alpha.hnn) so this bot is a
//! frozen version. NEURO_WEIGHTS=path loads other weights instead, NEURO_TEMP=t > 0 samples
//! moves from softmax(logits / t) instead of taking the most likely one.

use hlt::nn::{features, Net, NUM_MOVES};
use hlt::*;

const WEIGHTS: &[u8] = include_bytes!("../../weights/neuro_alpha.hnn");

fn main() {
    let (me, map, mut conn) = get_init();
    let (w, h) = (map.width as usize, map.height as usize);
    let bytes = match std::env::var("NEURO_WEIGHTS") {
        Ok(p) => std::fs::read(&p).unwrap_or_else(|e| panic!("cannot read {p}: {e}")),
        Err(_) => WEIGHTS.to_vec(),
    };
    let net = Net::load(&bytes).unwrap_or_else(|e| panic!("bad weights: {e}"));
    let temp: f32 = std::env::var("NEURO_TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let mut rng: u64 = 0x9E3779B97F4A7C15 ^ ((me as u64) << 32) ^ (w * 1000 + h) as u64;
    let productions: Vec<u8> = (0..w * h).map(|i| map.contents[i / w][i % w].production).collect();
    send_init("NeuroAlpha");

    let mut turn = 0;
    loop {
        let map = conn.get_frame();
        let mut owners = vec![0u8; w * h];
        let mut strengths = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                let s = map.contents[y][x];
                owners[y * w + x] = s.owner;
                strengths[y * w + x] = s.strength;
            }
        }
        let logits = net.forward(w, h, &features(w, h, &owners, &strengths, &productions, me, turn));
        let mut moves = Vec::new();
        for i in 0..w * h {
            if owners[i] != me {
                continue;
            }
            let l = &logits[i * NUM_MOVES..(i + 1) * NUM_MOVES];
            let dir = if temp > 0.0 {
                let m = l.iter().cloned().fold(f32::MIN, f32::max);
                let p: Vec<f32> = l.iter().map(|v| ((v - m) / temp).exp()).collect();
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let mut r = (rng >> 11) as f32 / (1u64 << 53) as f32 * p.iter().sum::<f32>();
                let mut k = 0;
                while k + 1 < NUM_MOVES && r >= p[k] {
                    r -= p[k];
                    k += 1;
                }
                k
            } else {
                (0..NUM_MOVES).max_by(|&a, &b| l[a].total_cmp(&l[b]).then(b.cmp(&a))).unwrap()
            };
            if dir != 0 {
                moves.push(Move { loc: Location { x: (i % w) as u16, y: (i / w) as u16 }, direction: dir as u8 });
            }
        }
        conn.send_frame(&moves);
        turn += 1;
    }
}
