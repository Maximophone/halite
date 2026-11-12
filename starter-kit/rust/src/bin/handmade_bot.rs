//! HandmadeBot: a small convolutional network whose weights were written by hand. No training,
//! no gradients, no imitation: every weight below has a name and a reason.
//!
//! The network reads five raw planes per site (mine, enemy, neutral, strength / 255,
//! production / 15) and has five layers, all 3x3 convolutions that wrap around the map, of the
//! form `out = relu(conv(x) + bias) + skip(x)` (`skip` is a linear 1x1 shortcut):
//!
//! 1. Gates (1x1). ReLU as an AND gate: `relu(s + mine - 1)` is "my strength" (s where the site
//!    is mine, else 0), likewise "neutral strength". Also the target value of a site that is not
//!    mine, `V = relu(wp * production - ws * strength + c) + floor` (plus `E` for an enemy site), and
//!    `wait = relu(k * production - strength)`: positive while a piece is not grown.
//!    Then a blur layer, applied 8 times: `V += 0.2 * (sum of the 4 neighbours' V)`, so a
//!    target next to other good targets is worth more (it opens the way to them).
//! 2. Heat (one layer applied `T` times, the same weights each time). `F = V` outside our land;
//!    inside, `F = lambda * (sum of the 4 neighbours' F)`. Value flows from the frontier into
//!    our territory like heat, so walking uphill on F leads to the best nearby frontier.
//! 3. Per-direction features. For each of the 4 neighbours: `up` (how much higher its F is than
//!    ours), `blocked` (it is neutral and at least as strong as we are), `waiting` (it is ours
//!    and we are not grown), `overflow` (it is ours and together we would pass 255).
//! 4. Head (1x1, linear): `logit[d] = up[d] - BIG * (blocked[d] + waiting[d] + overflow[d])`,
//!    and the logit of STILL is 0. Each of our sites plays its argmax.
//!
//! So a piece moves uphill when nothing forbids it: it captures a neighbour it can beat, or
//! carries its strength toward the frontier once it has grown. The handful of scalars
//! (`HAND_*` variables, defaults in `Knobs::load`) were set by playing games, not by descent.
//! `HAND_DEBUG=1` prints the number of non-zero weights to stderr (100).

use hlt::*;

// Input planes.
const I_MINE: usize = 0;
const I_ENEMY: usize = 1;
const I_NEUTRAL: usize = 2;
const I_S: usize = 3;
const I_P: usize = 4;
const N_IN: usize = 5;

// Channels after the gates, kept through the heat layer.
const MINE: usize = 0;
const MS: usize = 1; // my strength
const NS: usize = 2; // neutral strength
const V: usize = 3; // value of a site that is not mine
const WAIT: usize = 4; // > 0 while a piece of mine is not grown
const F: usize = 5; // the heat field
const N_MID: usize = 6;

// Channels of the per-direction layer: 4 directions (N, E, S, W) of each.
const UP: usize = 0;
const BLOCKED: usize = 4;
const WAITING: usize = 8;
const OVERFLOW: usize = 12;
const N_DIR: usize = 16;

/// (dy, dx) of NORTH, EAST, SOUTH, WEST.
const STEP: [(i32, i32); 4] = [(-1, 0), (0, 1), (1, 0), (0, -1)];
const BIG: f32 = 1e6;
/// One unit of strength on the network's scale.
const UNIT: f32 = 1.0 / 255.0;

struct Knobs {
    /// V = relu(wp * production - ws * strength + c) for a neutral site; + enemy for an enemy's.
    wp: f32,
    ws: f32,
    c: f32,
    enemy: f32,
    /// Value every neutral site has at least.
    floor: f32,
    /// Blur: share of each neighbour's value added to a target's, and how many times.
    blur: f32,
    blurs: usize,
    /// Heat kept per step into our land (0.25 = no loss with distance).
    lambda: f32,
    /// Heat steps: how deep into our territory the frontier is felt.
    steps: usize,
    /// A piece is grown at `k` times its production.
    k: f32,
    /// Strength above 255 tolerated when two of our pieces merge.
    slack: f32,
}

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

impl Knobs {
    fn load() -> Knobs {
        Knobs {
            wp: env("HAND_WP", 1.0),
            ws: env("HAND_WS", 0.6),
            c: env("HAND_C", 0.1),
            enemy: env("HAND_ENEMY", 20.0),
            floor: env("HAND_FLOOR", 0.02),
            blur: env("HAND_BLUR", 0.2),
            blurs: env("HAND_BLURS", 8.0) as usize,
            lambda: env("HAND_LAMBDA", 0.22),
            steps: env("HAND_STEPS", 24.0) as usize,
            k: env("HAND_K", 5.0),
            slack: env("HAND_SLACK", 0.0),
        }
    }
}

/// One weight of a 3x3 convolution: output channel, input channel, offset, value.
struct Tap {
    out: usize,
    inp: usize,
    dy: i32,
    dx: i32,
    w: f32,
}

/// `out = relu(conv(x) + bias) + skip(x)`, or without the ReLU for the head.
struct Layer {
    cin: usize,
    cout: usize,
    taps: Vec<Tap>,
    bias: Vec<f32>,
    /// (output channel, input channel, weight) of the linear shortcut.
    skip: Vec<(usize, usize, f32)>,
    relu: bool,
}

impl Layer {
    fn new(cin: usize, cout: usize, relu: bool) -> Layer {
        Layer { cin, cout, taps: Vec::new(), bias: vec![0.0; cout], skip: Vec::new(), relu }
    }
    /// A weight on the site itself.
    fn here(&mut self, out: usize, inp: usize, w: f32) {
        self.at(out, inp, (0, 0), w);
    }
    /// A weight on the neighbour at offset `d`.
    fn at(&mut self, out: usize, inp: usize, d: (i32, i32), w: f32) {
        self.taps.push(Tap { out, inp, dy: d.0, dx: d.1, w });
    }
    fn weights(&self) -> usize {
        self.taps.len() + self.bias.iter().filter(|&&b| b != 0.0).count() + self.skip.len()
    }

    fn forward(&self, w: usize, h: usize, x: &[f32]) -> Vec<f32> {
        let n = w * h;
        let mut y = vec![0.0f32; n * self.cout];
        for i in 0..n {
            y[i * self.cout..(i + 1) * self.cout].copy_from_slice(&self.bias);
        }
        for t in &self.taps {
            for r in 0..h {
                let rr = (r as i32 + t.dy).rem_euclid(h as i32) as usize;
                for c in 0..w {
                    let cc = (c as i32 + t.dx).rem_euclid(w as i32) as usize;
                    y[(r * w + c) * self.cout + t.out] += t.w * x[(rr * w + cc) * self.cin + t.inp];
                }
            }
        }
        if self.relu {
            for v in y.iter_mut() {
                *v = v.max(0.0);
            }
        }
        for &(o, i, wt) in &self.skip {
            for s in 0..n {
                y[s * self.cout + o] += wt * x[s * self.cin + i];
            }
        }
        y
    }
}

struct Net {
    gates: Layer,
    blur: Layer,
    heat: Layer,
    dirs: Layer,
    head: Layer,
    blurs: usize,
    steps: usize,
}

/// Every weight of the network, written by hand.
fn build(k: &Knobs) -> Net {
    // 1. Gates. For a 0/1 plane g and a value v in [0, 1], relu(v + g - 1) is "v where g".
    let mut gates = Layer::new(N_IN, N_MID, true);
    gates.here(MINE, I_MINE, 1.0);
    gates.here(MS, I_S, 1.0);
    gates.here(MS, I_MINE, 1.0);
    gates.bias[MS] = -1.0;
    gates.here(NS, I_S, 1.0);
    gates.here(NS, I_NEUTRAL, 1.0);
    gates.bias[NS] = -1.0;
    // Value of a target: rich and cheap is best; an enemy site is worth attacking; ours is 0.
    gates.here(V, I_P, k.wp);
    gates.here(V, I_S, -k.ws);
    gates.here(V, I_ENEMY, k.enemy);
    gates.here(V, I_MINE, -100.0);
    gates.bias[V] = k.c;
    // Never exactly worthless: barren or expensive land still attracts a little.
    gates.skip.push((V, I_NEUTRAL, k.floor));
    // wait > 0 exactly when strength < k * production (strength is s * 255, production p * 15).
    gates.here(WAIT, I_P, k.k * 15.0 * UNIT);
    gates.here(WAIT, I_S, -1.0);
    // F starts at 0; the first heat step sets it to V outside our land.

    // 1b. Blur: a target next to other good targets is worth more (it opens the way to them).
    let mut blur = Layer::new(N_MID, N_MID, true);
    for ch in [MINE, MS, NS, V, WAIT] {
        blur.skip.push((ch, ch, 1.0));
    }
    for d in STEP {
        blur.at(V, V, d, k.blur);
    }
    blur.here(V, MINE, -100.0);

    // 2. Heat, applied `steps` times. Channels pass through unchanged, except F:
    //    F = relu(lambda * (F north + east + south + west) - 100 * (1 - mine)) + V.
    let mut heat = Layer::new(N_MID, N_MID, true);
    for ch in [MINE, MS, NS, V, WAIT] {
        heat.skip.push((ch, ch, 1.0));
    }
    for d in STEP {
        heat.at(F, F, d, k.lambda);
    }
    heat.here(F, MINE, 100.0);
    heat.bias[F] = -100.0;
    heat.skip.push((F, V, 1.0));

    // 3. Per-direction features.
    let mut dirs = Layer::new(N_MID, N_DIR, true);
    for (d, &step) in STEP.iter().enumerate() {
        // up: the neighbour's heat above ours.
        dirs.at(UP + d, F, step, 1.0);
        dirs.here(UP + d, F, -1.0);
        // blocked: a neutral neighbour at least as strong as we are.
        dirs.at(BLOCKED + d, NS, step, 1.0);
        dirs.here(BLOCKED + d, MS, -1.0);
        dirs.bias[BLOCKED + d] = UNIT;
        // waiting: the neighbour is ours (an AND gate again) and we are not grown.
        dirs.here(WAITING + d, WAIT, 1.0);
        dirs.at(WAITING + d, MINE, step, 1.0);
        dirs.bias[WAITING + d] = -1.0;
        // overflow: our two strengths together pass 255 (plus the slack).
        dirs.at(OVERFLOW + d, MS, step, 1.0);
        dirs.here(OVERFLOW + d, MS, 1.0);
        dirs.bias[OVERFLOW + d] = -1.0 - k.slack * UNIT;
    }

    // 4. Head: move uphill unless something forbids it. STILL (logit 0) wins ties.
    let mut head = Layer::new(N_DIR, 5, false);
    for d in 0..4 {
        let mv = d + 1;
        head.here(mv, UP + d, 1.0);
        head.here(mv, BLOCKED + d, -BIG);
        head.here(mv, WAITING + d, -BIG);
        head.here(mv, OVERFLOW + d, -BIG);
    }
    Net { gates, blur, heat, dirs, head, blurs: k.blurs, steps: k.steps }
}

impl Net {
    /// Logits (5 per site) from the raw planes.
    fn forward(&self, w: usize, h: usize, input: &[f32]) -> Vec<f32> {
        let mut x = self.gates.forward(w, h, input);
        for _ in 0..self.blurs {
            x = self.blur.forward(w, h, &x);
        }
        for _ in 0..self.steps {
            x = self.heat.forward(w, h, &x);
        }
        let x = self.dirs.forward(w, h, &x);
        self.head.forward(w, h, &x)
    }
}

fn main() {
    let (me, map, mut conn) = get_init();
    let (w, h) = (map.width as usize, map.height as usize);
    let knobs = Knobs::load();
    let net = build(&knobs);
    if std::env::var("HAND_DEBUG").is_ok() {
        let total: usize = [&net.gates, &net.blur, &net.heat, &net.dirs, &net.head].iter().map(|l| l.weights()).sum();
        eprintln!("non-zero weights: {total}, heat steps: {}", net.steps);
    }
    send_init("HandmadeBot");
    let mut input = vec![0.0f32; w * h * N_IN];
    loop {
        let map = conn.get_frame();
        for y in 0..h {
            for x in 0..w {
                let s = map.contents[y][x];
                let cell = &mut input[(y * w + x) * N_IN..(y * w + x + 1) * N_IN];
                cell[I_MINE] = (s.owner == me) as u8 as f32;
                cell[I_ENEMY] = (s.owner != me && s.owner != 0) as u8 as f32;
                cell[I_NEUTRAL] = (s.owner == 0) as u8 as f32;
                cell[I_S] = s.strength as f32 * UNIT;
                cell[I_P] = s.production as f32 / 15.0;
            }
        }
        let logits = net.forward(w, h, &input);
        let mut moves = Vec::new();
        for y in 0..h {
            for x in 0..w {
                if map.contents[y][x].owner != me {
                    continue;
                }
                let l = &logits[(y * w + x) * 5..(y * w + x + 1) * 5];
                let mut best = 0;
                for d in 1..5 {
                    if l[d] > l[best] {
                        best = d;
                    }
                }
                if best != 0 {
                    moves.push(Move { loc: Location { x: x as u16, y: y as u16 }, direction: best as u8 });
                }
            }
        }
        conn.send_frame(&moves);
    }
}
