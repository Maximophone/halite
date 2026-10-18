//! Shared by the in-process evaluator and the protocol bot. No action rules,
//! search, teacher weights, gradients, or training-only behavior.
use std::collections::VecDeque;

pub const INPUTS: usize = 24;
pub const HIDDEN: usize = 8;
pub const PARAMS: usize = INPUTS * HIDDEN + HIDDEN + HIDDEN + INPUTS;

#[derive(Clone)]
pub struct Policy {
    pub weights: Vec<f32>,
}

impl Policy {
    pub fn load(text: &str) -> Result<Self, String> {
        let mut tokens = text.split_whitespace();
        if tokens.next() != Some("HALITE_EVO_V1") {
            return Err("expected HALITE_EVO_V1 checkpoint".into());
        }
        let weights: Result<Vec<f32>, _> = tokens.map(str::parse).collect();
        let weights = weights.map_err(|_| "invalid weight")?;
        if weights.len() != PARAMS || weights.iter().any(|v| !v.is_finite()) {
            return Err(format!("expected {PARAMS} finite weights"));
        }
        Ok(Self { weights })
    }

    pub fn score(&self, x: &[f32; INPUTS]) -> f32 {
        let w = &self.weights;
        let mut score = 0.0;
        for h in 0..HIDDEN {
            let mut v = w[INPUTS * HIDDEN + h];
            for j in 0..INPUTS {
                v += x[j] * w[h * INPUTS + j];
            }
            // Softsign is bounded like tanh, with inexpensive std-only inference.
            score += (v / (1.0 + v.abs())) * w[INPUTS * HIDDEN + HIDDEN + h];
        }
        for j in 0..INPUTS {
            score += x[j] * w[INPUTS * HIDDEN + 2 * HIDDEN + j];
        }
        score
    }

    pub fn act(&self, board: &Board, me: u8, turn: usize, dirs: &mut [u8]) {
        let obs = Observation::new(board, me);
        for i in 0..board.owner.len() {
            if board.owner[i] != me {
                continue;
            }
            let mut best = f32::NEG_INFINITY;
            let mut action = 0;
            for d in 0..5 {
                let score = self.score(&obs.features(i, d, turn));
                if score > best {
                    best = score;
                    action = d as u8;
                }
            }
            dirs[i] = action;
        }
    }
}

pub struct Board<'a> {
    pub width: usize,
    pub height: usize,
    pub owner: &'a [u8],
    pub strength: &'a [i32],
    pub production: &'a [i32],
}

impl Board<'_> {
    pub fn neighbor(&self, i: usize, d: usize) -> usize {
        let (x, y, w, h) = (i % self.width, i / self.width, self.width, self.height);
        match d {
            1 => ((y + h - 1) % h) * w + x,
            2 => y * w + (x + 1) % w,
            3 => ((y + 1) % h) * w + x,
            4 => y * w + (x + w - 1) % w,
            _ => i,
        }
    }
}

pub struct Observation<'a, 'b> {
    board: &'a Board<'b>,
    me: u8,
    border: Vec<f32>,
    enemy: Vec<f32>,
    neighborhood: Vec<[f32; 5]>,
    territory: f32,
    economy: f32,
}

fn distance(board: &Board, source: impl Fn(u8) -> bool) -> Vec<f32> {
    let mut dist = vec![(board.width + board.height) as f32; board.owner.len()];
    let mut queue = VecDeque::new();
    for (i, &o) in board.owner.iter().enumerate() {
        if source(o) {
            dist[i] = 0.0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for d in 1..5 {
            let j = board.neighbor(i, d);
            if dist[j] > dist[i] + 1.0 {
                dist[j] = dist[i] + 1.0;
                queue.push_back(j);
            }
        }
    }
    dist
}

impl<'a, 'b> Observation<'a, 'b> {
    pub fn new(board: &'a Board<'b>, me: u8) -> Self {
        let mut neighborhood = vec![[0.0; 5]; board.owner.len()];
        let (mut count, mut own_prod, mut enemy_prod) = (0.0, 0.0, 0.0);
        for i in 0..board.owner.len() {
            if board.owner[i] == me {
                count += 1.0;
                own_prod += board.production[i] as f32;
            } else if board.owner[i] != 0 {
                enemy_prod += board.production[i] as f32;
            }
            for d in 1..5 {
                let j = board.neighbor(i, d);
                let o = board.owner[j];
                if o == me {
                    neighborhood[i][0] += board.strength[j] as f32 / 1020.0;
                    neighborhood[i][2] += 0.25;
                } else if o != 0 {
                    neighborhood[i][1] += board.strength[j] as f32 / 1020.0;
                    neighborhood[i][3] += 0.25;
                } else {
                    neighborhood[i][4] += 0.25;
                }
            }
        }
        Self {
            board,
            me,
            border: distance(board, |o| o != me),
            enemy: distance(board, |o| o != 0 && o != me),
            neighborhood,
            territory: count / board.owner.len() as f32,
            economy: (own_prod - enemy_prod) / (1.0 + own_prod + enemy_prod),
        }
    }

    pub fn features(&self, i: usize, d: usize, turn: usize) -> [f32; INPUTS] {
        let b = self.board;
        let j = b.neighbor(i, d);
        let k = b.neighbor(j, d);
        let s = b.strength[i] as f32 / 255.0;
        let t = b.strength[j] as f32 / 255.0;
        let own = (b.owner[j] == self.me) as u8 as f32;
        let neutral = (b.owner[j] == 0) as u8 as f32;
        let scale = (b.width + b.height) as f32;
        let n = self.neighborhood[j];
        [
            s,
            b.production[i] as f32 / 15.0,
            t,
            b.production[j] as f32 / 15.0,
            own,
            neutral,
            1.0 - own - neutral,
            (d == 0) as u8 as f32,
            s - t,
            n[0],
            n[1],
            n[2],
            n[3],
            n[4],
            self.border[j] / scale,
            self.border[i] - self.border[j],
            self.enemy[j] / scale,
            self.enemy[i] - self.enemy[j],
            turn as f32 / (10.0 * (b.owner.len() as f32).sqrt()),
            self.territory,
            self.economy,
            if d == 0 {
                (s + b.production[i] as f32 / 255.0 - 1.0).max(0.0)
            } else {
                (s + t - 1.0).max(0.0) * own
            },
            b.strength[k] as f32 / 255.0,
            b.production[k] as f32 / 15.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoints_reject_malformed_and_nonfinite() {
        assert!(Policy::load("bad").is_err());
        assert!(Policy::load("HALITE_EVO_V1 1").is_err());
        assert!(Policy::load(&format!("HALITE_EVO_V1 {}", vec!["NaN"; PARAMS].join(" "))).is_err());
        assert!(Policy::load(&format!("HALITE_EVO_V1 {}", vec!["0"; PARAMS].join(" "))).is_ok());
    }
    #[test]
    fn wrapping_and_player_relabeling() {
        let owners = [1, 1, 0, 0, 2, 0];
        let relabeled = [2, 2, 0, 0, 1, 0];
        let strength = [255, 2, 30, 0, 20, 10];
        let production = [1, 2, 3, 4, 5, 6];
        let b = Board {
            width: 3,
            height: 2,
            owner: &owners,
            strength: &strength,
            production: &production,
        };
        let c = Board {
            width: 3,
            height: 2,
            owner: &relabeled,
            strength: &strength,
            production: &production,
        };
        assert_eq!(b.neighbor(0, 4), 2);
        assert_eq!(b.neighbor(0, 1), 3);
        let o = Observation::new(&b, 1);
        let p = Observation::new(&c, 2);
        for i in 0..6 {
            for d in 0..5 {
                assert_eq!(o.features(i, d, 4), p.features(i, d, 4));
                assert!(o.features(i, d, 4).iter().all(|x| x.is_finite()));
            }
        }
    }
}
