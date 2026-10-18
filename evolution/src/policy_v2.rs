//! V2 adds evolvable action-conditioned features to the V1 neural policy.
//! The inherited weights also mutate. No action is filtered or overwritten.
pub use crate::policy::Board;
use crate::policy::{Observation, Policy as Base, PARAMS as BASE_PARAMS};

pub const EXTRA: usize = 16;
pub const PARAMS: usize = BASE_PARAMS + EXTRA;

#[derive(Clone)]
pub struct Policy {
    base: Base,
    residual: Option<[f32; EXTRA]>,
}

impl Policy {
    pub fn load(text: &str) -> Result<Self, String> {
        if text.split_whitespace().next() == Some("HALITE_EVO_V1") {
            return Ok(Self { base: Base::load(text)?, residual: None });
        }
        let mut tokens = text.split_whitespace();
        if tokens.next() != Some("HALITE_EVO_V2") { return Err("unknown evolution checkpoint version".into()); }
        let weights: Vec<f32> = tokens.map(str::parse).collect::<Result<_, _>>().map_err(|_| "invalid weight")?;
        if weights.len() != PARAMS || weights.iter().any(|v| !v.is_finite()) {
            return Err(format!("expected {PARAMS} finite weights"));
        }
        Ok(Self { base: Base { weights: weights[..BASE_PARAMS].to_vec() },
                  residual: Some(weights[BASE_PARAMS..].try_into().unwrap()) })
    }

    pub fn extra(x: &[f32; 24]) -> [f32; EXTRA] {
        let (s, p, t, tp, own, neutral, enemy, still) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        [still * s, still * p, still * s * s, still * x[10], still * x[9],
         own * (s + t), neutral * (s - t), neutral * (s - t).min(0.0),
         enemy * (s - t), s * x[10], x[15] * s, x[15] * x[14], x[17] * s,
         neutral * (tp / (0.05 + t)).min(4.0) / 4.0, own * t,
         still * (s / (0.02 + p)).min(4.0) / 4.0]
    }

    pub fn act(&self, board: &Board, me: u8, turn: usize, dirs: &mut [u8]) {
        let Some(residual) = &self.residual else { self.base.act(board, me, turn, dirs); return; };
        let obs = Observation::new(board, me);
        for i in 0..board.owner.len() {
            if board.owner[i] != me { continue; }
            let mut best = f32::NEG_INFINITY;
            let mut action = 0;
            for d in 0..5 {
                let x = obs.features(i, d, turn);
                let mut score = self.base.score(&x);
                for (w, f) in residual.iter().zip(Self::extra(&x)) { score += w * f; }
                if score > best { best = score; action = d as u8; }
            }
            dirs[i] = action;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_extension_preserves_v1_moves_and_ids() {
        let text = include_str!("../../starter-kit/rust/weights/evo_bot.evo");
        let original = Policy::load(text).unwrap();
        let extended = Policy::load(&format!("{} {}", text.replacen("V1", "V2", 1), vec!["0"; EXTRA].join(" "))).unwrap();
        let mut env = crate::sim::Env::new(103, 20, 20, 2);
        for _ in 0..80 {
            if env.done { break; }
            let board = Board { width: env.width(), height: env.height(), owner: &env.state.owner,
                strength: &env.state.strength, production: &env.state.production };
            let mut a = vec![0; board.owner.len()];
            let mut b = a.clone();
            for player in 1..=2 {
                original.act(&board, player, env.turn, &mut a);
                extended.act(&board, player, env.turn, &mut b);
            }
            assert_eq!(a, b);
            env.step(&a);
        }
    }
    #[test]
    fn invalid_v2_is_rejected() {
        assert!(Policy::load("HALITE_EVO_V2 0").is_err());
        assert!(Policy::load(&format!("HALITE_EVO_V2 {}", vec!["inf"; PARAMS].join(" "))).is_err());
        assert!(Policy::load(&format!("HALITE_EVO_V2 {}", vec!["0"; PARAMS].join(" "))).is_ok());
    }
}
