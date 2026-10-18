//! In-process Halite games for reinforcement learning: the engine's own map generator and rules
//! (engine/src/game.rs, included at the crate root) plus the game loop of engine/src/main.rs
//! (deaths, end of game, turn limit, ranking), without bots, pipes or replays.

use crate::game::{generate_map, GameState};
use crate::pyrandom::PyRandom;

pub struct Env {
    pub state: GameState,
    pub n: usize,
    /// frames played so far; the network sees frame `turn` (0 = the initial map)
    pub turn: usize,
    pub limit: usize,
    /// per player 1..=n (index 0 unused): the turn they ran out of territory
    pub death: Vec<Option<usize>>,
    pub done: bool,
}

impl Env {
    /// A fresh game on the engine's map for (seed, size, players). Sizes are rounded down to the
    /// player layout's tile size by the generator, as in real games.
    pub fn new(seed: u32, width: usize, height: usize, n: usize) -> Env {
        let mut rng = PyRandom::from_key(&[seed]);
        let map = generate_map(width as i64, height as i64, n, &mut rng);
        let state = GameState::new(&map);
        let limit = (10.0 * ((map.width * map.height) as f64).sqrt()) as usize;
        let mut env = Env { state, n, turn: 0, limit, death: vec![None; n + 1], done: false };
        env.check_end();
        env
    }

    pub fn width(&self) -> usize {
        self.state.width
    }

    pub fn height(&self) -> usize {
        self.state.height
    }

    /// Players with territory at the current frame (who get to move).
    pub fn alive(&self, p: usize) -> bool {
        self.death[p].is_none()
    }

    /// Plays one turn: `dirs[i]` is the move of the piece on site i (all players at once).
    pub fn step(&mut self, dirs: &[u8]) {
        debug_assert!(!self.done);
        self.state.step(dirs);
        self.turn += 1;
        self.check_end();
    }

    /// Same checks as the engine before each turn: record deaths, stop when at most one player
    /// is left (none for a single-player game) or at the turn limit.
    fn check_end(&mut self) {
        let (territory, _, _) = self.state.stats();
        for p in 1..=self.n {
            if self.death[p].is_none() && territory[p] == 0 {
                self.death[p] = Some(self.turn);
            }
        }
        let living = (1..=self.n).filter(|&p| self.death[p].is_none()).count();
        if living <= if self.n == 1 { 0 } else { 1 } || self.turn >= self.limit {
            self.done = true;
        }
    }

    /// Final ranks (1 = best), indexed by player 1..=n: survivors first, then later death, then
    /// territory, then total strength; ties keep player order (the engine's stable sort).
    pub fn ranks(&self) -> Vec<usize> {
        let (territory, strength, _) = self.state.stats();
        let mut order: Vec<usize> = (1..=self.n).collect();
        order.sort_by_key(|&p| {
            let dt = self.death[p];
            (dt.is_some(), -(dt.unwrap_or(0) as i64), -territory[p], -strength[p])
        });
        let mut ranks = vec![0; self.n + 1];
        for (r, &p) in order.iter().enumerate() {
            ranks[p] = r + 1;
        }
        ranks
    }

    /// Owners and strengths as the u8 arrays the features read.
    pub fn frame(&self) -> (Vec<u8>, Vec<u8>) {
        (self.state.owner.clone(), self.state.strength.iter().map(|&s| s as u8).collect())
    }

    pub fn productions(&self) -> Vec<u8> {
        self.state.production.iter().map(|&p| p as u8).collect()
    }

    /// Production owned by each player 0..=n (0 = neutral).
    pub fn production_owned(&self) -> Vec<i64> {
        self.state.stats().2
    }
}

/// `halite-rl simtest --dir rl/data/val [--games 50]`: replays real engine games in Env and checks
/// the map (same seed, same default-size draw as the engine), every frame, the end turn and the
/// final ranking against the replay.
pub fn check_replays() -> candle_core::Result<()> {
    let a = crate::args();
    let dir = a.get("dir").cloned().unwrap_or("rl/data/val".into());
    let max: usize = a.get("games").map(|s| s.parse().unwrap()).unwrap_or(50);
    let mut paths: Vec<_> = std::fs::read_dir(&dir).expect("dir").filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    let (mut ok, mut total, mut frames) = (0, 0, 0);
    for path in paths.iter().take(max) {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let ints = |x: &serde_json::Value| -> Vec<i64> { x.as_array().unwrap().iter().map(|t| t.as_i64().unwrap()).collect() };
        let seed = v["seed"].as_u64().unwrap() as u32;
        let n = v["num_players"].as_u64().unwrap() as usize;
        // the engine draws the default size first, then generates the map from the same generator
        let mut rng = PyRandom::from_key(&[seed]);
        let side = rng.choice(&[20i64, 25, 30, 35, 40, 45, 50]);
        let map = generate_map(side, side, n, &mut rng);
        let state = GameState::new(&map);
        let limit = (10.0 * ((map.width * map.height) as f64).sqrt()) as usize;
        let mut env = Env { state, n, turn: 0, limit, death: vec![None; n + 1], done: false };
        env.check_end();
        let mut good = map.width as u64 == v["width"].as_u64().unwrap()
            && map.height as u64 == v["height"].as_u64().unwrap()
            && map.production.iter().map(|&p| p as i64).collect::<Vec<_>>() == ints(&v["productions"]);
        let fr = v["frames"].as_array().unwrap();
        let moves = v["moves"].as_array().unwrap();
        for (f, frame) in fr.iter().enumerate() {
            if !good {
                break;
            }
            good = env.state.owner.iter().map(|&o| o as i64).collect::<Vec<_>>() == ints(&frame[0])
                && env.state.strength.iter().map(|&s| s as i64).collect::<Vec<_>>() == ints(&frame[1]);
            frames += 1;
            if f < moves.len() {
                if env.done {
                    good = false;
                    break;
                }
                let dirs: Vec<u8> = moves[f].as_str().unwrap().bytes().map(|b| b - b'0').collect();
                env.step(&dirs);
            }
        }
        good &= env.done;
        let ranks = env.ranks();
        for r in v["results"].as_array().unwrap() {
            let p = r["player"].as_u64().unwrap() as usize;
            good &= ranks[p] as u64 == r["rank"].as_u64().unwrap();
        }
        total += 1;
        ok += good as usize;
        if !good {
            println!("MISMATCH {}", path.display());
        }
    }
    println!("{ok}/{total} replays identical ({frames} frames checked, end turn and ranking included)");
    Ok(())
}
