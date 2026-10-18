//! Minimal game loop over the actual engine rules and map generator.
use crate::{
    game::{generate_map, GameState},
    pyrandom::PyRandom,
};

pub struct Env {
    pub state: GameState,
    pub turn: usize,
    pub death: Vec<Option<usize>>,
    pub done: bool,
    limit: usize,
}

impl Env {
    pub fn new(seed: u32, width: usize, height: usize, players: usize) -> Self {
        let mut rng = PyRandom::from_key(&[seed]);
        let map = generate_map(width as i64, height as i64, players, &mut rng);
        Self {
            state: GameState::new(&map),
            turn: 0,
            death: vec![None; players + 1],
            done: false,
            limit: (10.0 * ((map.width * map.height) as f64).sqrt()) as usize,
        }
    }
    pub fn width(&self) -> usize {
        self.state.width
    }
    pub fn height(&self) -> usize {
        self.state.height
    }
    pub fn step(&mut self, dirs: &[u8]) {
        self.state.step(dirs);
        self.turn += 1;
        let (territory, _, _) = self.state.stats();
        for p in 1..=self.state.num_players {
            if self.death[p].is_none() && territory[p] == 0 {
                self.death[p] = Some(self.turn);
            }
        }
        self.done =
            self.death[1..].iter().filter(|d| d.is_none()).count() <= 1 || self.turn >= self.limit;
    }
}
