//! The original Halite I sample bot: every piece moves randomly.
//! Rust port of the Python RandomBot.

use hlt::*;

/// Tiny xorshift generator (no external crates).
struct Rng(u64);

impl Rng {
    fn new() -> Rng {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1);
        Rng(t ^ ((std::process::id() as u64) << 32) | 1)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

fn main() {
    let (my_id, _game_map, mut conn) = get_init();
    send_init("RandomBot");
    let mut rng = Rng::new();

    loop {
        let game_map = conn.get_frame();
        let mut moves = Vec::new();
        for y in 0..game_map.height {
            for x in 0..game_map.width {
                let loc = Location { x, y };
                if game_map.get_site(loc, STILL).owner == my_id {
                    moves.push(Move { loc, direction: DIRECTIONS[rng.below(5) as usize] });
                }
            }
        }
        conn.send_frame(&moves);
    }
}
