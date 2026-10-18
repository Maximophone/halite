//! A policy evolved from random weights exclusively through self-play.
//! EVO_WEIGHTS overrides the bundled checkpoint; inference matches the trainer.
#[path = "../../../../evolution/src/policy.rs"]
mod policy;
use hlt::*;

fn main() {
    let text = match std::env::var("EVO_WEIGHTS") {
        Ok(path) => std::fs::read_to_string(path).expect("read EVO_WEIGHTS"),
        Err(_) => include_str!("../../weights/evo_bot.evo").to_string(),
    };
    let net = policy::Policy::load(&text).expect("invalid evolution checkpoint");
    let (me, initial, mut conn) = get_init();
    let (width, height) = (initial.width as usize, initial.height as usize);
    let production: Vec<i32> = initial
        .contents
        .iter()
        .flatten()
        .map(|s| s.production as i32)
        .collect();
    send_init("EvoBot");
    let mut turn = 0;
    loop {
        let map = conn.get_frame();
        let owner: Vec<u8> = map.contents.iter().flatten().map(|s| s.owner).collect();
        let strength: Vec<i32> = map
            .contents
            .iter()
            .flatten()
            .map(|s| s.strength as i32)
            .collect();
        let board = policy::Board {
            width,
            height,
            owner: &owner,
            strength: &strength,
            production: &production,
        };
        let mut dirs = vec![0; width * height];
        net.act(&board, me, turn, &mut dirs);
        let moves: Vec<Move> = (0..owner.len())
            .filter(|&i| owner[i] == me && dirs[i] != 0)
            .map(|i| Move {
                loc: Location {
                    x: (i % width) as u16,
                    y: (i / width) as u16,
                },
                direction: dirs[i],
            })
            .collect();
        conn.send_frame(&moves);
        turn += 1;
    }
}
