//! Halite I game objects and networking for Rust bots.
//!
//! Speaks the same text protocol as the 2016 starter kits. Never print to
//! stdout from your bot: stdout is the game channel. Use `eprintln!` (the
//! environment saves stderr to logs/bot-N.log).

use std::io::{self, BufRead, Write};

/// Neural policy (features, std-only forward pass), shared with the rl/ trainer.
pub mod nn;

pub const STILL: u8 = 0;
pub const NORTH: u8 = 1;
pub const EAST: u8 = 2;
pub const SOUTH: u8 = 3;
pub const WEST: u8 = 4;

pub const DIRECTIONS: [u8; 5] = [STILL, NORTH, EAST, SOUTH, WEST];
pub const CARDINALS: [u8; 4] = [NORTH, EAST, SOUTH, WEST];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Location {
    pub x: u16,
    pub y: u16,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Site {
    pub owner: u8,
    pub strength: u8,
    pub production: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Move {
    pub loc: Location,
    pub direction: u8,
}

#[derive(Clone, Debug)]
pub struct GameMap {
    pub width: u16,
    pub height: u16,
    /// contents[y][x]
    pub contents: Vec<Vec<Site>>,
}

impl GameMap {
    pub fn in_bounds(&self, l: Location) -> bool {
        l.x < self.width && l.y < self.height
    }

    /// Manhattan distance on the wrap-around map.
    pub fn get_distance(&self, l1: Location, l2: Location) -> u16 {
        let mut dx = (l1.x as i32 - l2.x as i32).abs();
        let mut dy = (l1.y as i32 - l2.y as i32).abs();
        if dx > self.width as i32 / 2 {
            dx = self.width as i32 - dx;
        }
        if dy > self.height as i32 / 2 {
            dy = self.height as i32 - dy;
        }
        (dx + dy) as u16
    }

    pub fn get_angle(&self, l1: Location, l2: Location) -> f64 {
        let (w, h) = (self.width as i32, self.height as i32);
        let mut dx = l2.x as i32 - l1.x as i32;
        let mut dy = l2.y as i32 - l1.y as i32;
        if dx > w - dx {
            dx -= w;
        } else if -dx > w + dx {
            dx += w;
        }
        if dy > h - dy {
            dy -= h;
        } else if -dy > h + dy {
            dy += h;
        }
        (dy as f64).atan2(dx as f64)
    }

    pub fn get_location(&self, l: Location, direction: u8) -> Location {
        let (w, h) = (self.width, self.height);
        match direction {
            NORTH => Location { x: l.x, y: (l.y + h - 1) % h },
            EAST => Location { x: (l.x + 1) % w, y: l.y },
            SOUTH => Location { x: l.x, y: (l.y + 1) % h },
            WEST => Location { x: (l.x + w - 1) % w, y: l.y },
            _ => l,
        }
    }

    pub fn get_site(&self, l: Location, direction: u8) -> Site {
        let l = self.get_location(l, direction);
        self.contents[l.y as usize][l.x as usize]
    }
}

/// Connection to the game environment: reads stdin, writes moves to stdout.
pub struct Connection {
    stdin: io::StdinLock<'static>,
    width: u16,
    height: u16,
    productions: Vec<u8>,
    line: String,
}

impl Connection {
    fn get_string(&mut self) -> &str {
        self.line.clear();
        match self.stdin.read_line(&mut self.line) {
            Ok(0) | Err(_) => std::process::exit(0), // environment closed the pipe: game over
            Ok(_) => {}
        }
        self.line.trim_end()
    }

    fn deserialize_map(&mut self) -> GameMap {
        let (w, h) = (self.width as usize, self.height as usize);
        let productions = &self.productions;
        let line = {
            self.line.clear();
            if matches!(self.stdin.read_line(&mut self.line), Ok(0) | Err(_)) {
                std::process::exit(0);
            }
            &self.line
        };
        let mut nums = line.split_ascii_whitespace().map(|t| t.parse::<u32>().unwrap());
        let mut contents = vec![vec![Site::default(); w]; h];
        let (mut x, mut y) = (0, 0);
        while y < h {
            let counter = nums.next().unwrap();
            let owner = nums.next().unwrap() as u8;
            for _ in 0..counter {
                contents[y][x].owner = owner;
                x += 1;
                if x == w {
                    x = 0;
                    y += 1;
                }
            }
        }
        for (yy, row) in contents.iter_mut().enumerate() {
            for (xx, site) in row.iter_mut().enumerate() {
                site.strength = nums.next().unwrap() as u8;
                site.production = productions[yy * w + xx];
            }
        }
        GameMap { width: self.width, height: self.height, contents }
    }

    /// Reads the next turn's map.
    pub fn get_frame(&mut self) -> GameMap {
        self.deserialize_map()
    }

    /// Sends one turn's moves.
    pub fn send_frame(&mut self, moves: &[Move]) {
        let mut out = String::with_capacity(moves.len() * 10);
        for (k, m) in moves.iter().enumerate() {
            if k > 0 {
                out.push(' ');
            }
            out.push_str(&format!("{} {} {}", m.loc.x, m.loc.y, m.direction));
        }
        send_string(&out);
    }
}

fn send_string(s: &str) {
    let mut stdout = io::stdout().lock();
    let ok = stdout.write_all(s.replace('\n', "").as_bytes()).is_ok()
        && stdout.write_all(b"\n").is_ok()
        && stdout.flush().is_ok();
    if !ok {
        std::process::exit(0);
    }
}

/// Reads the init lines. Returns (player tag, initial map, connection).
pub fn get_init() -> (u8, GameMap, Connection) {
    let mut c = Connection {
        stdin: io::stdin().lock(),
        width: 0,
        height: 0,
        productions: Vec::new(),
        line: String::new(),
    };
    let tag: u8 = c.get_string().trim().parse().unwrap();
    let size: Vec<u16> = c.get_string().split_ascii_whitespace().map(|t| t.parse().unwrap()).collect();
    c.width = size[0];
    c.height = size[1];
    c.productions = c.get_string().split_ascii_whitespace().map(|t| t.parse().unwrap()).collect();
    let m = c.deserialize_map();
    (tag, m, c)
}

/// Replies to init with the bot's name.
pub fn send_init(name: &str) {
    send_string(name);
}
