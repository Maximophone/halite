//! Halite I game environment in Rust: a port of halite.py that plays the same
//! games (same maps from a seed, same rules, same replays, same console
//! output). halite.py stays the reference; check_engine.py proves they agree.
//!
//!     engine/target/release/halite -d "30 30" -s 7 starter-kit/rust/target/release/my_bot starter-kit/rust/target/release/improved_bot

mod bot;
mod game;
mod json;
mod pyrandom;

use bot::{turn_deadline, Bot};
use game::{generate_map, GameState, MAX_PLAYERS};
use json::Value;
use pyrandom::PyRandom;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

const INIT_TIMEOUT: f64 = 15.0;
const TURN_TIMEOUT: f64 = 1.0;
const VERSION: i64 = 1;

// ---------------------------------------------------------------------------
// Seeds: Python ints of any size
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Seed {
    /// Normalised decimal form, e.g. "-12" (what Python's %d prints).
    text: String,
    /// abs(seed) as 32-bit words, least significant first.
    key: Vec<u32>,
}

impl Seed {
    fn from_u64(v: u64) -> Seed {
        Seed::parse(&v.to_string()).unwrap()
    }

    /// Python's int(): surrounding whitespace, a sign, digits with single
    /// underscores between them.
    fn parse(s: &str) -> Option<Seed> {
        let t = s.trim_matches(is_py_space);
        let (neg, body) = match t.as_bytes().first() {
            Some(b'-') => (true, &t[1..]),
            Some(b'+') => (false, &t[1..]),
            _ => (false, t),
        };
        if body.is_empty() || body.starts_with('_') || body.ends_with('_') || body.contains("__") {
            return None;
        }
        let mut digits = String::new();
        for c in body.chars() {
            match c {
                '0'..='9' => digits.push(c),
                '_' => {}
                _ => return None,
            }
        }
        let digits = digits.trim_start_matches('0');
        let mut key: Vec<u32> = Vec::new();
        for c in digits.bytes() {
            let mut carry = (c - b'0') as u64;
            for limb in key.iter_mut() {
                let v = *limb as u64 * 10 + carry;
                *limb = v as u32;
                carry = v >> 32;
            }
            if carry > 0 {
                key.push(carry as u32);
            }
        }
        let text = if digits.is_empty() {
            "0".to_string()
        } else if neg {
            format!("-{}", digits)
        } else {
            digits.to_string()
        };
        Some(Seed { text, key })
    }
}

/// Python's str.isspace() for the characters that matter here.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's int() for small values (moves, dimensions, turns).
fn py_int(s: &str) -> Option<i64> {
    let seed = Seed::parse(s)?;
    seed.text.parse::<i64>().ok().or_else(|| {
        // Too big for i64: still a valid int, saturate (it is out of range anyway).
        Some(if seed.text.starts_with('-') { i64::MIN } else { i64::MAX })
    })
}

// ---------------------------------------------------------------------------
// Map setup shared by games and the --map-test mode
// ---------------------------------------------------------------------------

/// Mirrors the start of run_game: seeded rng, default size, generate_map.
fn setup_map(width: Option<i64>, height: Option<i64>, n: usize, seed: &Seed) -> game::Map {
    let mut rng = PyRandom::from_key(&seed.key);
    let (mut w, mut h) = (width, height);
    if w.is_none() || h.is_none() {
        let side = rng.choice(&[20i64, 25, 30, 35, 40, 45, 50]);
        // `width or side`: 0 counts as missing too.
        w = Some(w.filter(|&v| v != 0).unwrap_or(side));
        h = Some(h.filter(|&v| v != 0).unwrap_or(side));
    }
    generate_map(w.unwrap(), h.unwrap(), n, &mut rng)
}

fn random_seed() -> Seed {
    // Like random.randrange(2 ** 31) on an OS-seeded generator.
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut h = RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    Seed::from_u64(h.finish() % (1u64 << 31))
}

// ---------------------------------------------------------------------------
// Running a game
// ---------------------------------------------------------------------------

struct Options {
    bots: Vec<String>,
    width: Option<i64>,
    height: Option<i64>,
    seed: Option<Seed>,
    no_timeout: bool,
    quiet: bool,
    replay_dir: Option<String>,
    write_html: bool,
    max_turns: Option<i64>,
    log_dir: Option<String>,
}

fn parse_moves(line: &str, state: &GameState, player: usize, dirs: &mut [u8], seen: &mut [bool]) {
    let tokens: Vec<&str> = line.split(is_py_space).filter(|t| !t.is_empty()).collect();
    let w = state.width as i64;
    let h = state.height as i64;
    let mut i = 0;
    while i + 2 < tokens.len() {
        let parsed = (py_int(tokens[i]), py_int(tokens[i + 1]), py_int(tokens[i + 2]));
        i += 3;
        let (x, y, d) = match parsed {
            (Some(x), Some(y), Some(d)) => (x, y, d),
            _ => continue,
        };
        if 0 <= x && x < w && 0 <= y && y < h && 0 <= d && d <= 4 {
            let idx = (y * w + x) as usize;
            if state.owner[idx] as usize == player && !seen[idx] {
                seen[idx] = true;
                dirs[idx] = d as u8;
            }
        }
    }
}

fn path_join(dir: &str, name: &str) -> String {
    if dir.ends_with('/') {
        format!("{}{}", dir, name)
    } else {
        format!("{}/{}", dir, name)
    }
}

fn run_game(opt: &Options) -> String {
    let n = opt.bots.len();
    let seed = opt.seed.clone().unwrap_or_else(random_seed);
    let map = setup_map(opt.width, opt.height, n, &seed);
    let (w, h) = (map.width, map.height);
    let mut state = GameState::new(&map);
    let turn_limit = match opt.max_turns {
        Some(t) if t != 0 => t,
        _ => (10.0 * ((w * h) as f64).sqrt()) as i64,
    };

    let log_dir = opt.log_dir.as_deref().filter(|d| !d.is_empty());
    if let Some(dir) = log_dir {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut bots: Vec<Bot> = opt
        .bots
        .iter()
        .enumerate()
        .map(|(i, cmd)| {
            let log_path = log_dir.map(|d| path_join(d, &format!("bot-{}.log", i + 1)));
            Bot::new(i + 1, cmd, log_path.as_deref(), opt.no_timeout)
        })
        .collect();

    let quiet = opt.quiet;
    let say = |msg: &str| {
        if !quiet {
            let mut out = std::io::stdout().lock();
            let _ = writeln!(out, "{}", msg);
            let _ = out.flush();
        }
    };

    say(&format!(
        "Map {}x{}, seed {}, {} players, turn limit {}",
        w, h, seed.text, n, turn_limit
    ));

    // --- init ---
    let mut prod_line = String::with_capacity(w * h * 3);
    for (i, &p) in map.production.iter().enumerate() {
        if i > 0 {
            prod_line.push(' ');
        }
        game::push_int(&mut prod_line, p as i64);
    }
    let map_line = state.serialize();
    for b in bots.iter_mut() {
        b.send(&b.tag.to_string());
        b.send(&format!("{} {}", w, h));
        b.send(&prod_line);
        b.send(&map_line);
    }
    let deadline = turn_deadline(INIT_TIMEOUT);
    for b in bots.iter_mut() {
        if b.alive {
            if let Some(name) = b.recv(deadline) {
                let name: String = name.trim_matches(is_py_space).chars().take(30).collect();
                b.name = if name.is_empty() { b.command.clone() } else { name };
            }
        }
    }
    let mut death_turn: Vec<Option<i64>> = vec![None; n + 1];
    for k in 0..n {
        if !bots[k].alive {
            say(&format!(
                "Player {} ({}) {} during init",
                bots[k].tag,
                bots[k].name,
                bots[k].error.as_deref().unwrap_or("")
            ));
            state.eject(bots[k].tag);
            death_turn[bots[k].tag] = Some(0);
        }
    }

    let mut frames: Vec<(Vec<u8>, Vec<i32>)> = vec![(state.owner.clone(), state.strength.clone())];
    let mut move_frames: Vec<String> = Vec::new();
    let mut turn: i64 = 0;
    let mut dirs = vec![0u8; w * h];
    let mut seen = vec![false; w * h];

    while turn < turn_limit {
        let (territory, _, _) = state.stats();
        for p in 1..=n {
            if death_turn[p].is_none() && territory[p] == 0 {
                death_turn[p] = Some(turn);
            }
        }
        let active: Vec<usize> =
            (0..n).filter(|&k| bots[k].alive && death_turn[bots[k].tag].is_none()).collect();
        let living = (1..=n).filter(|&p| death_turn[p].is_none()).count();
        if living <= if n == 1 { 0 } else { 1 } {
            break;
        }

        turn += 1;
        let map_line = state.serialize();
        for &k in &active {
            bots[k].send(&map_line);
        }
        dirs.iter_mut().for_each(|d| *d = 0);
        seen.iter_mut().for_each(|s| *s = false);
        for &k in &active {
            let deadline = turn_deadline(TURN_TIMEOUT);
            let line = bots[k].recv(deadline);
            let tag = bots[k].tag;
            match line {
                None => {
                    say(&format!(
                        "Turn {}: player {} ({}) {} and was ejected",
                        turn,
                        tag,
                        bots[k].name,
                        bots[k].error.as_deref().unwrap_or("")
                    ));
                    state.eject(tag);
                    death_turn[tag] = Some(turn);
                }
                Some(line) => parse_moves(&line, &state, tag, &mut dirs, &mut seen),
            }
        }
        move_frames.push(dirs.iter().map(|&d| (b'0' + d) as char).collect());

        state.step(&dirs);
        frames.push((state.owner.clone(), state.strength.clone()));
    }

    let (territory, total_strength, _) = state.stats();
    for p in 1..=n {
        if death_turn[p].is_none() && territory[p] == 0 {
            death_turn[p] = Some(turn);
        }
    }

    for b in bots.iter_mut() {
        b.kill();
    }

    let mut order: Vec<usize> = (1..=n).collect();
    order.sort_by_key(|&p| {
        let dt = death_turn[p];
        (
            if dt.is_none() { 0 } else { 1 },
            -dt.unwrap_or(0),
            -territory[p],
            -total_strength[p],
        )
    });

    let mut results: Vec<Value> = Vec::new();
    let mut summary_rows: Vec<(usize, usize, String, i64, i64, String)> = Vec::new();
    for (rank, &p) in order.iter().enumerate() {
        let b = &bots[p - 1];
        let mut status = match death_turn[p] {
            None => "alive".to_string(),
            Some(t) => format!("died turn {}", t),
        };
        if let Some(e) = &b.error {
            status = format!("{} ({})", status, e);
        }
        summary_rows.push((rank + 1, p, b.name.clone(), territory[p], total_strength[p], status.clone()));
        results.push(Value::Object(vec![
            ("rank", Value::Int(rank as i64 + 1)),
            ("player", Value::Int(p as i64)),
            ("name", Value::Str(b.name.clone())),
            ("command", Value::Str(b.command.clone())),
            ("territory", Value::Int(territory[p])),
            ("strength", Value::Int(total_strength[p])),
            ("status", Value::Str(status)),
        ]));
    }
    let results_value = Value::List(results);

    let mut replay_path: Option<String> = None;
    let mut html_path: Option<String> = None;
    if let Some(dir) = opt.replay_dir.as_deref().filter(|d| !d.is_empty()) {
        let _ = std::fs::create_dir_all(dir);
        let base = path_join(dir, &format!("{}-{}", local_stamp(), seed.text));
        let rp = format!("{}.hlt", base);
        let replay = replay_json(&seed, &map, &bots, &frames, &move_frames, &results_value);
        if let Err(e) = std::fs::write(&rp, &replay) {
            eprintln!("cannot write {}: {}", rp, e);
            std::process::exit(1);
        }
        replay_path = Some(rp);
        if opt.write_html {
            let hp = format!("{}.html", base);
            write_replay_html(&replay, &hp);
            html_path = Some(hp);
        }
    }

    say("");
    say(&format!("Game over after {} turns.", turn));
    for (rank, player, name, terr, strength, status) in &summary_rows {
        say(&format!(
            "  #{}  P{} {:<24} territory {:>4}  strength {:>6}  {}",
            rank, player, name, terr, strength, status
        ));
    }
    if let Some(p) = &replay_path {
        say(&format!("Replay: {}", p));
    }
    if let Some(p) = &html_path {
        say(&format!("Viewer: {}", p));
    }

    let opt_str = |s: &Option<String>| match s {
        Some(s) => Value::Str(s.clone()),
        None => Value::Null,
    };
    let summary = Value::Object(vec![
        ("seed", Value::BigInt(seed.text.clone())),
        ("width", Value::Int(w as i64)),
        ("height", Value::Int(h as i64)),
        ("turns", Value::Int(turn)),
        ("results", results_value),
        ("replay", opt_str(&replay_path)),
        ("html", opt_str(&html_path)),
    ]);
    json::dumps(&summary, &json::DEFAULT)
}

fn replay_json(
    seed: &Seed,
    map: &game::Map,
    bots: &[Bot],
    frames: &[(Vec<u8>, Vec<i32>)],
    move_frames: &[String],
    results: &Value,
) -> String {
    // Written directly rather than through Value: frames are most of the file.
    let mut out = String::with_capacity(frames.len() * map.width * map.height * 6 + 4096);
    let push_ints = |out: &mut String, it: &mut dyn Iterator<Item = i64>| {
        out.push('[');
        let mut first = true;
        for v in it {
            if !first {
                out.push(',');
            }
            first = false;
            game::push_int(out, v);
        }
        out.push(']');
    };
    out.push_str("{\"version\":");
    game::push_int(&mut out, VERSION);
    out.push_str(&format!(
        ",\"width\":{},\"height\":{},\"num_players\":{},\"num_frames\":{},\"seed\":{},\"player_names\":",
        map.width,
        map.height,
        bots.len(),
        frames.len(),
        seed.text
    ));
    let names = Value::List(bots.iter().map(|b| Value::Str(b.name.clone())).collect());
    out.push_str(&json::dumps(&names, &json::COMPACT));
    out.push_str(",\"productions\":");
    push_ints(&mut out, &mut map.production.iter().map(|&v| v as i64));
    out.push_str(",\"frames\":[");
    for (k, (o, s)) in frames.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        out.push('[');
        push_ints(&mut out, &mut o.iter().map(|&v| v as i64));
        out.push(',');
        push_ints(&mut out, &mut s.iter().map(|&v| v as i64));
        out.push(']');
    }
    out.push_str("],\"moves\":[");
    for (k, m) in move_frames.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(m);
        out.push('"');
    }
    out.push_str("],\"results\":");
    out.push_str(&json::dumps(results, &json::COMPACT));
    out.push('}');
    out
}

/// viewer.html sits in the repository root, three levels above the binary
/// (engine/target/release/halite); fall back to the source tree location.
fn viewer_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        let exe = exe.canonicalize().unwrap_or(exe);
        for dir in exe.ancestors().skip(1) {
            let p = dir.join("viewer.html");
            if p.is_file() {
                return p;
            }
        }
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("viewer.html")
}

fn write_replay_html(replay: &str, path: &str) {
    let template = match std::fs::read_to_string(viewer_path()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read viewer.html: {}", e);
            std::process::exit(1);
        }
    };
    let tag = format!("<script>window.EMBEDDED_REPLAY = {};</script>", replay.replace("</", "<\\/"));
    if let Err(e) = std::fs::write(path, template.replace("<!--EMBEDDED_REPLAY-->", &tag)) {
        eprintln!("cannot write {}: {}", path, e);
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Local time for replay names (datetime.now().strftime("%Y%m%d-%H%M%S"))
// ---------------------------------------------------------------------------

#[repr(C)]
struct Tm {
    tm_sec: i32,
    tm_min: i32,
    tm_hour: i32,
    tm_mday: i32,
    tm_mon: i32,
    tm_year: i32,
    tm_wday: i32,
    tm_yday: i32,
    tm_isdst: i32,
    tm_gmtoff: i64,
    tm_zone: *const u8,
}

extern "C" {
    fn localtime_r(t: *const i64, out: *mut Tm) -> *mut Tm;
}

fn local_stamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let mut tm = Tm {
        tm_sec: 0,
        tm_min: 0,
        tm_hour: 0,
        tm_mday: 0,
        tm_mon: 0,
        tm_year: 0,
        tm_wday: 0,
        tm_yday: 0,
        tm_isdst: 0,
        tm_gmtoff: 0,
        tm_zone: std::ptr::null(),
    };
    // SAFETY: localtime_r only writes into the struct we pass.
    unsafe {
        localtime_r(&now, &mut tm);
    }
    format!(
        "{:04}{:02}{:02}-{:02}{:02}{:02}",
        tm.tm_year + 1900,
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec
    )
}

// ---------------------------------------------------------------------------
// Hidden test modes used by check_engine.py
// ---------------------------------------------------------------------------

/// --step-test: each stdin line is "W H N owners.. strengths.. productions..
/// dirs.." (W*H values each); prints "owners.. strengths.." after one step.
fn step_test() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let v: Vec<i64> = line.split_whitespace().map(|t| t.parse().expect("int")).collect();
        if v.is_empty() {
            continue;
        }
        let (w, h, n) = (v[0] as usize, v[1] as usize, v[2] as usize);
        let size = w * h;
        let field = |k: usize| &v[3 + k * size..3 + (k + 1) * size];
        let mut state = GameState {
            width: w,
            height: h,
            num_players: n,
            owner: field(0).iter().map(|&x| x as u8).collect(),
            strength: field(1).iter().map(|&x| x as i32).collect(),
            production: field(2).iter().map(|&x| x as i32).collect(),
        };
        let dirs: Vec<u8> = field(3).iter().map(|&x| x as u8).collect();
        state.step(&dirs);
        let mut s = String::new();
        for &o in &state.owner {
            game::push_int(&mut s, o as i64);
            s.push(' ');
        }
        for &x in &state.strength {
            game::push_int(&mut s, x as i64);
            s.push(' ');
        }
        writeln!(out, "{}", s.trim_end()).unwrap();
    }
}

/// --map-test: each stdin line is "W H N SEED" (W, H may be "-" for the
/// default size); prints "W H productions.. strengths.. sx sy sx sy ..".
fn map_test() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() {
            continue;
        }
        let dim = |s: &str| if s == "-" { None } else { Some(s.parse::<i64>().expect("int")) };
        let seed = Seed::parse(t[3]).expect("seed");
        let map = setup_map(dim(t[0]), dim(t[1]), t[2].parse().expect("int"), &seed);
        let mut s = format!("{} {}", map.width, map.height);
        for &x in map.production.iter().chain(map.strength.iter()) {
            s.push(' ');
            game::push_int(&mut s, x as i64);
        }
        for &(x, y) in &map.starts {
            s.push_str(&format!(" {} {}", x, y));
        }
        writeln!(out, "{}", s).unwrap();
    }
}

// ---------------------------------------------------------------------------
// CLI (the same options as halite.py's argparse)
// ---------------------------------------------------------------------------

const USAGE: &str = "usage: halite [-h] [-d DIMENSIONS] [-s SEED] [-t] [-q] [-o REPLAY_DIR] [--no-replay] [--no-html] [--turns TURNS] [--log-dir LOG_DIR] [--no-logs] bots [bots ...]";

const HELP: &str = "
Halite I clone game environment (Rust engine)

positional arguments:
  bots                  bot commands, e.g. \"./starter-kit/rust/target/release/my_bot\"

options:
  -h, --help            show this help message and exit
  -d, --dimensions DIMENSIONS
                        map size as \"WIDTH HEIGHT\" (default: random 20 to 50)
  -s, --seed SEED       map seed
  -t, --no-timeout      disable timeouts (for debugging)
  -q, --quiet           print only a JSON summary
  -o, --replay-dir REPLAY_DIR
                        where to write replays
  --no-replay           do not write replay files
  --no-html             skip the standalone HTML replay
  --turns TURNS         override the turn limit
  --log-dir LOG_DIR     where bot stderr logs go
  --no-logs             discard bot stderr";

fn usage_error(msg: &str) -> ! {
    eprintln!("{}\nhalite: error: {}", USAGE, msg);
    std::process::exit(2);
}

fn parse_args(args: Vec<String>) -> Options {
    const LONG: [&str; 11] = [
        "--help", "--dimensions", "--seed", "--no-timeout", "--quiet", "--replay-dir",
        "--no-replay", "--no-html", "--turns", "--log-dir", "--no-logs",
    ];
    let mut bots = Vec::new();
    let mut dims: Option<String> = None;
    let mut seed: Option<String> = None;
    let mut turns: Option<String> = None;
    let mut replay_dir = "replays".to_string();
    let mut log_dir = "logs".to_string();
    let (mut no_timeout, mut quiet, mut no_replay, mut no_html) = (false, false, false, false);
    let mut no_logs = false;

    let mut it = args.into_iter();
    let mut only_positional = false;
    while let Some(arg) = it.next() {
        if only_positional || !arg.starts_with('-') || arg == "-" || arg.parse::<f64>().is_ok() {
            bots.push(arg);
            continue;
        }
        if arg == "--" {
            only_positional = true;
            continue;
        }
        // Normalise to (long name, attached value).
        let (name, attached): (&str, Option<String>) = if arg.starts_with("--") {
            let (key, val) = match arg.find('=') {
                Some(p) => (&arg[..p], Some(arg[p + 1..].to_string())),
                None => (&arg[..], None),
            };
            let matches: Vec<&str> = LONG.iter().cloned().filter(|l| l.starts_with(key)).collect();
            let exact = LONG.iter().cloned().find(|l| *l == key);
            match (exact, matches.len()) {
                (Some(l), _) => (l, val),
                (None, 1) => (matches[0], val),
                (None, 0) => usage_error(&format!("unrecognized arguments: {}", arg)),
                _ => usage_error(&format!("ambiguous option: {} could match {}", key, matches.join(", "))),
            }
        } else {
            // Short options, possibly clustered (-tq) or with a value (-s7).
            let chars: Vec<char> = arg[1..].chars().collect();
            let mut k = 0;
            let mut result: Option<(&str, Option<String>)> = None;
            while k < chars.len() {
                let long = match chars[k] {
                    'h' => "--help",
                    't' => "--no-timeout",
                    'q' => "--quiet",
                    'd' => "--dimensions",
                    's' => "--seed",
                    'o' => "--replay-dir",
                    _ => usage_error(&format!("unrecognized arguments: {}", arg)),
                };
                let takes_value = matches!(long, "--dimensions" | "--seed" | "--replay-dir");
                if takes_value {
                    let rest: String = chars[k + 1..].iter().collect();
                    let rest = rest.strip_prefix('=').map(str::to_string).unwrap_or(rest);
                    result = Some((long, if rest.is_empty() { None } else { Some(rest) }));
                    break;
                }
                if k + 1 == chars.len() {
                    result = Some((long, None));
                } else {
                    apply_flag(long, &mut no_timeout, &mut quiet, &mut no_replay, &mut no_html);
                }
                k += 1;
            }
            result.unwrap()
        };
        match name {
            "--help" => {
                println!("{}\n{}", USAGE, HELP);
                std::process::exit(0);
            }
            "--no-logs" if attached.is_none() => no_logs = true,
            "--dimensions" | "--seed" | "--replay-dir" | "--turns" | "--log-dir" => {
                let value = attached.or_else(|| it.next()).unwrap_or_else(|| {
                    usage_error(&format!("argument {}: expected one argument", name))
                });
                match name {
                    "--dimensions" => dims = Some(value),
                    "--seed" => seed = Some(value),
                    "--turns" => turns = Some(value),
                    "--log-dir" => log_dir = value,
                    _ => replay_dir = value,
                }
            }
            _ => {
                if attached.is_some() {
                    usage_error(&format!("argument {}: ignored explicit argument", name));
                }
                apply_flag(name, &mut no_timeout, &mut quiet, &mut no_replay, &mut no_html);
            }
        }
    }
    if bots.is_empty() {
        usage_error("the following arguments are required: bots");
    }
    if bots.len() > MAX_PLAYERS {
        eprintln!("ValueError: Halite supports 1 to {} players", MAX_PLAYERS);
        std::process::exit(1);
    }

    let (mut width, mut height) = (None, None);
    if let Some(d) = dims {
        let parts: Vec<String> = d.replace('x', " ").split(is_py_space).filter(|t| !t.is_empty()).map(String::from).collect();
        if parts.len() < 2 {
            eprintln!("IndexError: -d expects \"WIDTH HEIGHT\"");
            std::process::exit(1);
        }
        let parse = |s: &str| {
            py_int(s).unwrap_or_else(|| {
                eprintln!("ValueError: invalid literal for int() with base 10: '{}'", s);
                std::process::exit(1);
            })
        };
        width = Some(parse(&parts[0]));
        height = Some(parse(&parts[1]));
    }
    let seed = seed.map(|s| {
        Seed::parse(&s).unwrap_or_else(|| usage_error(&format!("argument -s/--seed: invalid int value: '{}'", s)))
    });
    let max_turns = turns.map(|s| {
        py_int(&s).unwrap_or_else(|| usage_error(&format!("argument --turns: invalid int value: '{}'", s)))
    });

    Options {
        bots,
        width,
        height,
        seed,
        no_timeout,
        quiet,
        replay_dir: if no_replay { None } else { Some(replay_dir) },
        write_html: !no_html,
        max_turns,
        log_dir: if no_logs { None } else { Some(log_dir) },
    }
}

fn apply_flag(name: &str, no_timeout: &mut bool, quiet: &mut bool, no_replay: &mut bool, no_html: &mut bool) {
    match name {
        "--no-timeout" => *no_timeout = true,
        "--quiet" => *quiet = true,
        "--no-replay" => *no_replay = true,
        "--no-html" => *no_html = true,
        "--help" => {
            println!("{}\n{}", USAGE, HELP);
            std::process::exit(0);
        }
        _ => {}
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--step-test") => return step_test(),
        Some("--map-test") => return map_test(),
        _ => {}
    }
    let opt = parse_args(args);
    let summary = run_game(&opt);
    if opt.quiet {
        println!("{}", summary);
    }
}
