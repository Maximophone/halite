//! Batched self-play using the project's exact simulator, with no subprocess per bot.
#[allow(dead_code)]
#[path = "../../engine/src/game.rs"]
mod game;
mod policy;
mod policy_v2;
#[allow(dead_code)]
#[path = "../../engine/src/pyrandom.rs"]
mod pyrandom;
mod sim;

use policy_v2::{Board, Policy};
use std::{
    collections::HashMap,
    fs,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

struct Job {
    a: Arc<Policy>,
    b: Arc<Policy>,
    seed: u32,
    size: usize,
    swap: bool,
}

fn frame(env: &sim::Env, dirs: &[u8]) -> String {
    let (territory, strength, production) = env.state.stats();
    format!("{{\"owners\":{:?},\"strengths\":{:?},\"moves\":{:?},\"territory\":{:?},\"strength\":{:?},\"production\":{:?}}}",
        env.state.owner, env.state.strength, dirs, territory, strength, production)
}

fn play(job: &Job, replay: Option<&str>) -> String {
    let mut env = sim::Env::new(job.seed, job.size, job.size, 2);
    let mut dirs = vec![0; env.state.owner.len()];
    let mut frames = Vec::new();
    let mut moving = 0usize;
    let mut decisions = 0usize;
    let me = if job.swap { 2 } else { 1 };
    let other = 3 - me;
    while !env.done {
        let b = Board {
            width: env.width(),
            height: env.height(),
            owner: &env.state.owner,
            strength: &env.state.strength,
            production: &env.state.production,
        };
        job.a.act(&b, me, env.turn, &mut dirs);
        job.b.act(&b, other, env.turn, &mut dirs);
        for (i, &o) in b.owner.iter().enumerate() {
            if o == me {
                decisions += 1;
                moving += (dirs[i] != 0) as usize;
            }
        }
        if replay.is_some() {
            frames.push(frame(&env, &dirs));
        }
        env.step(&dirs);
    }
    if let Some(path) = replay {
        dirs.fill(0);
        frames.push(frame(&env, &dirs));
        fs::write(path, format!("{{\"width\":{},\"height\":{},\"seed\":{},\"candidate_player\":{},\"production\":{:?},\"frames\":[{}]}}",
            env.width(), env.height(), job.seed, me, env.state.production, frames.join(","))).unwrap();
    }
    let (territory, strength, _) = env.state.stats();
    // Exact ties receive 0.5; avoid learning the engine's player-ID tiebreak.
    let key = |p: usize| {
        (
            env.death[p].is_none(),
            env.death[p].unwrap_or(0),
            territory[p],
            strength[p],
        )
    };
    let result = match key(me as usize).cmp(&key(other as usize)) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => 0.0,
        _ => 0.5,
    };
    format!(
        "{{\"score\":{},\"turns\":{},\"territory\":{},\"coverage\":{},\"move_rate\":{}}}",
        result,
        env.turn,
        territory[me as usize],
        territory[me as usize] as f64 / env.state.owner.len() as f64,
        moving as f64 / decisions.max(1) as f64
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 4 {
        eprintln!("usage: evo-eval JOBS.tsv [WORKERS] [REPLAY.json (one job only)]");
        std::process::exit(2);
    }
    let workers = args
        .get(2)
        .map(|s| s.parse::<usize>().expect("workers must be an integer"))
        .unwrap_or(1)
        .max(1);
    let mut cache = HashMap::new();
    let mut jobs = Vec::new();
    for line in fs::read_to_string(&args[1]).expect("read jobs").lines() {
        let v: Vec<_> = line.split('\t').collect();
        assert_eq!(
            v.len(),
            5,
            "jobs: candidate TAB opponent TAB seed TAB size TAB swap(0/1)"
        );
        for path in &v[..2] {
            if !cache.contains_key(*path) {
                let p = Policy::load(&fs::read_to_string(path).expect("read policy"))
                    .expect("load policy");
                cache.insert(path.to_string(), Arc::new(p));
            }
        }
        let size = v[3].parse().expect("size");
        assert!((6..=100).contains(&size), "size must be 6..100");
        assert!(v[4] == "0" || v[4] == "1", "swap must be 0 or 1");
        jobs.push(Job {
            a: cache[v[0]].clone(),
            b: cache[v[1]].clone(),
            seed: v[2].parse().expect("seed"),
            size,
            swap: v[4] == "1",
        });
    }
    if let Some(replay) = args.get(3) {
        assert_eq!(jobs.len(), 1, "replay requires one job");
        println!("{}", play(&jobs[0], Some(replay)));
        return;
    }
    let next = AtomicUsize::new(0);
    let results = thread::scope(|scope| {
        let handles: Vec<_> = (0..workers.min(jobs.len()))
            .map(|_| {
                let (jobs, next) = (&jobs, &next);
                scope.spawn(move || {
                    let mut results = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= jobs.len() {
                            break;
                        }
                        results.push((i, play(&jobs[i], None)));
                    }
                    results
                })
            })
            .collect();
        let mut results: Vec<_> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker panicked"))
            .collect();
        results.sort_by_key(|r| r.0);
        results
    });
    for (_, line) in results {
        println!("{line}");
    }
}
