//! A sparring partner with the classic beginner improvements:
//!   1. Capture a neighbouring site when strong enough, preferring the best
//!      production-to-strength ratio (and enemy sites, for overkill damage).
//!   2. Otherwise stay still to grow until strength >= 5x production.
//!   3. Interior pieces move toward the nearest border instead of idling.
//! Good enough to beat RandomBot every time; beatable with some thought.
//!
//! Move-for-move port of the original Python ImprovedBot (identical replays).

use hlt::*;

fn site_value(site: Site) -> f64 {
    if site.owner == 0 && site.strength > 0 {
        return site.production as f64 / site.strength as f64;
    }
    // Enemy or empty neutral site: cheap to take, and fighting on an enemy
    // border damages every adjacent enemy piece at once.
    site.production as f64 + 1.0
}

fn nearest_border_direction(map: &GameMap, my_id: u8, loc: Location) -> u8 {
    let reach = map.width.max(map.height) / 2;
    let (mut best_dir, mut best_dist) = (NORTH, reach);
    for &d in CARDINALS.iter() {
        let mut dist = 0;
        let mut cur = loc;
        while map.get_site(cur, STILL).owner == my_id && dist < reach {
            dist += 1;
            cur = map.get_location(cur, d);
        }
        if dist < best_dist {
            best_dir = d;
            best_dist = dist;
        }
    }
    best_dir
}

fn move_for(map: &GameMap, my_id: u8, loc: Location) -> Move {
    let site = map.get_site(loc, STILL);
    // Best (value, direction); ties go to the larger direction, like the
    // Python original's max() over (value, direction) tuples.
    let mut best: Option<(f64, u8)> = None;
    let mut border = false;
    for &d in CARDINALS.iter() {
        let n = map.get_site(loc, d);
        if n.owner != my_id {
            border = true;
            if n.strength < site.strength {
                let v = site_value(n) + if n.owner != 0 { 1.0 } else { 0.0 };
                if best.map_or(true, |(bv, bd)| v > bv || (v == bv && d > bd)) {
                    best = Some((v, d));
                }
            }
        }
    }
    if let Some((_, d)) = best {
        return Move { loc, direction: d };
    }
    if (site.strength as u32) < site.production as u32 * 5 {
        return Move { loc, direction: STILL };
    }
    if !border {
        return Move { loc, direction: nearest_border_direction(map, my_id, loc) };
    }
    Move { loc, direction: STILL }
}

fn main() {
    let (my_id, _map, mut conn) = get_init();
    send_init("ImprovedBot");
    loop {
        let map = conn.get_frame();
        let mut moves = Vec::new();
        for y in 0..map.height {
            for x in 0..map.width {
                let loc = Location { x, y };
                if map.get_site(loc, STILL).owner == my_id {
                    moves.push(move_for(&map, my_id, loc));
                }
            }
        }
        conn.send_frame(&moves);
    }
}
