//! NeuroGamma: NeuroBeta with the predicted enemy moves weighted 1.0 instead of 0.6 (see
//! src/neuro_search.rs). ApexBeta's 0.6 was tuned for predictions from a hand-written policy; the
//! network imitates real play and predicts better. vs DexBot, 1v1: 62% (NeuroBeta 53%) on seed 23,
//! 67% (59%) on fresh seed 29 with 200 games.

fn main() {
    hlt::neuro_search::run("NeuroGamma", 1.0, include_bytes!("../../weights/neuro_alpha.hnn"));
}
