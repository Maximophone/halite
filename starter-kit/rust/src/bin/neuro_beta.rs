//! NeuroBeta: NeuroAlpha's network plus ApexBeta's one-turn combat search at the front, with
//! ApexBeta's prediction weight 0.6 (see src/neuro_search.rs). Frozen as released; NeuroGamma
//! is the same bot trusting the network's enemy predictions fully.

fn main() {
    hlt::neuro_search::run("NeuroBeta", 0.6, include_bytes!("../../weights/neuro_alpha.hnn"));
}
