//! NeuroDelta: NeuroGamma's search (prediction weight 1.0, see src/neuro_search.rs) with a bigger
//! network: 64 channels x 12 blocks (about 510k parameters, `weights/neuro_delta.hnn`), cloned
//! from DexBot for 30k steps (validation loss 0.180, NeuroAlpha's network 0.217). Imitating DexBot
//! better mostly sharpens the predictions of enemy moves that the search relies on: vs DexBot,
//! 1v1, 78% on seed 23 (NeuroGamma 62%) and 73% on fresh seed 29 over 200 games (67%). The
//! convolutions run on 4 threads (NEURO_THREADS overrides).

fn main() {
    hlt::neuro_search::run("NeuroDelta", 1.0, include_bytes!("../../weights/neuro_delta.hnn"));
}
