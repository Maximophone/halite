//! Wrap-around 3x3 convolution as a candle custom op with hand-written Metal kernels.
//!
//! Same semantics as `conv` in main.rs (x (B,H,W,Cin) channel-last, w (9*Cin, Cout), tap
//! k = 3 * (dy + 1) + (dx + 1) reading (y + dy * d, x + dx * d) mod the map size), without the
//! 9x im2col tensor: candle's generic gather moved about 4 GB/s and dominated training time.
//! Kernels are compiled from source at runtime (no Xcode needed).
//!   - conv_fwd: one thread per output value. Also computes the input gradient, since that is
//!     the same convolution of the output gradient with flipped taps and transposed weights.
//!   - conv_wgrad: weight gradient, one threadgroup per (tap, chunk of pixels) reducing through
//!     threadgroup memory; the chunks are summed by candle afterwards.

use candle_core::backend::BackendStorage;
use candle_core::{bail, CpuStorage, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLSize;
use std::collections::HashMap;

const SRC: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
struct P { uint b, h, w, cin, cout, d, chunks, bias, residual, relu; };

// out = conv(in, wt) [+ bias[b, co]] [+ in[pixel, co]] [then relu]; bias is (B, Cout), one row per image.
kernel void conv_fwd(device const float* in [[buffer(0)]], device const float* wt [[buffer(1)]],
                     device float* out [[buffer(2)]], constant P& p [[buffer(3)]],
                     device const float* bias [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= p.b * p.h * p.w * p.cout) return;
    uint co = gid % p.cout;
    uint pix = gid / p.cout;
    uint x = pix % p.w;
    uint t = pix / p.w;
    uint y = t % p.h;
    uint b = t / p.h;
    float acc = 0.0;
    for (uint ky = 0; ky < 3; ky++) {
        uint sy = (y + p.h * p.d + ky * p.d - p.d) % p.h;
        for (uint kx = 0; kx < 3; kx++) {
            uint sx = (x + p.w * p.d + kx * p.d - p.d) % p.w;
            device const float* src = in + ((b * p.h + sy) * p.w + sx) * p.cin;
            device const float* wk = wt + (ky * 3 + kx) * p.cin * p.cout + co;
            for (uint ci = 0; ci < p.cin; ci++) {
                acc += src[ci] * wk[ci * p.cout];
            }
        }
    }
    if (p.bias) acc += bias[b * p.cout + co];
    if (p.residual) acc += in[pix * p.cin + co];
    if (p.relu) acc = max(acc, 0.0f);
    out[gid] = acc;
}

// out = gy * (y > 0): the gradient through a relu, given its output y. n = p.b elements.
kernel void relu_mask(device const float* gy [[buffer(0)]], device const float* y [[buffer(1)]],
                      device float* out [[buffer(2)]], constant P& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid >= p.b) return;
    out[gid] = y[gid] > 0.0f ? gy[gid] : 0.0f;
}

// One threadgroup per (tap k, pixel chunk s). Tiles of 32 pixels of the shifted input and of the
// output gradient go to threadgroup memory; each of the 256 threads accumulates up to 16 of the
// Cin*Cout weights of tap k. Needs Cin, Cout <= 64 and Cin*Cout <= 4096 (checked on the Rust side).
kernel void conv_wgrad(device const float* in [[buffer(0)]], device const float* gy [[buffer(1)]],
                       device float* part [[buffer(2)]], constant P& p [[buffer(3)]],
                       uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[32 * 64];
    threadgroup float gs[32 * 64];
    uint k = tg % 9;
    uint s = tg / 9;
    uint nout = p.cin * p.cout;
    float acc[16];
    for (uint j = 0; j < 16; j++) acc[j] = 0.0;
    uint oy = (k / 3) * p.d + p.h * p.d - p.d;
    uint ox = (k % 3) * p.d + p.w * p.d - p.d;
    uint npix = p.b * p.h * p.w;
    uint per = (npix + p.chunks - 1) / p.chunks;
    uint start = s * per;
    uint end = min(npix, start + per);
    for (uint t0 = start; t0 < end; t0 += 32) {
        uint n = min(32u, end - t0);
        for (uint i = tid; i < n * p.cin; i += 256) {
            uint pix = t0 + i / p.cin;
            uint x = pix % p.w;
            uint t = pix / p.w;
            uint y = t % p.h;
            uint b = t / p.h;
            xs[i] = in[((b * p.h + (y + oy) % p.h) * p.w + (x + ox) % p.w) * p.cin + i % p.cin];
        }
        for (uint i = tid; i < n * p.cout; i += 256) gs[i] = gy[t0 * p.cout + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 16; j++) {
            uint o = tid + j * 256;
            if (o >= nout) break;
            uint ci = o / p.cout;
            uint co = o % p.cout;
            float a = acc[j];
            for (uint q = 0; q < n; q++) a += xs[q * p.cin + ci] * gs[q * p.cout + co];
            acc[j] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 16; j++) {
        uint o = tid + j * 256;
        if (o < nout) part[(s * 9 + k) * nout + o] = acc[j];
    }
}

// The same conv as conv_fwd on the 8x8 matrix units (simdgroup_matrix), as an implicit GEMM:
// out[pixel, co] = sum over taps k and ci of in[neighbour_k(pixel), ci] * wt[k, ci, co].
// A threadgroup (4 simdgroups, 128 threads) owns 32 consecutive pixels and every output channel;
// simdgroup s owns pixels 8s..8s+7. The 9 neighbour addresses of the 32 pixels are computed once;
// per tap the 32 shifted input rows are staged in threadgroup memory (zero padded to cinp, a power
// of two >= 8, so the staging loop has no division) and multiplied 8x8 block by 8x8 block with the
// tap's weights, read straight from device memory when Cin is a multiple of 8.
// Needs Cout a multiple of 8, Cin and Cout <= 32. Threadgroup memory is kept small (about 7 KB) on
// purpose: several threadgroups per GPU core hide the latency of the scattered neighbour reads
// (halving it made this kernel 1.8x faster; 128-pixel tiles with 24 KB were 2x slower).
kernel void conv_fwd_sg(device const float* in [[buffer(0)]], device const float* wt [[buffer(1)]],
                        device float* out [[buffer(2)]], constant P& p [[buffer(3)]],
                        device const float* bias [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float As[32 * 32];
    threadgroup float Ws[16 * 32];   // only for Cin not a multiple of 8 (padded to 16)
    threadgroup int base[9 * 32];   // input offset of each tap's neighbour, -1 past the last pixel
    uint cinp = p.chunks;           // host passes the padded Cin in the chunks field
    uint npix = p.b * p.h * p.w;
    uint pix0 = tg * 32;
    uint nb = p.cout / 8;
    bool direct = (p.cin % 8) == 0;
    for (uint i = tid; i < 9 * 32; i += 128) {
        uint k = i / 32, r = i % 32, pix = pix0 + r;
        int v = -1;
        if (pix < npix) {
            uint x = pix % p.w, t = pix / p.w, y = t % p.h, b = t / p.h;
            uint sy = (y + (k / 3) * p.d + p.h * p.d - p.d) % p.h;
            uint sx = (x + (k % 3) * p.d + p.w * p.d - p.d) % p.w;
            v = int(((b * p.h + sy) * p.w + sx) * p.cin);
        }
        base[i] = v;
    }
    simdgroup_float8x8 acc[8];
    for (uint j = 0; j < 8; j++) acc[j] = simdgroup_float8x8(0);
    uint c = tid % cinp, rstep = 128 / cinp;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 9; k++) {
        for (uint r = tid / cinp; r < 32; r += rstep) {
            int o = base[k * 32 + r];
            As[r * cinp + c] = (o >= 0 && c < p.cin) ? in[o + c] : 0.0f;
        }
        if (!direct) {
            for (uint i = tid; i < cinp * p.cout; i += 128) {
                uint r = i / p.cout;
                Ws[i] = r < p.cin ? wt[(k * p.cin + r) * p.cout + i % p.cout] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float* wk = wt + k * p.cin * p.cout;
        for (uint c0 = 0; c0 < cinp; c0 += 8) {
            if (c0 >= p.cin && direct) break;
            simdgroup_float8x8 a;
            simdgroup_load(a, As, cinp, ulong2(c0, sg * 8));
            for (uint j = 0; j < 8; j++) {
                if (j >= nb) break;
                simdgroup_float8x8 bm;
                if (direct) simdgroup_load(bm, wk, p.cout, ulong2(j * 8, c0));
                else simdgroup_load(bm, Ws, p.cout, ulong2(j * 8, c0));
                simdgroup_multiply_accumulate(acc[j], a, bm, acc[j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = 0; j < 8; j++) {
        if (j >= nb) break;
        simdgroup_store(acc[j], As, p.cout, ulong2(j * 8, sg * 8));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 32 * p.cout; i += 128) {
        uint r = i / p.cout, co = i % p.cout, pix = pix0 + r;
        if (pix >= npix) continue;
        float v = As[i];
        if (p.bias) v += bias[(pix / (p.h * p.w)) * p.cout + co];
        if (p.residual) v += in[pix * p.cin + co];
        if (p.relu) v = max(v, 0.0f);
        out[pix * p.cout + co] = v;
    }
}

// Weight gradient on the matrix units: for tap k, dW_k = shifted(in)^T (Cin x pixels) * gy (pixels x Cout)
// over one chunk of pixels. Threadgroup per (tap, chunk), 4 simdgroups; simdgroup s owns the 8-row
// blocks of input channels s and s + 4. Pixel tiles of 32 are staged in threadgroup memory (input
// padded to cinp, a power of two; addresses walked incrementally, no division per element) and read
// back transposed. Needs Cout a multiple of 8, Cin and Cout <= 32. p.chunks = number of chunks,
// p.relu carries cinp here.
kernel void conv_wgrad_sg(device const float* in [[buffer(0)]], device const float* gy [[buffer(1)]],
                          device float* part [[buffer(2)]], constant P& p [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sm[2048];            // staging (2 x 32 x 32), then the [cinp][cout] result
    threadgroup float* As = sm;            // [32 pixels][cinp]
    threadgroup float* Gs = sm + 32 * 32;  // [32 pixels][cout]
    threadgroup int base[32];
    uint k = tg % 9, s = tg / 9;
    uint cinp = p.relu;
    uint nb = p.cout / 8, nr = cinp / 8;
    simdgroup_float8x8 acc[16];
    for (uint j = 0; j < 16; j++) acc[j] = simdgroup_float8x8(0);
    uint oy = (k / 3) * p.d + p.h * p.d - p.d;
    uint ox = (k % 3) * p.d + p.w * p.d - p.d;
    uint npix = p.b * p.h * p.w;
    uint per = (npix + p.chunks - 1) / p.chunks;
    uint start = s * per, end = min(npix, start + per);
    uint c = tid % cinp, rstep = 128 / cinp;
    uint gc = tid % p.cout;
    for (uint t0 = start; t0 < end; t0 += 32) {
        if (tid < 32) {
            uint pix = t0 + tid;
            int v = -1;
            if (pix < end) {
                uint x = pix % p.w, t = pix / p.w, y = t % p.h, b = t / p.h;
                v = int(((b * p.h + (y + oy) % p.h) * p.w + (x + ox) % p.w) * p.cin);
            }
            base[tid] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = tid / cinp; r < 32; r += rstep) {
            int o = base[r];
            As[r * cinp + c] = (o >= 0 && c < p.cin) ? in[o + c] : 0.0f;
        }
        for (uint r = tid / p.cout; r < 32; r += 128 / p.cout) {
            uint pix = t0 + r;
            Gs[r * p.cout + gc] = pix < end ? gy[pix * p.cout + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p0 = 0; p0 < 32; p0 += 8) {
            for (uint q = 0; q < 2; q++) {
                uint rb = sg + 4 * q;
                if (rb >= nr) break;
                simdgroup_float8x8 a;
                simdgroup_load(a, As, cinp, ulong2(rb * 8, p0), true);
                for (uint j = 0; j < 8; j++) {
                    if (j >= nb) break;
                    simdgroup_float8x8 g;
                    simdgroup_load(g, Gs, p.cout, ulong2(j * 8, p0));
                    simdgroup_multiply_accumulate(acc[q * 8 + j], a, g, acc[q * 8 + j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // results to threadgroup memory as [cinp][cout], then the real rows to the output
    for (uint q = 0; q < 2; q++) {
        uint rb = sg + 4 * q;
        if (rb >= nr) break;
        for (uint j = 0; j < 8; j++) {
            if (j >= nb) break;
            simdgroup_store(acc[q * 8 + j], sm, p.cout, ulong2(j * 8, rb * 8));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nout = p.cin * p.cout;
    for (uint i = tid; i < nout; i += 128) part[(s * 9 + k) * nout + i] = sm[i];
}


// Partial sums over pixels: in (B, N, C) contiguous -> part (B, S, C), threadgroup (b, s) sums
// the pixels of chunk s of image b. Thread t handles channel t % C and every (256 / C)-th pixel,
// then the threads of each channel are summed through threadgroup memory. p.b = B, p.h = N,
// p.cin = C (<= 256), p.chunks = S.
kernel void pixel_sum(device const float* in [[buffer(0)]], device const float* unused [[buffer(1)]],
                      device float* part [[buffer(2)]], constant P& p [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float acc[256];
    uint b = tg / p.chunks, s = tg % p.chunks;
    uint c = tid % p.cin, lanes = 256 / p.cin, r = tid / p.cin;
    uint per = (p.h + p.chunks - 1) / p.chunks;
    uint start = s * per, end = min(p.h, start + per);
    float v = 0.0f;
    if (r < lanes) {
        device const float* src = in + b * p.h * p.cin + c;
        for (uint i = start + r; i < end; i += lanes) v += src[i * p.cin];
    }
    acc[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.cin) {
        float t = 0.0f;
        for (uint k = 0; k < lanes; k++) t += acc[k * p.cin + tid];
        part[(b * p.chunks + s) * p.cin + tid] = t;
    }
}
"#;

#[repr(C)]
struct Params {
    b: u32,
    h: u32,
    w: u32,
    cin: u32,
    cout: u32,
    d: u32,
    chunks: u32,
    bias: u32,
    residual: u32,
    relu: u32,
}

impl Params {
    fn new(b: usize, h: usize, w: usize, cin: usize, cout: usize, d: usize) -> Params {
        let u = |v: usize| v as u32;
        Params { b: u(b), h: u(h), w: u(w), cin: u(cin), cout: u(cout), d: u(d), chunks: 0, bias: 0, residual: 0, relu: 0 }
    }
}

thread_local! {
    static PIPELINES: std::cell::RefCell<HashMap<&'static str, ComputePipeline>> = std::cell::RefCell::new(HashMap::new());
}

fn pipeline(dev: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
    if let Some(p) = PIPELINES.with(|m| m.borrow().get(name).cloned()) {
        return Ok(p);
    }
    let lib = dev.metal_device().new_library_with_source(SRC, None).map_err(|e| candle_core::Error::Msg(format!("{e:?}")))?;
    for n in ["conv_fwd", "conv_wgrad", "relu_mask", "conv_fwd_sg", "conv_wgrad_sg", "pixel_sum"] {
        let f = lib.get_function(n, None).map_err(|e| candle_core::Error::Msg(format!("{e:?}")))?;
        let p = dev.metal_device().new_compute_pipeline_state_with_function(&f).map_err(|e| candle_core::Error::Msg(format!("{e:?}")))?;
        PIPELINES.with(|m| m.borrow_mut().insert(n, p));
    }
    Ok(PIPELINES.with(|m| m.borrow()[name].clone()))
}

/// Runs kernel `name` with `threads` threads (in groups of 256) and an output of `n_out` floats.
/// Inputs go to buffers 0 and 1, the output to 2, the parameters to 3, further inputs to 4...
/// With `groups`, dispatches that many threadgroups of 128 threads instead.
fn launch(name: &'static str, ins: &[(&MetalStorage, &Layout)], p: Params, threads: usize, n_out: usize) -> Result<MetalStorage> {
    launch_with(name, ins, p, threads, None, n_out)
}

fn launch_groups256(name: &'static str, ins: &[(&MetalStorage, &Layout)], p: Params, groups: usize, n_out: usize) -> Result<MetalStorage> {
    let dev = ins[0].0.device();
    let out = dev.new_buffer(n_out, DType::F32, name)?;
    let pl = pipeline(dev, name)?;
    let guard = dev.command_encoder()?;
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
    enc.set_compute_pipeline_state(&pl);
    for (k, (st, l)) in ins.iter().enumerate() {
        if !l.is_contiguous() || st.dtype() != DType::F32 {
            bail!("{name}: inputs must be contiguous f32");
        }
        let idx = if k < 2 { k } else { k + 2 };
        enc.set_input_buffer(idx, Some(st.buffer()), l.start_offset() * 4);
    }
    enc.set_output_buffer(2, Some(&out), 0);
    enc.set_bytes(3, &p);
    enc.dispatch_thread_groups(MTLSize { width: groups, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
    drop(guard);
    Ok(MetalStorage::new(out, dev.clone(), n_out, DType::F32))
}

fn launch_with(name: &'static str, ins: &[(&MetalStorage, &Layout)], p: Params, threads: usize, groups: Option<usize>, n_out: usize) -> Result<MetalStorage> {
    for (st, l) in ins {
        if !l.is_contiguous() || st.dtype() != DType::F32 {
            bail!("{name}: inputs must be contiguous f32");
        }
    }
    let dev = ins[0].0.device();
    let out = dev.new_buffer(n_out, DType::F32, name)?;
    let pl = pipeline(dev, name)?;
    let guard = dev.command_encoder()?;
    let enc: &candle_metal_kernels::metal::ComputeCommandEncoder = guard.as_ref();
    enc.set_compute_pipeline_state(&pl);
    for (k, (st, l)) in ins.iter().enumerate() {
        let idx = if k < 2 { k } else { k + 2 };
        enc.set_input_buffer(idx, Some(st.buffer()), l.start_offset() * 4);
    }
    enc.set_output_buffer(2, Some(&out), 0);
    enc.set_bytes(3, &p);
    match groups {
        Some(g) => enc.dispatch_thread_groups(MTLSize { width: g, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 }),
        None => enc.dispatch_threads(MTLSize { width: threads, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 }),
    }
    drop(guard);
    Ok(MetalStorage::new(out, dev.clone(), n_out, DType::F32))
}

/// Use the matrix-unit kernels (`*_sg`) when the shapes allow; MATRIX_UNITS=0 forces the scalar ones.
pub static MATRIX_UNITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

fn use_sg(cin: usize, cout: usize) -> bool {
    // a Cin that is not a multiple of 8 is staged with its weights, in room for 16 padded channels
    MATRIX_UNITS.load(std::sync::atomic::Ordering::Relaxed) && matches!(cout, 8 | 16 | 32) && cin <= 32 && (cin % 8 == 0 || cin <= 16)
}

/// Cin padded to a power of two >= 8 (the matrix kernels' staging needs 128 % cinp == 0).
fn cin_padded(cin: usize) -> u32 {
    cin.next_power_of_two().max(8) as u32
}

fn conv_launch(ins: &[(&MetalStorage, &Layout)], p: Params, npix: usize, cin: usize, cout: usize) -> Result<MetalStorage> {
    if use_sg(cin, cout) {
        let mut p = p;
        p.chunks = cin_padded(cin);
        launch_with("conv_fwd_sg", ins, p, 0, Some(npix.div_ceil(32)), npix * cout)
    } else {
        launch("conv_fwd", ins, p, npix * cout, npix * cout)
    }
}

fn conv_shapes(l1: &Layout, l2: &Layout) -> Result<(usize, usize, usize, usize, usize)> {
    let (b, h, w, cin) = l1.shape().dims4()?;
    let (k, cout) = l2.shape().dims2()?;
    if k != 9 * cin {
        bail!("wrap_conv: weight is ({k}, {cout}), input has {cin} channels");
    }
    Ok((b, h, w, cin, cout))
}

/// Weights for the input gradient: taps mirrored (k -> 8 - k), each tap transposed.
fn flipped(w: &Tensor) -> Result<Tensor> {
    let (k, cout) = w.dims2()?;
    let cin = k / 9;
    let flip = Tensor::from_vec((0..9u32).rev().collect::<Vec<_>>(), 9, w.device())?;
    w.reshape((9, cin, cout))?.index_select(&flip, 0)?.transpose(1, 2)?.contiguous()?.reshape((9 * cout, cin))
}

/// y = conv(x, w); x (B,H,W,Cin), w (9*Cin, Cout) -> (B,H,W,Cout). With `residual`, y += x
/// (Cin == Cout). No bias, no relu: the plain op, with its own backward.
pub struct WrapConv {
    pub d: usize,
    pub residual: bool,
}

impl CustomOp2 for WrapConv {
    fn name(&self) -> &'static str {
        "wrap_conv"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("wrap_conv has only a Metal implementation")
    }

    fn metal_fwd(&self, s1: &MetalStorage, l1: &Layout, s2: &MetalStorage, l2: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, h, w, cin, cout) = conv_shapes(l1, l2)?;
        let mut p = Params::new(b, h, w, cin, cout, self.d);
        p.residual = self.residual as u32;
        // buffer 4 (bias) is unused here, bind the input there
        let st = conv_launch(&[(s1, l1), (s2, l2), (s1, l1)], p, b * h * w, cin, cout)?;
        Ok((st, Shape::from((b, h, w, cout))))
    }

    fn bwd(&self, x: &Tensor, w: &Tensor, _res: &Tensor, gy: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let gy = gy.contiguous()?;
        let gx = gy.apply_op2_no_bwd(&flipped(w)?, &WrapConv { d: self.d, residual: self.residual })?;
        let gw = wgrad(x, &gy, self.d)?;
        Ok((Some(gx), Some(gw)))
    }
}

/// One whole layer in a single kernel: y = relu(conv(x, w) + gb[image] [+ x]).
/// gb (B, Cout) is a per-image bias (the layer bias plus the block's global term), so candle's
/// autograd carries its gradient on to the bias, G and the map mean. Backward: one masking pass,
/// one transposed conv with the residual folded in, the weight gradient and a sum for gb.
pub struct FusedConv {
    pub d: usize,
    pub residual: bool,
    /// false for the first layer: its input is data, no gradient needed
    pub input_grad: bool,
    /// Some(G) when gb = mean_xy(x) . G + b was built from a *detached* mean of x: the backward then
    /// adds that path's input gradient, dgb . G^T / (H*W) per image, inside the transposed conv
    /// (as its per-image bias) instead of a full-size broadcast and add in candle's graph.
    pub global: Option<Tensor>,
}

impl CustomOp3 for FusedConv {
    fn name(&self) -> &'static str {
        "fused_conv"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("fused_conv has only a Metal implementation")
    }

    fn metal_fwd(&self, s1: &MetalStorage, l1: &Layout, s2: &MetalStorage, l2: &Layout, s3: &MetalStorage, l3: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, h, w, cin, cout) = conv_shapes(l1, l2)?;
        if l3.shape().dims() != [b, cout] {
            bail!("fused_conv: bias must be (B, Cout)");
        }
        if self.residual && cin != cout {
            bail!("fused_conv: residual needs Cin == Cout");
        }
        let mut p = Params::new(b, h, w, cin, cout, self.d);
        p.bias = 1;
        p.residual = self.residual as u32;
        p.relu = 1;
        let st = conv_launch(&[(s1, l1), (s2, l2), (s3, l3)], p, b * h * w, cin, cout)?;
        Ok((st, Shape::from((b, h, w, cout))))
    }

    fn bwd(&self, x: &Tensor, w: &Tensor, _gb: &Tensor, y: &Tensor, gy: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let g = gy.contiguous()?.apply_op2_no_bwd(y, &ReluMask)?;
        let gb = g.sum((1, 2))?;
        let gx = match (&self.global, self.input_grad) {
            (_, false) => None,
            (None, true) => Some(g.apply_op2_no_bwd(&flipped(w)?, &WrapConv { d: self.d, residual: self.residual })?),
            (Some(gm), true) => {
                let (_, h, wd, _) = x.dims4()?;
                let extra = (gb.matmul(&gm.t()?)? / (h * wd) as f64)?.contiguous()?;
                Some(g.apply_op3_no_bwd(&flipped(w)?, &extra, &ConvBias { d: self.d, residual: self.residual })?)
            }
        };
        let gw = wgrad(x, &g, self.d)?;
        Ok((gx, Some(gw), Some(gb)))
    }
}

/// conv(x, w) + bias[image] [+ x], no relu, no backward (used for input gradients).
struct ConvBias {
    d: usize,
    residual: bool,
}

impl CustomOp3 for ConvBias {
    fn name(&self) -> &'static str {
        "conv_bias"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("conv_bias has only a Metal implementation")
    }

    fn metal_fwd(&self, s1: &MetalStorage, l1: &Layout, s2: &MetalStorage, l2: &Layout, s3: &MetalStorage, l3: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, h, w, cin, cout) = conv_shapes(l1, l2)?;
        if l3.shape().dims() != [b, cout] {
            bail!("conv_bias: bias must be (B, Cout)");
        }
        let mut p = Params::new(b, h, w, cin, cout, self.d);
        p.bias = 1;
        p.residual = self.residual as u32;
        let st = conv_launch(&[(s1, l1), (s2, l2), (s3, l3)], p, b * h * w, cin, cout)?;
        Ok((st, Shape::from((b, h, w, cout))))
    }
}

/// Mean over the middle dimension: (B, N, C) -> (B, C), no backward (callers detach). candle's
/// reduction over a middle dimension ran at about 8 GB/s, slower than half a conv layer.
pub fn pixel_mean(x: &Tensor) -> Result<Tensor> {
    let (_, n, _) = x.dims3()?;
    let chunks = (n / 256).clamp(1, 32);
    x.contiguous()?.apply_op1_no_bwd(&PixelSum { chunks })?.sum(1)? / n as f64
}

struct PixelSum {
    chunks: usize,
}

impl candle_core::CustomOp1 for PixelSum {
    fn name(&self) -> &'static str {
        "pixel_sum"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("pixel_sum has only a Metal implementation")
    }

    fn metal_fwd(&self, st: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, n, c) = l.shape().dims3()?;
        if c > 256 {
            bail!("pixel_sum: at most 256 channels");
        }
        let mut p = Params::new(b, n, 0, c, 0, 0);
        p.chunks = self.chunks as u32;
        let groups = b * self.chunks;
        let out = launch_groups256("pixel_sum", &[(st, l), (st, l)], p, groups, groups * c)?;
        Ok((out, Shape::from((b, self.chunks, c))))
    }
}

/// gy * (y > 0)
pub struct ReluMask;

impl CustomOp2 for ReluMask {
    fn name(&self) -> &'static str {
        "relu_mask"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("relu_mask has only a Metal implementation")
    }

    fn metal_fwd(&self, s1: &MetalStorage, l1: &Layout, s2: &MetalStorage, l2: &Layout) -> Result<(MetalStorage, Shape)> {
        let n = l1.shape().elem_count();
        if l2.shape().elem_count() != n {
            bail!("relu_mask: shapes differ");
        }
        let mut p = Params::new(n, 0, 0, 0, 0, 0);
        p.b = n as u32;
        let st = launch("relu_mask", &[(s1, l1), (s2, l2)], p, n, n)?;
        Ok((st, l1.shape().clone()))
    }
}

/// Weight gradient (9*Cin, Cout) of a conv with input x and output gradient gy.
pub fn wgrad(x: &Tensor, gy: &Tensor, d: usize) -> Result<Tensor> {
    let (cin, cout) = (x.dim(3)?, gy.dim(3)?);
    x.contiguous()?.apply_op2_no_bwd(&gy.contiguous()?, &WrapConvWgrad { d })?.sum(0)?.reshape((9 * cin, cout))
}

/// Partial weight gradients (chunks, 9*Cin*Cout) from x (B,H,W,Cin) and gy (B,H,W,Cout).
struct WrapConvWgrad {
    d: usize,
}

impl CustomOp2 for WrapConvWgrad {
    fn name(&self) -> &'static str {
        "wrap_conv_wgrad"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        bail!("wrap_conv_wgrad has only a Metal implementation")
    }

    fn metal_fwd(&self, s1: &MetalStorage, l1: &Layout, s2: &MetalStorage, l2: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, h, w, cin) = l1.shape().dims4()?;
        let cout = l2.shape().dims4()?.3;
        let npix = b * h * w;
        if cin > 64 || cout > 64 || cin * cout > 4096 {
            bail!("wrap_conv_wgrad: at most 64 channels in and out");
        }
        let chunks = (npix / 512).clamp(1, 256);
        let mut p = Params::new(b, h, w, cin, cout, self.d);
        p.chunks = chunks as u32;
        let n = chunks * 9 * cin * cout;
        let st = if use_sg(cin, cout) {
            let mut p = p;
            p.relu = cin_padded(cin);
            launch_with("conv_wgrad_sg", &[(s1, l1), (s2, l2)], p, 0, Some(chunks * 9), n)?
        } else {
            launch("conv_wgrad", &[(s1, l1), (s2, l2)], p, chunks * 9 * 256, n)?
        };
        Ok((st, Shape::from((chunks, 9 * cin * cout))))
    }
}
