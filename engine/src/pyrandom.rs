//! Python's `random.Random`, bit for bit: MT19937 seeded the way CPython seeds
//! it from an int, plus `random()`, `_randbelow`, `choice` and `shuffle`.

const N: usize = 624;
const M: usize = 397;

pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for an int seed. `key` is abs(seed) as 32-bit
    /// words, least significant first (CPython's `init_by_array` key).
    pub fn from_key(key: &[u32]) -> PyRandom {
        let key: Vec<u32> = if key.is_empty() { vec![0] } else { key.to_vec() };
        let mut r = PyRandom { mt: [0; N], index: N };
        r.init_genrand(19650218);
        let mt = &mut r.mt;
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = (mt[i] ^ prev.wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = mt[i - 1] ^ (mt[i - 1] >> 30);
            mt[i] = (mt[i] ^ prev.wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x8000_0000;
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn genrand_u32(&mut self) -> u32 {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        if self.index >= N {
            let mt = &mut self.mt;
            for kk in 0..N {
                let y = (mt[kk] & UPPER) | (mt[(kk + 1) % N] & LOWER);
                let mag = if y & 1 != 0 { MATRIX_A } else { 0 };
                mt[kk] = mt[(kk + M) % N] ^ (y >> 1) ^ mag;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.random()`: 53-bit float in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `getrandbits(k)` for 0 < k <= 32.
    fn getrandbits(&mut self, k: u32) -> u32 {
        self.genrand_u32() >> (32 - k)
    }

    /// `_randbelow_with_getrandbits(n)` for 0 < n < 2**32.
    pub fn randbelow(&mut self, n: u32) -> u32 {
        let k = 32 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    pub fn choice<T: Copy>(&mut self, seq: &[T]) -> T {
        seq[self.randbelow(seq.len() as u32) as usize]
    }

    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u32 + 1) as usize;
            x.swap(i, j);
        }
    }
}
