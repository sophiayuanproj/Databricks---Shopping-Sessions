mod alloc;
mod parts;
mod pq;
use parts::Parts;

#[global_allocator]
static GA: alloc::Huge = alloc::Huge;

use memchr::memchr;
use memmap2::Mmap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
// partition = top NB_BITS bits of a hash; NB_BITS is chosen once at startup (before any
// worker thread exists) from the input size, then only read.
static mut NB_SHIFT: u32 = 64 - 10;
const NOCAT: u16 = u16::MAX;

#[inline(always)]
fn bucket(h: u64) -> usize {
    (h >> unsafe { NB_SHIFT }) as usize
}

// ---------- string refs (pointer into a leaked mmap + length) ----------
#[inline(always)]
fn mkref(s: &[u8]) -> u64 {
    ((s.as_ptr() as u64) << 16) | (s.len() as u64 & 0xffff)
}
#[inline(always)]
fn getref(r: u64) -> &'static [u8] {
    unsafe { std::slice::from_raw_parts((r >> 16) as *const u8, (r & 0xffff) as usize) }
}

// ---------- hashing ----------
/// Load the 1..7 bytes s[i..] into a u64 (zero-padded). Reads 8 bytes when that stays
/// inside the current page (the slices point into mmapped input), else copies.
#[inline(always)]
fn tail_load(s: &[u8], i: usize) -> u64 {
    let r = s.len() - i;
    let p = unsafe { s.as_ptr().add(i) };
    if (p as usize & 4095) <= 4088 {
        (unsafe { (p as *const u64).read_unaligned() }) & (u64::MAX >> (64 - 8 * r))
    } else {
        let mut buf = [0u8; 8];
        buf[..r].copy_from_slice(&s[i..]);
        u64::from_le_bytes(buf)
    }
}
#[inline(always)]
fn fold(a: u64, b: u64) -> u64 {
    let m = (a as u128).wrapping_mul(b as u128);
    (m as u64) ^ ((m >> 64) as u64)
}
#[inline(always)]
fn hash(s: &[u8]) -> u64 {
    const P0: u64 = 0xa076_1d64_78bd_642f;
    const P1: u64 = 0xe703_7ed1_a0b4_28db;
    const P2: u64 = 0x8ebc_6af0_9c88_c6e3;
    const P3: u64 = 0x5899_65cc_7537_4cc3;
    let n = s.len();
    let mut h = (n as u64) ^ P0;
    let mut i = 0;
    while i + 16 <= n {
        let a = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        let b = u64::from_le_bytes(s[i + 8..i + 16].try_into().unwrap());
        h = fold(a ^ P1, b ^ h);
        i += 16;
    }
    if i + 8 <= n {
        let a = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        h = fold(a ^ P2, h ^ P3);
        i += 8;
    }
    if i < n {
        h = fold(tail_load(s, i) ^ P3, h ^ P1);
    }
    fold(h ^ P0, P2 ^ (n as u64)) | 1
}

/// Independent second hash (different constants and state), used to extend event_id identity to 96 bits.
#[inline(always)]
fn hash2(s: &[u8]) -> u32 {
    const Q0: u64 = 0x2d35_8dcc_aa6c_78a5;
    const Q1: u64 = 0x8bb8_4b93_962e_acc9;
    const Q2: u64 = 0x4b33_a62e_d433_d4a3;
    let n = s.len();
    let mut h = (n as u64).wrapping_mul(Q1) ^ Q2;
    let mut i = 0;
    while i + 8 <= n {
        let a = u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
        h = fold(a ^ Q0, h ^ Q1);
        i += 8;
    }
    if i < n {
        h = fold(tail_load(s, i) ^ Q2, h ^ Q0);
    }
    let h = fold(h ^ Q1, Q0 ^ (n as u64));
    (h ^ (h >> 32)) as u32
}

// ---------- open-addressing u64 -> u32 map (key 0 = empty) ----------
struct U64Map {
    mask: usize,
    tab: Vec<(u64, u32)>,
    len: usize,
}
impl U64Map {
    fn new(n: usize) -> Self {
        let cap = (n * 2).next_power_of_two().max(16);
        U64Map { mask: cap - 1, tab: vec![(0, 0); cap], len: 0 }
    }
    fn reset(&mut self, n: usize) {
        let cap = (n * 2).next_power_of_two().max(16);
        if cap != self.tab.len() {
            self.tab = vec![(0, 0); cap];
            self.mask = cap - 1;
        } else {
            self.tab.fill((0, 0));
        }
        self.len = 0;
    }
    #[inline(always)]
    fn get(&self, k: u64) -> Option<u32> {
        let mut s = (k as usize) & self.mask;
        loop {
            let (kk, v) = unsafe { *self.tab.get_unchecked(s) };
            if kk == k {
                return Some(v);
            }
            if kk == 0 {
                return None;
            }
            s = (s + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn get_or_insert(&mut self, k: u64, v: u32) -> (u32, bool) {
        if (self.len + 1) * 2 > self.tab.len() {
            self.grow();
        }
        let mut s = (k as usize) & self.mask;
        loop {
            let (kk, vv) = unsafe { *self.tab.get_unchecked(s) };
            if kk == k {
                return (vv, false);
            }
            if kk == 0 {
                unsafe { *self.tab.get_unchecked_mut(s) = (k, v) };
                self.len += 1;
                return (v, true);
            }
            s = (s + 1) & self.mask;
        }
    }
    #[cold]
    fn grow(&mut self) {
        let old = std::mem::take(&mut self.tab);
        let cap = old.len() * 2;
        self.mask = cap - 1;
        self.tab = vec![(0, 0); cap];
        for (k, v) in old {
            if k != 0 {
                let mut s = (k as usize) & self.mask;
                while self.tab[s].0 != 0 {
                    s = (s + 1) & self.mask;
                }
                self.tab[s] = (k, v);
            }
        }
    }
}

// exact string map (small tables: products, campaigns); keys up to 16 bytes are stored inline
#[derive(Clone, Copy)]
struct SSlot {
    h: u64,
    k0: u64,
    k1: u64,
    len: u32,
    v: u32,
    r: u64,
}
struct StrMap {
    mask: usize,
    slots: Vec<SSlot>,
}
#[inline(always)]
fn load16(s: &[u8]) -> (u64, u64) {
    let mut b = [0u8; 16];
    let n = s.len().min(16);
    b[..n].copy_from_slice(&s[..n]);
    (u64::from_le_bytes(b[..8].try_into().unwrap()), u64::from_le_bytes(b[8..].try_into().unwrap()))
}
impl StrMap {
    fn new(items: &[(&'static [u8], u32)]) -> Self {
        let cap = (items.len() * 4).next_power_of_two().max(16);
        let mut slots = vec![SSlot { h: 0, k0: 0, k1: 0, len: 0, v: 0, r: 0 }; cap];
        for &(s, v) in items {
            let h = hash(s);
            let mut i = (h as usize) & (cap - 1);
            loop {
                if slots[i].h == 0 {
                    let (k0, k1) = load16(s);
                    slots[i] = SSlot { h, k0, k1, len: s.len() as u32, v, r: mkref(s) };
                    break;
                }
                if slots[i].h == h && getref(slots[i].r) == s {
                    break; // keep first
                }
                i = (i + 1) & (cap - 1);
            }
        }
        StrMap { mask: cap - 1, slots }
    }
    #[inline(always)]
    fn get(&self, s: &[u8]) -> Option<u32> {
        let h = hash(s);
        let mut i = (h as usize) & self.mask;
        loop {
            let e = unsafe { self.slots.get_unchecked(i) };
            if e.h == 0 {
                return None;
            }
            if e.h == h && e.len as usize == s.len() {
                if s.len() <= 16 {
                    let (k0, k1) = load16(s);
                    if k0 == e.k0 && k1 == e.k1 {
                        return Some(e.v);
                    }
                } else if eq_bytes(getref(e.r), s) {
                    return Some(e.v);
                }
            }
            i = (i + 1) & self.mask;
        }
    }
}

// ---------- CSV ----------
#[inline(always)]
fn split<const N: usize>(line: &'static [u8], out: &mut [&'static [u8]; N]) -> bool {
    let len = line.len();
    let mut i = 0usize;
    let mut f = 0usize;
    loop {
        if f == N {
            return false;
        }
        let content;
        let more;
        if i < len && line[i] == b'"' {
            let st = i + 1;
            let mut j = st;
            while j < len {
                let c = line[j];
                if c == b'"' {
                    break;
                }
                if c == b',' {
                    return false;
                }
                j += 1;
            }
            if j >= len {
                return false;
            }
            content = &line[st..j];
            i = j + 1;
            if i < len {
                if line[i] != b',' {
                    return false;
                }
                i += 1;
                more = true;
            } else {
                more = false;
            }
        } else {
            let st = i;
            while i < len {
                let c = line[i];
                if c == b',' {
                    break;
                }
                if c == b'"' {
                    return false;
                }
                i += 1;
            }
            content = &line[st..i];
            if i < len {
                i += 1;
                more = true;
            } else {
                more = false;
            }
        }
        out[f] = content;
        f += 1;
        if !more {
            break;
        }
    }
    f == N
}

#[inline(always)]
fn for_lines(c: &'static [u8], mut f: impl FnMut(&'static [u8])) {
    let mut p = 0;
    let n = c.len();
    while p < n {
        let e = match memchr(b'\n', &c[p..]) {
            Some(x) => p + x,
            None => n,
        };
        f(&c[p..e]);
        p = e + 1;
    }
}

// SIMD line scanner: calls f(fields) for every well-formed line with exactly N fields.
#[inline(always)]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    let a = _mm256_loadu_si256(p as *const __m256i);
    let b = _mm256_loadu_si256(p.add(32) as *const __m256i);
    let nl = _mm256_set1_epi8(b'\n' as i8);
    let cm = _mm256_set1_epi8(b',' as i8);
    let qt = _mm256_set1_epi8(b'"' as i8);
    let m = |x: __m256i, y: __m256i| -> u64 {
        let lo = _mm256_movemask_epi8(_mm256_cmpeq_epi8(x, y)) as u32 as u64;
        lo
    };
    let n = m(a, nl) | (m(b, nl) << 32);
    let c = m(a, cm) | (m(b, cm) << 32);
    let q = m(a, qt) | (m(b, qt) << 32);
    (n, c, q)
}

#[inline(always)]
fn scan<const N: usize>(c: &'static [u8], mut f: impl FnMut(&[&'static [u8]; N])) {
    // Iterate comma/newline positions only. A field is quoted iff its first byte is '"';
    // it must then end with '"' and be >= 2 long. Stray quotes are caught by comparing
    // the line's quote count (popcount) with 2 * the number of quoted fields.
    let n = c.len();
    let mut fields: [&'static [u8]; N] = [&[]; N];
    let mut nf = 0usize;
    let mut fs = 0usize;
    let mut ls = 0usize;
    let mut bad = false;
    let mut qexp = 0u32; // quotes accounted for by quoted fields
    let mut qprev = 0u32; // quotes of this line in earlier blocks
    let mut lsb = 0u32; // bit where the current line starts in this block
    let mut base = 0usize;
    let mut tail = [0u8; 64];
    let p = c.as_ptr();
    #[inline(always)]
    fn range_mask(lo: u32, hi: u32) -> u64 {
        // bits lo..hi (hi exclusive, lo, hi <= 64)
        if lo >= 64 {
            return 0;
        }
        let upto = if hi >= 64 { u64::MAX } else { (1u64 << hi) - 1 };
        upto & !((1u64 << lo) - 1)
    }
    while base < n {
        let (nl, cm, qt) = if base + 64 <= n {
            unsafe { masks64(p.add(base)) }
        } else {
            let r = n - base;
            tail[..r].copy_from_slice(&c[base..]);
            tail[r..].fill(0);
            unsafe { masks64(tail.as_ptr()) }
        };
        let mut bits = nl | cm;
        while bits != 0 {
            let t = bits.trailing_zeros();
            bits &= bits - 1;
            let pos = base + t as usize;
            if nf < N {
                // branchless: quoted iff the field is non-empty and starts with '"'
                let s = fs;
                let e = pos;
                let len = e - s;
                let first = unsafe { *p.add(s) }; // s <= e < n, so readable (the separator if empty)
                let last = unsafe { *p.add(if len > 0 { e - 1 } else { s }) };
                let quoted = (len > 0) & (first == b'"');
                let okq = !quoted | ((len >= 2) & (last == b'"'));
                bad |= !okq;
                let q = quoted as usize;
                qexp += 2 * q as u32;
                let flen = if okq { len - 2 * q } else { 0 };
                fields[nf] = unsafe { std::slice::from_raw_parts(p.add(s + q), flen) };
            }
            nf += 1;
            fs = pos + 1;
            if (nl >> t) & 1 != 0 {
                let q = qprev + (qt & range_mask(lsb, t)).count_ones();
                if !bad && nf == N && q == qexp {
                    f(&fields);
                }
                nf = 0;
                bad = false;
                qexp = 0;
                qprev = 0;
                lsb = t + 1;
                ls = pos + 1;
            }
        }
        // quotes of the line still open at the end of this block
        qprev += (qt & range_mask(lsb, 64)).count_ones();
        lsb = 0;
        base += 64;
    }
    if ls < n {
        // last line without a trailing newline
        if nf < N {
            let s = fs;
            let e = n;
            let quoted = e > s && c[s] == b'"';
            if quoted {
                if e < s + 2 || c[e - 1] != b'"' {
                    bad = true;
                }
                qexp += 2;
                fields[nf] = &c[s + 1..(e - 1).max(s + 1)];
            } else {
                fields[nf] = &c[s..e];
            }
        }
        nf += 1;
        if !bad && nf == N && qprev == qexp {
            f(&fields);
        }
    }
}

// ---------- timestamps ----------
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

#[inline(always)]
fn d2(s: &[u8], i: usize) -> Option<i64> {
    let a = s[i].wrapping_sub(b'0');
    let b = s[i + 1].wrapping_sub(b'0');
    if a > 9 || b > 9 {
        None
    } else {
        Some((a * 10 + b) as i64)
    }
}

#[inline(always)]
fn rd64(s: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(s[i..i + 8].try_into().unwrap())
}

#[inline(always)]
fn digits_ok(w: u64, dmask: u64) -> bool {
    const HI: u64 = 0xF0F0_F0F0_F0F0_F0F0;
    const D30: u64 = 0x3030_3030_3030_3030;
    (w & HI & dmask) == (D30 & dmask) && (w.wrapping_add(0x0606_0606_0606_0606) & HI & dmask) == (D30 & dmask)
}

#[inline(always)]
fn parse_ts(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 13 {
        parse_f3_13(s)
    } else if (n == 20 || n == 25) && s[4] == b'-' {
        parse_f12(s)
    } else {
        parse_f3(s)
    }
}

/// 8 ASCII digits (first digit in the lowest byte) -> value; caller validated digits.
#[inline(always)]
fn swar8(w: u64) -> u64 {
    let v = w.wrapping_sub(0x3030_3030_3030_3030);
    let v = v.wrapping_mul(10).wrapping_add(v >> 8);
    let a = (v & 0x0000_00FF_0000_00FF).wrapping_mul(100 + (1_000_000u64 << 32));
    let b = ((v >> 16) & 0x0000_00FF_0000_00FF).wrapping_mul(1 + (10_000u64 << 32));
    (a.wrapping_add(b)) >> 32
}

/// F3 with exactly 13 characters (the usual epoch-millisecond width).
#[inline(always)]
fn parse_f3_13(s: &[u8]) -> Option<i64> {
    let w0 = rd64(s, 0);
    let w1 = rd64(s, 5);
    if !digits_ok(w0, u64::MAX) || !digits_ok(w1, u64::MAX) {
        return None;
    }
    Some((swar8(w0) * 100_000 + swar8(w1) % 100_000) as i64)
}

#[inline(always)]
fn parse_f3(s: &[u8]) -> Option<i64> {
    if s.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    let mut sig = 0usize;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        if sig > 0 || d != 0 {
            sig += 1;
            if sig > 19 {
                return None; // too long for i64: outside any window
            }
            v = v * 10 + d as u64;
        }
    }
    if v > i64::MAX as u64 {
        return None;
    }
    Some(v as i64)
}

#[inline(always)]
fn parse_f12(s: &[u8]) -> Option<i64> {
    // s[0..8] = YYYY-MM-, s[8..16] = DDTHH:MM, s[12..20] = H:MM:SSx
    let w0 = rd64(s, 0);
    let w1 = rd64(s, 8);
    let w2 = rd64(s, 12);
    // digit masks (little endian: byte k at bits 8k)
    const M0: u64 = 0x00FF_FF00_FFFF_FFFF; // bytes 0-3,5,6
    const M1: u64 = 0xFFFF_00FF_FF00_FFFF; // bytes 8,9,11,12,14,15
    const M2: u64 = 0x00FF_FF00_0000_0000; // bytes 17,18 (offsets 5,6 of w2)
    if !digits_ok(w0, M0) || !digits_ok(w1, M1) || !digits_ok(w2, M2) {
        return None;
    }
    // separators: s[7]='-', s[10]='T', s[13]=':', s[16]=':'  (s[4] already checked)
    if s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':' {
        return None;
    }
    let x0 = w0 ^ 0x3030_3030_3030_3030;
    let x1 = w1 ^ 0x3030_3030_3030_3030;
    let x2 = w2 ^ 0x3030_3030_3030_3030;
    let b = |x: u64, k: u32| ((x >> (8 * k)) & 0xff) as i64;
    let y = b(x0, 0) * 1000 + b(x0, 1) * 100 + b(x0, 2) * 10 + b(x0, 3);
    let mo = b(x0, 5) * 10 + b(x0, 6);
    let d = b(x1, 0) * 10 + b(x1, 1);
    let h = b(x1, 3) * 10 + b(x1, 4);
    let mi = b(x1, 6) * 10 + b(x1, 7);
    let se = b(x2, 5) * 10 + b(x2, 6);
    if !(1..=12).contains(&mo) || d < 1 || h > 23 || mi > 59 || se > 59 {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let dim = match mo {
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => 31,
    };
    if d > dim {
        return None;
    }
    let mut off = 0i64;
    if s.len() == 20 {
        if s[19] != b'Z' {
            return None;
        }
    } else {
        let sign = match s[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if s[22] != b':' {
            return None;
        }
        let oh = d2(s, 20)?;
        let om = d2(s, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        off = sign * (oh * 60 + om) * 60_000;
    }
    let yi = (y - 1970) as usize;
    let days = if yi < 200 { MONTH_START[yi * 12 + mo as usize - 1] as i64 + d - 1 } else { days_from_civil(y, mo, d) };
    // independent products instead of a serial multiply chain
    Some(days * 86_400_000 + h * 3_600_000 + mi * 60_000 + se * 1000 - off)
}

// days since epoch of the first of each month, 1970-01 .. 2169-12
static MONTH_START: [i32; 2400] = {
    let mut t = [0i32; 2400];
    let mut i = 0;
    while i < 2400 {
        let y = 1970 + (i / 12) as i64;
        let m = (i % 12) as i64 + 1;
        // days_from_civil(y, m, 1), const-evaluable
        let yy = if m <= 2 { y - 1 } else { y };
        let era = yy / 400;
        let yoe = yy - era * 400;
        let mp = (m + 9) % 12;
        let doy = (153 * mp + 2) / 5;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        t[i] = (era * 146097 + doe - 719468) as i32;
        i += 1;
    }
    t
};

#[inline(always)]
fn eq_bytes(a: &[u8], b: &[u8]) -> bool {
    let n = a.len();
    if n != b.len() {
        return false;
    }
    let mut i = 0;
    while i + 8 <= n {
        if rd64(a, i) != rd64(b, i) {
            return false;
        }
        i += 8;
    }
    while i < n {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[inline(always)]
fn parse_price(s: &[u8]) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        v = v * 10 + d as u64;
        if v > 10_000_000 {
            // keep scanning digits? any further digit only grows it (or non-digit -> bad): bad either way
            return None;
        }
    }
    if v == 0 {
        None
    } else {
        Some(v as u32)
    }
}

// ---------- records ----------
// All times are stored as u32 milliseconds relative to cws (= window_start - 7 days).
#[derive(Clone, Copy)]
#[repr(C)]
struct Ev1 {
    eh: u64, // event_id hash (identity together with eh2)
    uh: u64, // 0 = invalid event
    uref: u64,
    eh2: u32,
    ingest: i32, // seconds relative to cws
    ts: u32,
    pc: u32, // price | cat << 24, cat 255 = not a purchase
}

static DUMMY: Ev1 = Ev1 { eh: 0, uh: 0, uref: 0, eh2: 0, ingest: 0, ts: 0, pc: 0 };

#[derive(Clone, Copy)]
#[repr(C)]
struct Ev2 {
    key: u64,
    nref: u64,
    ts: u32,
    pc: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Ck {
    key: u64,
    cref: u64,
    ts: u32,
    camp: u32,
}

struct URow {
    nref: u64,
    revenue: i64,
    day: i32,
    sessions: u32,
    events: u32,
    top: u16,
}

// ---------- io helpers ----------
fn list_csv(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().map(|x| x == "csv").unwrap_or(false)
                    && !p.file_name().map(|n| n.to_string_lossy().starts_with('.')).unwrap_or(true)
                    && p.is_file()
            })
            .collect(),
        Err(_) => vec![],
    };
    v.sort();
    v
}

fn map_file(p: &Path) -> &'static [u8] {
    let f = std::fs::File::open(p).expect("open");
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len == 0 {
        return &[];
    }
    let m = unsafe { memmap2::MmapOptions::new().populate().map(&f).expect("mmap") };
    let m: &'static Mmap = Box::leak(Box::new(m));
    &m[..]
}

fn chunks_of(files: &[&'static [u8]], target: usize) -> Vec<&'static [u8]> {
    let mut out = Vec::new();
    for &f in files {
        let n = f.len();
        let mut p = match memchr(b'\n', f) {
            Some(x) => x + 1,
            None => n,
        };
        while p < n {
            let mut e = (p + target).min(n);
            if e < n {
                e = match memchr(b'\n', &f[e..]) {
                    Some(x) => e + x + 1,
                    None => n,
                };
            }
            out.push(&f[p..e]);
            p = e;
        }
    }
    // biggest first for better balance
    out.sort_by_key(|c| std::cmp::Reverse(c.len()));
    out
}

fn par<T: Send>(nt: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt).map(|t| { let f = &f; sc.spawn(move || f(t)) }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn small_table(p: &Path) -> Vec<(&'static [u8], &'static [u8])> {
    let data = if p.exists() { map_file(p) } else { &[] };
    let mut v = Vec::new();
    let body = match memchr(b'\n', data) {
        Some(x) => &data[x + 1..],
        None => &data[data.len()..],
    };
    for_lines(body, |l| {
        let mut f: [&'static [u8]; 2] = [&[]; 2];
        if split(l, &mut f) && !f[0].is_empty() {
            v.push((f[0], f[1]));
        }
    });
    v
}

fn manifest_ts(text: &str, key: &str) -> i64 {
    let i = text.find(key).expect("manifest key") + key.len();
    let rest = &text[i..];
    let rest = &rest[rest.find(':').unwrap() + 1..];
    let a = rest.find('"').unwrap() + 1;
    let b = a + rest[a..].find('"').unwrap();
    parse_ts(rest[a..b].as_bytes()).expect("manifest ts")
}

fn main() {
    let t0 = Instant::now();
    let data_dir = PathBuf::from(std::env::var("DATA_DIR").unwrap_or("/data".into()));
    let out_dir = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or("/output".into()));
    let nt: usize = std::env::var("NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|x| x.get()).unwrap_or(8))
        .max(1);
    let lg = |m: &str| {
        let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
        unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
        eprintln!("[{:7.3}s] {} (minflt {} utime {:.2} stime {:.2})", t0.elapsed().as_secs_f64(), m, ru.ru_minflt,
            ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 * 1e-6, ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 * 1e-6);
    };

    let man = std::fs::read_to_string(data_dir.join("manifest.json")).expect("manifest");
    let ws = manifest_ts(&man, "\"window_start\"");
    let we = manifest_ts(&man, "\"window_end\"");
    let cws = ws - WEEK;
    let day0 = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - day0 + 1).max(1) as usize;
    assert!(we - cws < (1i64 << 32) && we > ws, "window too long for 32-bit relative timestamps");

    // products & campaigns
    let prows = small_table(&data_dir.join("products.csv"));
    let mut catnames: Vec<&'static [u8]> = Vec::new();
    let mut prod_items = Vec::new();
    {
        let mut cm: std::collections::HashMap<&[u8], u32> = Default::default();
        for &(p, c) in &prows {
            let id = *cm.entry(c).or_insert_with(|| {
                catnames.push(c);
                (catnames.len() - 1) as u32
            });
            prod_items.push((p, id));
        }
    }
    let unknown_cat = catnames.len() as u16;
    catnames.push(b"unknown");
    let ncat = catnames.len();
    // rank of categories by bytes, for tie-breaking
    let mut order: Vec<usize> = (0..ncat).collect();
    order.sort_by(|&a, &b| catnames[a].cmp(catnames[b]));
    let mut cat_rank = vec![0u32; ncat];
    for (r, &c) in order.iter().enumerate() {
        cat_rank[c] = r as u32;
    }
    let products = StrMap::new(&prod_items);

    let crows = small_table(&data_dir.join("campaigns.csv"));
    let camp_items: Vec<(&'static [u8], u32)> =
        crows.iter().enumerate().map(|(i, &(c, _))| (c, i as u32)).collect();
    let campaigns = StrMap::new(&camp_items);
    let ncamp = crows.len();
    let organic = ncamp as u32;

    // map inputs
    let map_dir = |name: &str| -> Vec<&'static [u8]> {
        let files = list_csv(&data_dir.join(name));
        let next = AtomicUsize::new(0);
        let parts: Vec<Vec<(usize, &'static [u8])>> = par(nt, |_| {
            let mut v = Vec::new();
            loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= files.len() {
                    break;
                }
                v.push((i, map_file(&files[i])));
            }
            v
        });
        let mut res: Vec<&'static [u8]> = vec![&[]; files.len()];
        for (i, m) in parts.into_iter().flatten() {
            res[i] = m;
        }
        res
    };
    let link_files = map_dir("links");
    let ev_files = map_dir("events");
    let ck_files = map_dir("clicks");
    lg("mapped");
    {
        let ev_bytes: usize = ev_files.iter().map(|f| f.len()).sum();
        let est_events = ev_bytes / 80 + 1;
        let target = (est_events / 24_000).max(1).next_power_of_two();
        let bits = target.trailing_zeros().clamp(8, 12);
        unsafe { NB_SHIFT = 64 - bits };
    }
    #[allow(non_snake_case)]
    let NB: usize = 1 << (64 - unsafe { NB_SHIFT });

    // ---------------- events parse ----------------
    assert!(ncat < 255, "too many categories");
    let parse_chunk = |out: &mut Parts<Ev1>, chunk: &'static [u8]| {
        scan::<7>(chunk, |f| {
                let eid = f[0];
                if eid.is_empty() || eid.len() > 0xffff {
                    return;
                }
                let eh = hash(eid);
                let ingest_ms = if f[3].len() == 20 {
                    parse_f12(f[3]).unwrap_or(i64::MAX)
                } else {
                    parse_ts(f[3]).unwrap_or(i64::MAX)
                };
                let ingest = ((ingest_ms.saturating_sub(cws)).div_euclid(1000)).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
                let mut e = Ev1 { eh, uh: 0, uref: 0, eh2: hash2(eid), ingest, ts: 0, pc: 0 };
                'v: {
                    let user = f[1];
                    if user.is_empty() || user.len() > 0xffff || f[5].is_empty() {
                        break 'v;
                    }
                    let ts = match parse_ts(f[2]) {
                        Some(t) => t,
                        None => break 'v,
                    };
                    if ts < ws || ts >= we {
                        break 'v;
                    }
                    let et = f[4];
                    let purchase = if et.len() == 4 {
                        let w = u32::from_le_bytes(et[..4].try_into().unwrap());
                        if w == u32::from_le_bytes(*b"view") || w == u32::from_le_bytes(*b"cart") {
                            false
                        } else {
                            break 'v;
                        }
                    } else if et.len() == 8 && rd64(et, 0) == u64::from_le_bytes(*b"purchase") {
                        true
                    } else {
                        break 'v;
                    };
                    if ingest_ms.saturating_sub(ts) > DAY {
                        break 'v;
                    }
                    if purchase {
                        let p = match parse_price(f[6]) {
                            Some(p) => p,
                            None => break 'v,
                        };
                        let cat = products.get(f[5]).map(|c| c as u32).unwrap_or(unknown_cat as u32);
                        e.pc = p | (cat << 24);
                    } else {
                        e.pc = 255 << 24;
                    }
                    e.ts = (ts - cws) as u32;
                    e.uh = hash(user);
                    e.uref = mkref(user);
                }
                out.push(bucket(eh), e);
            });
    };
    #[inline(always)]
    fn lprefetch(ltab: &[[u64; 3]], lmask: usize, uh: u64) {
        unsafe {
            std::arch::x86_64::_mm_prefetch(ltab.as_ptr().add((uh as usize) & lmask) as *const i8, std::arch::x86_64::_MM_HINT_T0)
        };
    }
    #[inline(always)]
    fn lresolve(ltab: &[[u64; 3]], lmask: usize, uh: u64, uref: u64) -> (u64, u64) {
        let mut s = (uh as usize) & lmask;
        loop {
            let e = unsafe { ltab.get_unchecked(s) };
            if e[0] == uh {
                return (e[1], e[2]);
            }
            if e[0] == 0 {
                return (uh, uref);
            }
            s = (s + 1) & lmask;
        }
    }

    // ---------------- parse events, then clicks, in one pass over a shared queue ----------------
    let echunks = chunks_of(&ev_files, 2 << 20);
    let cchunks = chunks_of(&ck_files, 2 << 20);
    let next = AtomicUsize::new(0);
    let cnext = AtomicUsize::new(0);
    let link_slot: std::sync::OnceLock<(Vec<[u64; 3]>, usize)> = std::sync::OnceLock::new();
    let parsed: Vec<(Parts<Ev1>, Parts<Ck>)> = par(nt, |tid| {
        if tid == 0 {
            // linked IDs are built on one thread while the others parse events
            let _ = link_slot.set(build_links(&link_files));
        }
        let mut out: Parts<Ev1> = Parts::new(NB, 16 << 10);
        loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= echunks.len() {
                break;
            }
            parse_chunk(&mut out, echunks[i]);
        }
        // clicks need the linked-ID table
        let (ltab, lmask) = loop {
            if let Some(x) = link_slot.get() {
                break (&x.0[..], x.1);
            }
            std::thread::yield_now();
        };
        let mut cout: Parts<Ck> = Parts::new(NB, 8 << 10);
        const BATCH: usize = 32;
        let mut batch = [Ck { key: 0, cref: 0, ts: 0, camp: 0 }; BATCH];
        let mut nb = 0usize;
        loop {
            let i = cnext.fetch_add(1, Relaxed);
            if i >= cchunks.len() {
                break;
            }
            scan::<4>(cchunks[i], |f| {
                let (cid, user) = (f[0], f[1]);
                if cid.is_empty() || user.is_empty() || cid.len() > 0xffff || user.len() > 0xffff {
                    return;
                }
                let ts = match parse_ts(f[3]) {
                    Some(t) => t,
                    None => return,
                };
                if ts < cws || ts >= we {
                    return;
                }
                let camp = match campaigns.get(f[2]) {
                    Some(c) => c,
                    None => return,
                };
                let uh = hash(user);
                lprefetch(ltab, lmask, uh);
                batch[nb] = Ck { key: uh, cref: mkref(cid), ts: (ts - cws) as u32, camp };
                nb += 1;
                if nb == BATCH {
                    for c in &batch[..nb] {
                        let (key, _) = lresolve(ltab, lmask, c.key, 0);
                        cout.push(bucket(key), Ck { key, ..*c });
                    }
                    nb = 0;
                }
            });
        }
        for c in &batch[..nb] {
            let (key, _) = lresolve(ltab, lmask, c.key, 0);
            cout.push(bucket(key), Ck { key, ..*c });
        }
        out.finish();
        cout.finish();
        (out, cout)
    });
    lg("events+clicks parsed");
    let (ev1, cks): (Vec<Parts<Ev1>>, Vec<Parts<Ck>>) = parsed.into_iter().unzip();
    let (ltab, lmask) = link_slot.into_inner().unwrap();
    let ltab = &ltab[..];
    let resolve = |uh: u64, uref: u64| -> (u64, u64) { lresolve(ltab, lmask, uh, uref) };

    // ---------------- dedup + resolve ----------------
    let next = AtomicUsize::new(0);
    let ev2: Vec<Parts<Ev2>> = par(nt, |_| {
        // Ev2 blocks reuse the memory of consumed Ev1 blocks (same byte size or less)
        let ev1_blk_bytes = ev1[0].block_bytes();
        let blk2 = (ev1_blk_bytes / std::mem::size_of::<Ev2>()) / 8 * 8;
        let mut out: Parts<Ev2> = Parts::with_blk(NB, blk2);
        let mut tab: Vec<u64> = Vec::new();
        let mut loser: Vec<u64> = Vec::new();
        const BATCH: usize = 32;
        let mut batch: [&Ev1; BATCH] = [&DUMMY; BATCH];
        let mut nbat = 0usize;
        let mut tbase = vec![0usize; ev1.len()];
        loop {
            let b = next.fetch_add(1, Relaxed);
            if b >= NB {
                break;
            }
            let mut n = 0usize;
            for (t, tb) in ev1.iter().enumerate() {
                tbase[t] = n;
                n += tb.len(b);
            }
            if n == 0 {
                continue;
            }
            let cap = (n * 2).next_power_of_two().max(16);
            let mask = cap - 1;
            tab.clear();
            tab.resize(cap, 0);
            loser.clear();
            loser.resize(n / 64 + 1, 0);
            for (t, tb) in ev1.iter().enumerate() {
                let mut i = 0u32;
                for sl in tb.slices(b) {
                    for (j, e) in sl.iter().enumerate() {
                        if j + 8 < sl.len() {
                            let s2 = (sl[j + 8].eh as usize) & mask;
                            unsafe { std::arch::x86_64::_mm_prefetch(tab.as_ptr().add(s2) as *const i8, std::arch::x86_64::_MM_HINT_T0) };
                        }
                        let packed = ((t as u32) << 24) | i;
                        let g = tbase[t] + i as usize;
                        i += 1;
                        let fp = ((e.eh >> 24) as u32) | 1;
                        let mut s = (e.eh as usize) & mask;
                        loop {
                            let x = unsafe { *tab.get_unchecked(s) };
                            if x == 0 {
                                unsafe { *tab.get_unchecked_mut(s) = ((fp as u64) << 32) | packed as u64 };
                                break;
                            }
                            if (x >> 32) as u32 == fp {
                                let p = x as u32;
                                let o = ev1[(p >> 24) as usize].get(b, (p & 0xffffff) as usize);
                                if o.eh == e.eh && o.eh2 == e.eh2 {
                                    if e.ingest < o.ingest {
                                        let go = tbase[(p >> 24) as usize] + (p & 0xffffff) as usize;
                                        loser[go >> 6] |= 1 << (go & 63);
                                        tab[s] = ((fp as u64) << 32) | packed as u64;
                                    } else {
                                        loser[g >> 6] |= 1 << (g & 63);
                                    }
                                    break;
                                }
                            }
                            s = (s + 1) & mask;
                        }
                    }
                }
            }
            // emit survivors in storage order
            for (t, tb) in ev1.iter().enumerate() {
                let mut g = tbase[t];
                for sl in tb.slices(b) {
                    for e in sl {
                        let lose = (loser[g >> 6] >> (g & 63)) & 1 != 0;
                        g += 1;
                        if lose || e.uh == 0 {
                            continue;
                        }
                        lprefetch(ltab, lmask, e.uh);
                        batch[nbat] = e;
                        nbat += 1;
                        if nbat == BATCH {
                            for e in &batch[..nbat] {
                                let (key, nref) = resolve(e.uh, e.uref);
                                out.push(bucket(key), Ev2 { key, nref, ts: e.ts, pc: e.pc });
                            }
                            nbat = 0;
                        }
                    }
                }
            }
            // flush pending records (they point into this bucket's blocks), then hand the
            // bucket's Ev1 blocks to the Ev2 output; nothing reads them again
            for e in &batch[..nbat] {
                let (key, nref) = resolve(e.uh, e.uref);
                out.push(bucket(key), Ev2 { key, nref, ts: e.ts, pc: e.pc });
            }
            nbat = 0;
            for tb in ev1.iter() {
                for &bp in tb.block_ptrs(b) {
                    out.recycle(bp as *mut u8);
                }
            }
        }
        for e in &batch[..nbat] {
            let (key, nref) = resolve(e.uh, e.uref);
            out.push(bucket(key), Ev2 { key, nref, ts: e.ts, pc: e.pc });
        }
        out.finish();
        out
    });
    lg("dedup done");

    // ---------------- sessionize ----------------
    let next = AtomicUsize::new(0);
    let nslots = (ncamp + 1) * ndays;
    let ud = out_dir.join("user_daily");
    let cd = out_dir.join("campaign_daily");
    std::fs::create_dir_all(&ud).expect("mkdir");
    std::fs::create_dir_all(&cd).expect("mkdir");
    // biggest person-buckets first (whales), so no thread is left with a huge bucket at the end
    let mut border: Vec<(usize, usize)> = (0..NB).map(|b| (ev2.iter().map(|t| t.len(b)).sum::<usize>(), b)).collect();
    border.sort_unstable_by(|a, b| b.cmp(a));
    let results: Vec<Vec<[i64; 4]>> = par(nt, |tid| {
        let mut urows: Vec<URow> = Vec::new();
        let mut acc: Vec<[i64; 4]> = vec![[0; 4]; nslots];
        let mut evs: Vec<Ev2> = Vec::new();
        let mut cls: Vec<Ck> = Vec::new();
        let mut starts: Vec<i64> = Vec::new();
        let mut sdays: Vec<u32> = Vec::new();
        let mut cc: Vec<(u32, i64)> = Vec::new();
        let mut catrev: Vec<i64> = vec![0; ncat];
        let mut touched: Vec<u16> = Vec::new();
        let mut pmap = U64Map::new(16);
        let mut names: Vec<u8> = Vec::new();
        const GROUP_ROWS: usize = 1 << 17;
        let mut uw: Option<pq::Writer> = None;
        // write the buffered rows as one row group (rows of a person never straddle groups)
        let flush_group = |uw: &mut Option<pq::Writer>, rows: &mut Vec<URow>, names: &mut Vec<u8>| {
            if uw.is_none() {
                *uw = Some(pq::Writer::new(&ud.join(format!("part-{tid}.parquet"))).expect("create user_daily"));
            }
            let w = uw.as_mut().unwrap();
            let n = rows.len();
            w.begin_group(n);
            {
                let names = &*names;
                w.str_col("user_id", rows.iter().map(|r| {
                    let o = (r.nref >> 16) as usize;
                    &names[o..o + (r.nref & 0xffff) as usize]
                }));
            }
            w.i32_col("day", true, rows.iter().map(|r| r.day));
            w.i32_col("sessions", false, rows.iter().map(|r| r.sessions as i32));
            w.i32_col("events", false, rows.iter().map(|r| r.events as i32));
            w.i64_col("revenue_cents", rows.iter().map(|r| r.revenue));
            w.opt_str_col(
                "top_category",
                rows.iter().map(|r| if r.top == NOCAT { None } else { Some(catnames[r.top as usize]) }),
            );
            w.end_group();
            rows.clear();
            names.clear();
        };

        let (mut cnt, mut ccnt, mut pid_of, mut offs, mut coffs, mut pos): (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) =
            Default::default();
        loop {
            let bi = next.fetch_add(1, Relaxed);
            if bi >= NB {
                break;
            }
            let b = border[bi].1;
            // group by person: dense ids, counting sort, then a small ts sort per person
            let n: usize = ev2.iter().map(|t| t.len(b)).sum();
            if n == 0 {
                continue;
            }
            pmap.reset(n / 8 + 16);
            cnt.clear();
            pid_of.clear();
            for t in &ev2 {
                for sl in t.slices(b) {
                    for e in sl {
                        let nx = cnt.len() as u32;
                        let (p, ins) = pmap.get_or_insert(e.key, nx);
                        if ins {
                            cnt.push(0);
                        }
                        cnt[p as usize] += 1;
                        pid_of.push(p);
                    }
                }
            }
            let np = cnt.len();
            offs.clear();
            offs.push(0);
            let mut acc_n = 0u32;
            for &x in &cnt {
                acc_n += x;
                offs.push(acc_n);
            }
            pos.clear();
            pos.extend_from_slice(&offs[..np]);
            evs.clear();
            evs.reserve(n);
            unsafe { evs.set_len(n) };
            let mut k = 0usize;
            for t in &ev2 {
                for sl in t.slices(b) {
                    for e in sl {
                        let p = pid_of[k] as usize;
                        k += 1;
                        evs[pos[p] as usize] = *e;
                        pos[p] += 1;
                    }
                }
            }
            for p in 0..np {
                let g = &mut evs[offs[p] as usize..offs[p + 1] as usize];
                if g.len() > 1 {
                    g.sort_unstable_by_key(|e| e.ts);
                }
            }

            // clicks of the same persons
            ccnt.clear();
            ccnt.resize(np, 0);
            pid_of.clear();
            for t in &cks {
                for sl in t.slices(b) {
                    for c in sl {
                        match pmap.get(c.key) {
                            Some(p) => {
                                ccnt[p as usize] += 1;
                                pid_of.push(p);
                            }
                            None => pid_of.push(u32::MAX),
                        }
                    }
                }
            }
            coffs.clear();
            coffs.push(0);
            let mut acc_n = 0u32;
            for &x in &ccnt {
                acc_n += x;
                coffs.push(acc_n);
            }
            pos.clear();
            pos.extend_from_slice(&coffs[..np]);
            cls.clear();
            cls.reserve(acc_n as usize);
            unsafe { cls.set_len(acc_n as usize) };
            let mut k = 0usize;
            for t in &cks {
                for sl in t.slices(b) {
                    for c in sl {
                        let p = pid_of[k];
                        k += 1;
                        if p != u32::MAX {
                            cls[pos[p as usize] as usize] = *c;
                            pos[p as usize] += 1;
                        }
                    }
                }
            }
            for p in 0..np {
                let e = &evs[offs[p] as usize..offs[p + 1] as usize];
                let c = &mut cls[coffs[p] as usize..coffs[p + 1] as usize];
                if c.len() > 1 {
                    c.sort_unstable_by(|a, b| a.ts.cmp(&b.ts).then_with(|| getref(a.cref).cmp(getref(b.cref))));
                }
                let c = &*c;
                // prefetch the name strings of the next persons (random places in the input)
                if p + 4 < np {
                    let r = evs[offs[p + 4] as usize].nref;
                    unsafe { std::arch::x86_64::_mm_prefetch((r >> 16) as *const i8, std::arch::x86_64::_MM_HINT_T0) };
                }
                let nm = getref(e[0].nref);
                let nref = ((names.len() as u64) << 16) | nm.len() as u64;
                names.extend_from_slice(nm);

                starts.clear();
                sdays.clear();
                let mut cur_day: u32 = u32::MAX;
                let (mut ds, mut de, mut dr) = (0i64, 0i64, 0i64);
                let mut k = 0usize;
                let mut ck = 0usize;
                while k < e.len() {
                    let start = e[k].ts as i64;
                    let day = ((cws + start).div_euclid(DAY) - day0) as u32;
                    if day != cur_day {
                        if cur_day != u32::MAX {
                            flush_row(&mut urows, nref, day0, cur_day, ds, de, dr, &mut catrev, &mut touched, &cat_rank);
                        }
                        cur_day = day;
                        ds = 0;
                        de = 0;
                        dr = 0;
                    }
                    let mut last = start;
                    let mut nev = 0i64;
                    let mut rev = 0i64;
                    let mut purch = false;
                    while k < e.len() && e[k].ts as i64 - last <= GAP {
                        let x = &e[k];
                        last = x.ts as i64;
                        nev += 1;
                        let cat = x.pc >> 24;
                        if cat != 255 {
                            let price = (x.pc & 0xff_ffff) as i64;
                            rev += price;
                            purch = true;
                            if catrev[cat as usize] == 0 {
                                touched.push(cat as u16);
                            }
                            catrev[cat as usize] += price;
                        }
                        k += 1;
                    }
                    ds += 1;
                    de += nev;
                    dr += rev;
                    // credit
                    while ck < c.len() && c[ck].ts as i64 <= start {
                        ck += 1;
                    }
                    let mut camp = organic;
                    if ck > 0 {
                        let tt = c[ck - 1].ts as i64;
                        if start - tt <= WEEK {
                            let mut q = ck - 1;
                            while q > 0 && c[q - 1].ts as i64 == tt {
                                q -= 1;
                            }
                            camp = c[q].camp;
                        }
                    }
                    let a = &mut acc[camp as usize * ndays + day as usize];
                    a[0] += 1;
                    a[1] += purch as i64;
                    a[2] += rev;
                    if !c.is_empty() {
                        starts.push(start);
                        sdays.push(day);
                    }
                }
                if cur_day != u32::MAX {
                    flush_row(&mut urows, nref, day0, cur_day, ds, de, dr, &mut catrev, &mut touched, &cat_rank);
                }
                // assists
                if !c.is_empty() && !starts.is_empty() {
                    cc.clear();
                    cc.extend(c.iter().map(|x| (x.camp, x.ts as i64)));
                    cc.sort_unstable();
                    let mut q = 0;
                    while q < cc.len() {
                        let camp = cc[q].0;
                        let mut lo = cc[q].1;
                        let mut hi = lo + WEEK;
                        q += 1;
                        loop {
                            let more = q < cc.len() && cc[q].0 == camp;
                            if more && cc[q].1 <= hi + 1 {
                                hi = cc[q].1 + WEEK;
                                q += 1;
                                continue;
                            }
                            // emit [lo, hi]
                            let mut a = starts.partition_point(|&s| s < lo);
                            let bnd = starts.partition_point(|&s| s <= hi);
                            while a < bnd {
                                let d = sdays[a];
                                let e2 = a + sdays[a..bnd].partition_point(|&x| x <= d);
                                acc[camp as usize * ndays + d as usize][3] += (e2 - a) as i64;
                                a = e2;
                            }
                            if more {
                                lo = cc[q].1;
                                hi = lo + WEEK;
                                q += 1;
                            } else {
                                break;
                            }
                        }
                    }
                }
                if urows.len() >= GROUP_ROWS {
                    flush_group(&mut uw, &mut urows, &mut names);
                }
            }
        }
        if !urows.is_empty() || uw.is_none() {
            flush_group(&mut uw, &mut urows, &mut names);
        }
        if let Some(w) = uw.take() {
            w.finish().expect("write user_daily");
        }
        acc
    });
    lg("sessionized+user_daily");

    // ---------------- campaign_daily ----------------
    let mut tot = vec![[0i64; 4]; nslots];
    for a in &results {
        for (t, x) in tot.iter_mut().zip(a) {
            for q in 0..4 {
                t[q] += x[q];
            }
        }
    }
    {
        let mut cid: Vec<&[u8]> = Vec::new();
        let mut ch: Vec<&[u8]> = Vec::new();
        let mut dd: Vec<i32> = Vec::new();
        let (mut s, mut p, mut r, mut a) = (vec![], vec![], vec![], vec![]);
        for c in 0..=ncamp {
            for d in 0..ndays {
                let x = tot[c * ndays + d];
                if x[0] > 0 || x[3] > 0 {
                    if c == ncamp {
                        cid.push(b"organic");
                        ch.push(b"organic");
                    } else {
                        cid.push(crows[c].0);
                        ch.push(crows[c].1);
                    }
                    dd.push((day0 + d as i64) as i32);
                    s.push(x[0]);
                    p.push(x[1]);
                    r.push(x[2]);
                    a.push(if c == ncamp { 0 } else { x[3] });
                }
            }
        }
        let n = dd.len();
        let mut w = pq::Writer::new(&cd.join("part-0.parquet")).expect("create campaign_daily");
        w.begin_group(n);
        w.str_col("campaign_id", cid.iter().copied());
        w.str_col("channel", ch.iter().copied());
        w.i32_col("day", true, dd.iter().copied());
        w.i64_col("sessions", s.iter().copied());
        w.i64_col("purchasing_sessions", p.iter().copied());
        w.i64_col("revenue_cents", r.iter().copied());
        w.i64_col("assisted_sessions", a.iter().copied());
        w.end_group();
        w.finish().expect("write campaign_daily");
    }
    lg("written");
    // ======================= PROBE (diagnostics only) =======================
    {
        let rd = |p: &str| std::fs::read_to_string(p).unwrap_or_else(|e| format!("<{e}>"));
        let cpu = rd("/proc/cpuinfo");
        let model = cpu.lines().find(|l| l.starts_with("model name")).unwrap_or("").to_string();
        let flags = cpu.lines().find(|l| l.starts_with("flags")).unwrap_or("").to_string();
        eprintln!("PROBE cpu: {model}; logical={} avx512f={} ", cpu.matches("processor").count(), flags.contains("avx512f"));
        eprintln!("PROBE kernel: {}", rd("/proc/version").trim());
        eprintln!("PROBE thp enabled={} defrag={}", rd("/sys/kernel/mm/transparent_hugepage/enabled").trim(), rd("/sys/kernel/mm/transparent_hugepage/defrag").trim());
        eprintln!("PROBE cgroup cpu.max={} memory.max={}", rd("/sys/fs/cgroup/cpu.max").trim(), rd("/sys/fs/cgroup/memory.max").trim());
        let mi = rd("/proc/meminfo");
        for l in mi.lines().take(5) { eprintln!("PROBE meminfo {l}"); }
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
        eprintln!("PROBE affinity cpus={} nt={}", unsafe { libc::CPU_COUNT(&set) }, nt);
        let up = rd("/proc/uptime");
        let st = rd("/proc/self/stat");
        eprintln!("PROBE uptime={} selfstat_starttime_field22={}", up.trim(), st.split_whitespace().nth(21).unwrap_or(""));
        eprintln!("PROBE input bytes events={} clicks={} links={}", ev_files.iter().map(|f| f.len()).sum::<usize>(), ck_files.iter().map(|f| f.len()).sum::<usize>(), link_files.iter().map(|f| f.len()).sum::<usize>());

        // fresh memory zeroing: 1 thread 512MB, then nt threads x 256MB
        {
            let t = Instant::now();
            let mut v: Vec<u8> = Vec::with_capacity(512 << 20);
            unsafe { v.set_len(512 << 20) };
            let mut i = 0; while i < v.len() { v[i] = 1; i += 4096; }
            eprintln!("PROBE zero 1thr 512MB: {:.3}s", t.elapsed().as_secs_f64());
            std::mem::forget(v);
            let t = Instant::now();
            par(nt, |_| { let mut v: Vec<u8> = Vec::with_capacity(256 << 20); unsafe { v.set_len(256 << 20) }; let mut i = 0; while i < v.len() { v[i] = 1; i += 4096; } std::mem::forget(v); });
            eprintln!("PROBE zero {nt}thr x256MB: {:.3}s", t.elapsed().as_secs_f64());
            // touching already-backed memory (second pass) for comparison
            let t = Instant::now();
            let mut v: Vec<u8> = vec![0u8; 512 << 20];
            let z = t.elapsed().as_secs_f64();
            let t = Instant::now();
            v.iter_mut().step_by(64).for_each(|x| *x = 2);
            eprintln!("PROBE vec![0;512MB] {:.3}s, rewrite backed 512MB {:.3}s", z, t.elapsed().as_secs_f64());
            std::mem::forget(v);
        }
        // memory bandwidth: copy 512MB (backed) single / nt threads
        {
            let a: Vec<u8> = vec![1u8; 256 << 20];
            let mut b: Vec<u8> = vec![0u8; 256 << 20];
            b.copy_from_slice(&a);
            let t = Instant::now();
            b.copy_from_slice(&a);
            eprintln!("PROBE memcpy 256MB 1thr: {:.3}s", t.elapsed().as_secs_f64());
        }
        // parse throughput on repeated event chunks (~1.2 GB of input)
        let echunks = chunks_of(&ev_files, 2 << 20);
        let ebytes: usize = echunks.iter().map(|c| c.len()).sum();
        let reps = ((1200usize << 20) / ebytes.max(1)).max(1);
        let mut work: Vec<&'static [u8]> = Vec::new();
        for _ in 0..reps { work.extend_from_slice(&echunks); }
        let wbytes = ebytes * reps;
        let lines_est = wbytes / 88;
        for rep in 0..2 {
            let t = Instant::now();
            let mut cnt = 0usize;
            for &c in &work { scan::<7>(c, |f| cnt += f[0].len() & 1); }
            let d = t.elapsed().as_secs_f64();
            eprintln!("PROBE rep{rep} scan-only 1thr: {:.3}s {:.2}GB/s {:.1}ns/line ({cnt})", d, wbytes as f64 / d / 1e9, d * 1e9 / lines_est as f64);
        }
        for rep in 0..2 {
            let t = Instant::now();
            let mut out: Parts<Ev1> = Parts::new(NB, 16 << 10);
            for &c in &work { parse_chunk(&mut out, c); }
            out.finish();
            let d = t.elapsed().as_secs_f64();
            eprintln!("PROBE rep{rep} full event parse 1thr: {:.3}s {:.1}ns/line", d, d * 1e9 / lines_est as f64);
        }
        for threads in [2usize, 4, 8, nt].iter().copied().collect::<std::collections::BTreeSet<_>>() {
            for rep in 0..2 {
                let next = AtomicUsize::new(0);
                let t = Instant::now();
                par(threads, |_| {
                    let mut out: Parts<Ev1> = Parts::new(NB, 16 << 10);
                    loop { let i = next.fetch_add(1, Relaxed); if i >= work.len() { break; } parse_chunk(&mut out, work[i]); }
                    out.finish();
                    std::mem::forget(out);
                });
                let d = t.elapsed().as_secs_f64();
                eprintln!("PROBE rep{rep} full event parse {threads}thr: {:.3}s  {:.1}ns/line/thread-equivalent  {:.2}GB/s", d, d * 1e9 * threads as f64 / lines_est as f64, wbytes as f64 / d / 1e9);
            }
        }
        let t = Instant::now();
        par(nt, |_| { let mut x = 0u64; for i in 0..200_000_000u64 { x = std::hint::black_box(x.wrapping_mul(6364136223846793005) ^ (x >> 7) ^ i); } std::hint::black_box(x); });
        eprintln!("PROBE spin 200M mul-add per thread on {nt} threads: {:.3}s", t.elapsed().as_secs_f64());
        let t = Instant::now();
        { let mut x = 0u64; for i in 0..200_000_000u64 { x = std::hint::black_box(x.wrapping_mul(6364136223846793005) ^ (x >> 7) ^ i); } std::hint::black_box(x); }
        eprintln!("PROBE spin 200M mul-add 1 thread: {:.3}s", t.elapsed().as_secs_f64());
        eprintln!("PROBE total elapsed {:.3}s", t0.elapsed().as_secs_f64());
        // make the dev check fail on purpose so the logs are returned
        // one extra, valid-but-wrong row so the dev check reports a differing row (and our logs)
        let mut w = pq::Writer::new(&cd.join("part-probe.parquet")).expect("probe file");
        w.begin_group(1);
        w.str_col("campaign_id", [&b"probe_row"[..]].into_iter());
        w.str_col("channel", [&b"probe"[..]].into_iter());
        w.i32_col("day", true, [day0 as i32].into_iter());
        w.i64_col("sessions", [1i64].into_iter());
        w.i64_col("purchasing_sessions", [0i64].into_iter());
        w.i64_col("revenue_cents", [0i64].into_iter());
        w.i64_col("assisted_sessions", [0i64].into_iter());
        w.end_group();
        w.finish().expect("probe file");
        eprintln!("PROBE done: wrote one intentional extra campaign_daily row");
    }
    std::process::exit(0);
}


fn build_links(link_files: &[&'static [u8]]) -> (Vec<[u64; 3]>, usize) {
    let tl = Instant::now();
    let lchunks = chunks_of(link_files, 1 << 20);
    #[inline(always)]
    fn prefix(s: &[u8]) -> u64 {
        if s.len() >= 8 {
            u64::from_be_bytes(s[..8].try_into().unwrap())
        } else {
            let mut b = [0u8; 8];
            b[..s.len()].copy_from_slice(s);
            u64::from_be_bytes(b)
        }
    }
    let mut edges: Vec<[u64; 6]> = Vec::new();
    for &c in &lchunks {
        scan::<2>(c, |f| {
            if !f[0].is_empty() && !f[1].is_empty() && f[0].len() <= 0xffff && f[1].len() <= 0xffff {
                edges.push([hash(f[0]), mkref(f[0]), prefix(f[0]), hash(f[1]), mkref(f[1]), prefix(f[1])]);
            }
        });
    }
    let nlinks = edges.len();
    let t_parse = tl.elapsed().as_secs_f64();
    // id hash -> node, one combined open-addressing table sized for the worst case (no rehash)
    let cap = (nlinks * 4).next_power_of_two().max(16);
    let mask = cap - 1;
    let mut tab: Vec<(u64, u32)> = vec![(0, 0); cap];
    let tabp = tab.as_ptr() as usize;
    let mut node_h: Vec<u64> = Vec::with_capacity(nlinks * 2);
    let mut node_r: Vec<u64> = Vec::with_capacity(nlinks * 2);
    let mut node_p: Vec<u64> = Vec::with_capacity(nlinks * 2); // first 8 bytes, big-endian
    let mut parent: Vec<u32> = Vec::with_capacity(nlinks * 2);
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            let gp = p[p[x as usize] as usize];
            p[x as usize] = gp;
            x = gp;
        }
        x
    }
    let mut node_of = |h: u64, r: u64, pf: u64, node_h: &mut Vec<u64>, node_r: &mut Vec<u64>, node_p: &mut Vec<u64>, parent: &mut Vec<u32>| -> u32 {
        let mut s = (h as usize) & mask;
        loop {
            let (k, v) = tab[s];
            if k == h {
                return v;
            }
            if k == 0 {
                let n = node_h.len() as u32;
                tab[s] = (h, n);
                node_h.push(h);
                node_r.push(r);
                node_p.push(pf);
                parent.push(n);
                return n;
            }
            s = (s + 1) & mask;
        }
    };
    for (i, &[ha, ra, pa, hb, rb, pb]) in edges.iter().enumerate() {
        if let Some(nx) = edges.get(i + 12) {
            unsafe {
                use std::arch::x86_64::*;
                let tp = tabp as *const (u64, u32);
                _mm_prefetch(tp.add((nx[0] as usize) & mask) as *const i8, _MM_HINT_T0);
                _mm_prefetch(tp.add((nx[3] as usize) & mask) as *const i8, _MM_HINT_T0);
            }
        }
        let a = node_of(ha, ra, pa, &mut node_h, &mut node_r, &mut node_p, &mut parent);
        let b = node_of(hb, rb, pb, &mut node_h, &mut node_r, &mut node_p, &mut parent);
        let ra_ = find(&mut parent, a);
        let rb_ = find(&mut parent, b);
        if ra_ != rb_ {
            if ra_ < rb_ {
                parent[rb_ as usize] = ra_;
            } else {
                parent[ra_ as usize] = rb_;
            }
        }
    }
    parts::donate(edges);
    parts::donate(tab);
    let t_uf = tl.elapsed().as_secs_f64();
    let nn = node_h.len();
    // name of each component: byte-wise smallest id (compare prefixes first)
    let mut minnode: Vec<u32> = vec![u32::MAX; nn];
    let mut root: Vec<u32> = vec![0; nn];
    for x in 0..nn {
        let r = find(&mut parent, x as u32) as usize;
        root[x] = r as u32;
        let m = minnode[r];
        let smaller = if m == u32::MAX {
            true
        } else {
            let (px, pm) = (node_p[x], node_p[m as usize]);
            if px != pm {
                px < pm
            } else {
                getref(node_r[x]) < getref(node_r[m as usize])
            }
        };
        if smaller {
            minnode[r] = x as u32;
        }
    }
    let t_min = tl.elapsed().as_secs_f64();
    // final lookup table: id hash -> (name hash, name ref)
    let lcap = (nn * 2).next_power_of_two().max(16);
    let mut ltab: Vec<[u64; 3]> = vec![[0; 3]; lcap];
    let lmask = lcap - 1;
    for x in 0..nn {
        let m = minnode[root[x] as usize] as usize;
        let h = node_h[x];
        let mut s = (h as usize) & lmask;
        while ltab[s][0] != 0 {
            s = (s + 1) & lmask;
        }
        ltab[s] = [h, node_h[m], node_r[m]];
    }
    eprintln!("links: {} lines, {} ids in {:.3}s (parse {:.3} uf {:.3} min {:.3})", nlinks, nn, tl.elapsed().as_secs_f64(), t_parse, t_uf, t_min);
    parts::donate(node_h);
    parts::donate(node_r);
    parts::donate(node_p);
    parts::donate(parent);
    parts::donate(minnode);
    parts::donate(root);
    (ltab, lmask)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn flush_row(urows: &mut Vec<URow>, nref: u64, day0: i64, day: u32, ds: i64, de: i64, dr: i64,
             catrev: &mut [i64], touched: &mut Vec<u16>, cat_rank: &[u32]) {
    let mut top = NOCAT;
    let mut best = 0i64;
    for &t in touched.iter() {
        let v = catrev[t as usize];
        if top == NOCAT || v > best || (v == best && cat_rank[t as usize] < cat_rank[top as usize]) {
            top = t;
            best = v;
        }
        catrev[t as usize] = 0;
    }
    touched.clear();
    urows.push(URow { nref, revenue: dr, day: (day0 + day as i64) as i32, sessions: ds as u32, events: de as u32, top });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f3_13_matches_generic() {
        let mut x: u64 = 0x1234_5678_9abc_def1;
        let alphabet = b"0123456789000000000999999999.-e +Z\x00\xff/:";
        for _ in 0..3_000_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let mut s = [0u8; 13];
            let mut y = x;
            for c in s.iter_mut() {
                // mostly digits, sometimes something else
                let r = (y % 97) as usize;
                *c = if r < 90 { b'0' + (r % 10) as u8 } else { alphabet[(y >> 8) as usize % alphabet.len()] };
                y = y.rotate_left(7) ^ 0x9e37_79b9_7f4a_7c15;
            }
            assert_eq!(parse_f3_13(&s), parse_f3(&s), "{:?}", std::str::from_utf8(&s));
        }
        assert_eq!(parse_f3_13(b"0000000000000"), Some(0));
        assert_eq!(parse_f3_13(b"9999999999999"), Some(9_999_999_999_999));
        assert_eq!(parse_f3_13(b"1791071890000"), Some(1_791_071_890_000));
    }
    #[test]
    fn f12_cases() {
        assert_eq!(parse_ts(b"2026-10-03T23:58:10Z"), parse_ts(b"2026-10-03T19:58:10-04:00"));
        assert_eq!(parse_ts(b"2026-10-03T23:58:10Z"), Some(1_791_071_890_000));
        assert_eq!(parse_ts(b"2028-02-29T00:00:00Z").is_some(), true);
        assert_eq!(parse_ts(b"2027-02-29T00:00:00Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T24:00:00Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:59:60Z"), None);
        assert_eq!(parse_ts(b"2026-13-03T23:59:00Z"), None);
        assert_eq!(parse_ts(b"2026-10-03t23:59:00Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:59:00z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:59:00+24:00"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:59:00+05:60"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:59:00+0200"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:59:00-00:00"), parse_ts(b"2026-10-03T23:59:00Z"));
        assert_eq!(parse_ts(b"0001791071890000"), Some(1_791_071_890_000));
        assert_eq!(parse_ts(b"1791071890000.0"), None);
        assert_eq!(parse_ts(b"-5"), None);
        assert_eq!(parse_ts(b""), None);
        assert_eq!(parse_ts(b"123456789012345678901"), None);
    }
}
