//! Partitioned append-only storage: per bucket, a list of fixed-size blocks carved
//! from large (huge-page backed) arenas owned by one thread.
use std::mem::MaybeUninit;

const ARENA_BYTES: usize = 64 << 20;

/// Already-touched memory regions (2 MB aligned) donated for reuse as blocks; reusing them
/// avoids paying for fresh zeroed pages.
static POOL: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Donate a Vec's buffer to the block pool (only large, page-aligned buffers; others are dropped).
pub fn donate<T>(v: Vec<T>) {
    let bytes = v.capacity() * std::mem::size_of::<T>();
    let p = v.as_ptr() as usize;
    if bytes >= (2 << 20) && p % (2 << 20) == 0 {
        std::mem::forget(v);
        POOL.lock().unwrap().push((p, bytes));
    }
}

const STAGE: usize = 8;

pub struct Parts<T: Copy> {
    stage: Vec<[MaybeUninit<T>; STAGE]>, // per-bucket write-combining buffer
    scnt: Vec<u8>,
    blk: usize,
    arenas: Vec<Box<[MaybeUninit<T>]>>,
    arena_used: usize,
    next_arena: usize,
    blocks: Vec<Vec<*mut T>>,
    free: Vec<*mut u8>, // recycled blocks (from another Parts) of at least blk * size bytes
    cur: Vec<(*mut T, u32)>, // current block ptr, used count
}
unsafe impl<T: Copy + Send> Send for Parts<T> {}
unsafe impl<T: Copy + Sync> Sync for Parts<T> {}

impl<T: Copy> Parts<T> {
    pub fn new(nb: usize, blk_bytes: usize) -> Self {
        let blk = (blk_bytes / std::mem::size_of::<T>()).max(16);
        let blk = if blk.is_power_of_two() { blk } else { blk.next_power_of_two() >> 1 };
        Self::with_blk(nb, blk)
    }
    /// Explicit block capacity (a multiple of the staging size; `get` needs a power of two).
    pub fn with_blk(nb: usize, blk: usize) -> Self {
        let nt_ok = (STAGE * std::mem::size_of::<T>()) % 64 == 0 && (blk * std::mem::size_of::<T>()) % 64 == 0;
        assert!(nt_ok, "record size not suited to 64-byte streaming");
        Parts { stage: vec![[MaybeUninit::uninit(); STAGE]; nb], scnt: vec![0; nb], blk, arenas: Vec::new(), arena_used: usize::MAX, next_arena: 0, blocks: vec![Vec::new(); nb], free: Vec::new(), cur: vec![(std::ptr::null_mut(), 0); nb] }
    }
    /// Bytes per block.
    pub fn block_bytes(&self) -> usize {
        self.blk * std::mem::size_of::<T>()
    }
    /// The blocks of bucket b (for recycling once the bucket is consumed).
    pub fn block_ptrs(&self, b: usize) -> &[*mut T] {
        &self.blocks[b]
    }
    /// Offer a dead block (64-byte aligned, at least block_bytes() long) for reuse.
    pub fn recycle(&mut self, p: *mut u8) {
        self.free.push(p);
    }
    #[cold]
    fn new_block(&mut self, b: usize) {
        if self.free.is_empty() {
            // carve blocks out of a donated region, if any
            let bb = self.block_bytes();
            if let Some((p, n)) = POOL.lock().unwrap().pop() {
                let mut off = 0;
                while off + bb <= n {
                    self.free.push((p + off) as *mut u8);
                    off += bb;
                }
                self.free.reverse();
            }
        }
        if let Some(p) = self.free.pop() {
            let p = p as *mut T;
            self.blocks[b].push(p);
            self.cur[b] = (p, 0);
            return;
        }
        let per = (ARENA_BYTES / std::mem::size_of::<T>()) / self.blk * self.blk;
        if self.arena_used == usize::MAX || self.arena_used + self.blk > per || self.next_arena == 0 {
            if self.next_arena >= self.arenas.len() {
                let mut v: Vec<MaybeUninit<T>> = Vec::with_capacity(per);
                unsafe { v.set_len(per) };
                self.arenas.push(v.into_boxed_slice());
            }
            self.next_arena += 1;
            self.arena_used = 0;
        }
        let a = &mut self.arenas[self.next_arena - 1];
        let p = unsafe { a.as_mut_ptr().add(self.arena_used) as *mut T };
        self.arena_used += self.blk;
        self.blocks[b].push(p);
        self.cur[b] = (p, 0);
    }
    /// Append directly (no staging).
    #[inline(always)]
    fn push_direct(&mut self, b: usize, x: T) {
        let c = unsafe { self.cur.get_unchecked_mut(b) };
        if c.0.is_null() || c.1 as usize == self.blk {
            self.new_block(b);
        }
        let c = unsafe { self.cur.get_unchecked_mut(b) };
        unsafe { c.0.add(c.1 as usize).write(x) };
        c.1 += 1;
    }
    /// Append through the bucket's staging buffer; full buffers are flushed to the
    /// block with non-temporal 64-byte stores (no read-for-ownership of cold lines).
    #[inline(always)]
    pub fn push(&mut self, b: usize, x: T) {
        let n = unsafe { *self.scnt.get_unchecked(b) } as usize;
        unsafe { self.stage.get_unchecked_mut(b).get_unchecked_mut(n).write(x) };
        if n + 1 == STAGE {
            self.flush_stage(b);
            unsafe { *self.scnt.get_unchecked_mut(b) = 0 };
        } else {
            unsafe { *self.scnt.get_unchecked_mut(b) = (n + 1) as u8 };
        }
    }
    #[inline(always)]
    fn flush_stage(&mut self, b: usize) {
        // blocks only ever receive whole stages until finish(), so a block has room
        // for a stage iff it is not full
        let c = unsafe { *self.cur.get_unchecked(b) };
        if c.0.is_null() || c.1 as usize == self.blk {
            self.new_block(b);
        }
        let c = unsafe { self.cur.get_unchecked_mut(b) };
        unsafe {
            use std::arch::x86_64::*;
            let src = self.stage.get_unchecked(b).as_ptr() as *const __m256i;
            let dst = c.0.add(c.1 as usize) as *mut __m256i;
            let words = STAGE * std::mem::size_of::<T>() / 32;
            for k in 0..words {
                _mm256_stream_si256(dst.add(k), _mm256_loadu_si256(src.add(k)));
            }
        }
        c.1 += STAGE as u32;
    }
    /// Move staged records into the blocks; call once before reading.
    pub fn finish(&mut self) {
        for b in 0..self.scnt.len() {
            let n = self.scnt[b] as usize;
            for k in 0..n {
                let x = unsafe { self.stage[b][k].assume_init() };
                self.push_direct(b, x);
            }
            self.scnt[b] = 0;
        }
        unsafe { std::arch::x86_64::_mm_sfence() };
    }
    pub fn len(&self, b: usize) -> usize {
        let n = self.blocks[b].len();
        if n == 0 { 0 } else { (n - 1) * self.blk + self.cur[b].1 as usize }
    }
    /// Iterate the bucket's items as slices.
    #[inline]
    pub fn slices(&self, b: usize) -> impl Iterator<Item = &[T]> + '_ {
        let n = self.blocks[b].len();
        self.blocks[b].iter().enumerate().map(move |(i, &p)| {
            let len = if i + 1 == n { self.cur[b].1 as usize } else { self.blk };
            unsafe { std::slice::from_raw_parts(p as *const T, len) }
        })
    }
    #[inline(always)]
    pub fn get(&self, b: usize, i: usize) -> &T {
        debug_assert!(self.blk.is_power_of_two());
        let sh = self.blk.trailing_zeros();
        unsafe { &*self.blocks.get_unchecked(b).get_unchecked(i >> sh).add(i & (self.blk - 1)) }
    }
}
