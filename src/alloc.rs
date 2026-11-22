//! Global allocator: large blocks come straight from mmap with MADV_HUGEPAGE
//! (fewer page faults / TLB misses); small blocks go to the system allocator.
use std::alloc::{GlobalAlloc, Layout, System};

pub struct Huge;
const BIG: usize = 1 << 20;
const HP: usize = 2 << 20;

#[inline]
fn round(n: usize) -> usize {
    (n + HP - 1) & !(HP - 1)
}

unsafe fn map_big(size: usize) -> *mut u8 {
    let sz = round(size);
    // over-map by one huge page so the block can start on a 2 MB boundary
    let p = libc::mmap(std::ptr::null_mut(), sz + HP, libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0);
    if p == libc::MAP_FAILED {
        return std::ptr::null_mut();
    }
    let a = p as usize;
    let al = (a + HP - 1) & !(HP - 1);
    if al > a {
        libc::munmap(a as *mut _, al - a);
    }
    let tail = (a + sz + HP) - (al + sz);
    if tail > 0 {
        libc::munmap((al + sz) as *mut _, tail);
    }
    libc::madvise(al as *mut _, sz, libc::MADV_HUGEPAGE);
    al as *mut u8
}

unsafe impl GlobalAlloc for Huge {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.size() >= BIG && l.align() <= 4096 { map_big(l.size()) } else { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if l.size() >= BIG && l.align() <= 4096 { map_big(l.size()) } else { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if l.size() >= BIG && l.align() <= 4096 { libc::munmap(p as *mut _, round(l.size())); } else { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let old = l.size();
        let ob = old >= BIG && l.align() <= 4096;
        let nb = new >= BIG && l.align() <= 4096;
        if ob && nb {
            let (os, ns) = (round(old), round(new));
            if os == ns {
                return p;
            }
            if ns < os {
                libc::munmap(p.add(ns) as *mut _, os - ns);
                return p;
            }
            // try to grow in place, else allocate aligned and move the pages
            let r = libc::mremap(p as *mut _, os, ns, 0);
            if r != libc::MAP_FAILED {
                return p;
            }
            let q = map_big(new);
            if q.is_null() {
                return q;
            }
            let r = libc::mremap(p as *mut _, os, os, libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED, q as *mut libc::c_void);
            if r == libc::MAP_FAILED {
                std::ptr::copy_nonoverlapping(p, q, old);
                libc::munmap(p as *mut _, os);
            }
            return q;
        }
        if !ob && !nb {
            return System.realloc(p, l, new);
        }
        let nl = Layout::from_size_align_unchecked(new, l.align());
        let q = self.alloc(nl);
        if !q.is_null() {
            std::ptr::copy_nonoverlapping(p, q, old.min(new));
            self.dealloc(p, l);
        }
        q
    }
}
