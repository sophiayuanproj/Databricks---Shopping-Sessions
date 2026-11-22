//! Minimal Parquet writer: one row group, one PLAIN uncompressed data page per column.

struct Tw {
    b: Vec<u8>,
    last: i16,
    stack: Vec<i16>,
}

impl Tw {
    fn new() -> Self {
        Tw { b: Vec::new(), last: 0, stack: Vec::new() }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.b.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.b.push(v as u8);
    }
    fn zz(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }
    fn field(&mut self, id: i16, ty: u8) {
        let d = id - self.last;
        if d > 0 && d <= 15 {
            self.b.push(((d as u8) << 4) | ty);
        } else {
            self.b.push(ty);
            self.varint(Self::zz(id as i64));
        }
        self.last = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, 5);
        self.varint(Self::zz(v as i64));
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, 6);
        self.varint(Self::zz(v));
    }
    fn bin(&mut self, id: i16, s: &[u8]) {
        self.field(id, 8);
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s);
    }
    fn sbegin(&mut self, id: i16) {
        self.field(id, 12);
        self.stack.push(self.last);
        self.last = 0;
    }
    fn send(&mut self) {
        self.b.push(0);
        self.last = self.stack.pop().unwrap();
    }
    fn ebegin(&mut self) {
        self.stack.push(self.last);
        self.last = 0;
    }
    fn lbegin(&mut self, id: i16, ety: u8, n: usize) {
        self.field(id, 9);
        if n < 15 {
            self.b.push(((n as u8) << 4) | ety);
        } else {
            self.b.push(0xF0 | ety);
            self.varint(n as u64);
        }
    }
    fn lstr(&mut self, s: &[u8]) {
        self.varint(s.len() as u64);
        self.b.extend_from_slice(s);
    }
    fn li32(&mut self, v: i32) {
        self.varint(Self::zz(v as i64));
    }
}

struct ChunkMeta {
    name: String,
    ptype: i32,
    offset: i64,
    size: i64,
    nvals: i64,
}

/// Streaming writer: columns are encoded straight into a buffered file, one row group at a time.
pub struct Writer {
    f: std::io::BufWriter<std::fs::File>,
    off: u64,
    groups: Vec<(Vec<(ChunkMeta, u8)>, usize)>, // kind: 0 str, 1 opt str, 2 i64, 3 date, 4 i32
    cur: Vec<(ChunkMeta, u8)>,
    nrows: usize,
    scratch: Vec<u8>,
}

impl Writer {
    pub fn new(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Write;
        let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        f.write_all(b"PAR1")?;
        Ok(Writer { f, off: 4, groups: Vec::new(), cur: Vec::new(), nrows: 0, scratch: Vec::new() })
    }
    pub fn begin_group(&mut self, nrows: usize) {
        self.cur.clear();
        self.nrows = nrows;
    }
    pub fn end_group(&mut self) {
        let g = std::mem::take(&mut self.cur);
        self.groups.push((g, self.nrows));
    }
    fn put(&mut self, b: &[u8]) {
        use std::io::Write;
        self.f.write_all(b).expect("write parquet");
        self.off += b.len() as u64;
    }
    fn header(&mut self, data_len: usize) {
        let mut h = Tw::new();
        h.i32(1, 0);
        h.i32(2, data_len as i32);
        h.i32(3, data_len as i32);
        h.sbegin(5);
        h.i32(1, self.nrows as i32);
        h.i32(2, 0);
        h.i32(3, 3);
        h.i32(4, 3);
        h.send();
        h.b.push(0);
        self.put(&h.b);
    }
    fn push_meta(&mut self, name: &str, ptype: i32, offset: u64, kind: u8) {
        let size = (self.off - offset) as i64;
        self.cur.push((ChunkMeta { name: name.to_string(), ptype, offset: offset as i64, size, nvals: self.nrows as i64 }, kind));
    }
    fn flush_scratch(&mut self) {
        let sc = std::mem::take(&mut self.scratch);
        self.put(&sc);
        self.scratch = sc;
        self.scratch.clear();
    }
    pub fn str_col<'a, I: Iterator<Item = &'a [u8]> + Clone>(&mut self, name: &str, it: I) {
        let len: usize = it.clone().map(|s| 4 + s.len()).sum();
        let off = self.off;
        self.header(len);
        self.scratch.clear();
        for s in it {
            self.scratch.extend_from_slice(&(s.len() as u32).to_le_bytes());
            self.scratch.extend_from_slice(s);
            if self.scratch.len() >= 1 << 16 {
                self.flush_scratch();
            }
        }
        self.flush_scratch();
        self.push_meta(name, 6, off, 0);
    }
    pub fn opt_str_col<'a, I: Iterator<Item = Option<&'a [u8]>> + Clone>(&mut self, name: &str, it: I) {
        let mut lv = Tw::new();
        let mut run_v = false;
        let mut run_n = 0u64;
        let mut len = 0usize;
        for x in it.clone() {
            let d = x.is_some();
            if let Some(s) = x {
                len += 4 + s.len();
            }
            if run_n > 0 && d != run_v {
                lv.varint(run_n << 1);
                lv.b.push(run_v as u8);
                run_n = 0;
            }
            run_v = d;
            run_n += 1;
        }
        if run_n > 0 {
            lv.varint(run_n << 1);
            lv.b.push(run_v as u8);
        }
        let off = self.off;
        self.header(4 + lv.b.len() + len);
        self.put(&(lv.b.len() as u32).to_le_bytes());
        self.put(&lv.b);
        self.scratch.clear();
        for s in it.flatten() {
            self.scratch.extend_from_slice(&(s.len() as u32).to_le_bytes());
            self.scratch.extend_from_slice(s);
            if self.scratch.len() >= 1 << 16 {
                self.flush_scratch();
            }
        }
        self.flush_scratch();
        self.push_meta(name, 6, off, 1);
    }
    pub fn i64_col<I: Iterator<Item = i64>>(&mut self, name: &str, it: I) {
        let off = self.off;
        self.header(self.nrows * 8);
        self.scratch.clear();
        for x in it {
            self.scratch.extend_from_slice(&x.to_le_bytes());
            if self.scratch.len() >= 1 << 16 {
                self.flush_scratch();
            }
        }
        self.flush_scratch();
        self.push_meta(name, 2, off, 2);
    }
    pub fn i32_col<I: Iterator<Item = i32>>(&mut self, name: &str, date: bool, it: I) {
        let off = self.off;
        self.header(self.nrows * 4);
        self.scratch.clear();
        for x in it {
            self.scratch.extend_from_slice(&x.to_le_bytes());
            if self.scratch.len() >= 1 << 16 {
                self.flush_scratch();
            }
        }
        self.flush_scratch();
        self.push_meta(name, 1, off, if date { 3 } else { 4 });
    }
    pub fn finish(mut self) -> std::io::Result<()> {
        use std::io::Write;
        let schema: Vec<(String, i32, u8)> = self.groups[0].0.iter().map(|(m, k)| (m.name.clone(), m.ptype, *k)).collect();
        let ncols = schema.len();
        let total_rows: usize = self.groups.iter().map(|g| g.1).sum();
        let mut t = Tw::new();
        t.i32(1, 1);
        t.lbegin(2, 12, ncols + 1);
        t.ebegin();
        t.bin(4, b"schema");
        t.i32(5, ncols as i32);
        t.send();
        for (name, ptype, kind) in &schema {
            t.ebegin();
            t.i32(1, *ptype);
            t.i32(3, if *kind == 1 { 1 } else { 0 });
            t.bin(4, name.as_bytes());
            match kind {
                0 | 1 => {
                    t.i32(6, 0);
                    t.sbegin(10);
                    t.sbegin(1);
                    t.send();
                    t.send();
                }
                3 => {
                    t.i32(6, 6);
                    t.sbegin(10);
                    t.sbegin(6);
                    t.send();
                    t.send();
                }
                _ => {}
            }
            t.send();
        }
        t.i64(3, total_rows as i64);
        t.lbegin(4, 12, self.groups.len());
        for (metas, nrows) in &self.groups {
            t.ebegin();
            t.lbegin(1, 12, metas.len());
            let mut total = 0i64;
            for (m, _) in metas {
                total += m.size;
                t.ebegin();
                t.i64(2, m.offset);
                t.sbegin(3);
                t.i32(1, m.ptype);
                t.lbegin(2, 5, 2);
                t.li32(0);
                t.li32(3);
                t.lbegin(3, 8, 1);
                t.lstr(m.name.as_bytes());
                t.i32(4, 0);
                t.i64(5, m.nvals);
                t.i64(6, m.size);
                t.i64(7, m.size);
                t.i64(9, m.offset);
                t.send();
                t.send();
            }
            t.i64(2, total);
            t.i64(3, *nrows as i64);
            t.send();
        }
        t.bin(6, b"shop");
        t.b.push(0);
        let tl = t.b.len() as u32;
        self.put(&t.b);
        self.put(&tl.to_le_bytes());
        self.put(b"PAR1");
        self.f.flush()
    }
}
