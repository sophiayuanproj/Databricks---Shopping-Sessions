# Shopping Sessions: native Rust pipeline

One static binary (no runtime dependencies) on an empty image. `docker build .` builds it from
source; the container runs `/shop`, which reads `$DATA_DIR` and writes `$OUTPUT_DIR`.

## Design

All phases run on `NCPU` threads with dynamic work distribution.

1. **Map inputs.** Every CSV is `mmap`ed with `MAP_POPULATE` (files in parallel) and split into
   ~2 MB line-aligned chunks. The partition count (256–4096) is chosen from the input size.
2. **Parse (one shared queue: event chunks, then click chunks).**
   * An AVX2 scanner classifies `\n`, `,` and `"` 64 bytes at a time and walks only the
     comma/newline positions. Quoted fields are recognised branch-free; stray quotes are
     caught by comparing the line's quote count (popcount) with 2 × quoted fields. `\r` is an
     ordinary byte, and the last line may lack a newline.
   * Timestamps: SWAR validation/decoding of F1/F2 and of 13-digit F3, a generic F3 path, and
     a month-start table for dates; everything outside the rules is rejected.
   * Each event becomes a 40-byte record (96-bit event_id identity from two independent
     hashes, user hash and pointer, times relative to `window_start − 7 d`, price/category)
     appended to a partition chosen by event_id hash. Partitions are block lists in huge-page
     arenas, written through per-partition 8-record staging buffers flushed with 64-byte
     non-temporal stores (software write-combining).
   * Meanwhile thread 0 builds the linked-ID table: union-find over link endpoints, each
     component named by its byte-wise smallest ID, and an `id hash → (name hash, name)` table.
   * Clicks are validated, resolved to their person and partitioned by person.
3. **Dedup** per event_id partition: an 8-byte fingerprint table finds copies, the earliest
   ingest wins and losers are marked in a bitset; the kept copy's validity decides. Survivors
   are resolved to their person (batched, prefetched lookups) and re-partitioned by person
   into the memory of the partitions already consumed.
4. **Sessionize** per person partition (largest first): dense person ids and a counting sort,
   a per-person time sort, sessions split on gaps over 30 min, credit through a merge with
   the person's clicks (latest click ≤ start within 7 days, ties to the smallest click_id),
   assists by merging each campaign's `[click, click + 7 d]` intervals and binary-searching
   session starts per day. Each thread streams its user_daily rows to its own Parquet file in
   row groups.
5. **campaign_daily** is summed from per-thread dense (campaign × day) counters.

Parquet is written by a small built-in writer (PLAIN, uncompressed, `DATE` and `STRING`
logical types), readable by pyarrow, polars and DuckDB.

## Testing

```sh
cargo test --release --target x86_64-unknown-linux-gnu        # parser unit tests
cargo build --release --target x86_64-unknown-linux-gnu
python3 scripts/run_fixtures.py fixtures -- target/x86_64-unknown-linux-gnu/release/shop
```
