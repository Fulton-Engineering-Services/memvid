//! The untested state: a checkpoint taken while the ring sits on the exact-fit boundary.
//!
//! `EmbeddedWal::write_head` is a ring cursor, and since `6c9ba86` an exactly full ring is
//! allowed to leave it at `region_size` instead of folding it to 0. Nothing exercised a
//! checkpoint taken in that state:
//!
//! * `wrap_and_checkpoint` (`src/io/wal.rs`) does checkpoint an exactly full ring - two
//!   80-byte entries in a 160-byte region - but never reopens it, so the persisted value and
//!   the sentinel behaviour are both unobserved.
//! * `reopen_at_exact_fill_preserves_pending_records` (`tests/replay_integrity.rs`) reopens an
//!   exactly full ring but deliberately never checkpoints at the boundary.
//!
//! These two tests close that gap from both sides, and are deliberately ungated `#[test]`
//! functions: CI runs plain `cargo test`, whose default features are `lex`, `pdf_extract` and
//! `simd` and *not* `replay`, so anything gated on `replay` contributes zero tests there.
//!
//! The fixture is the one `tests/replay_integrity.rs` uses for its exact-divisor trigger, with
//! `region = 64 * entry` so the 64th record ends flush with the region end. The mid-ring
//! checkpoint is not decoration: it zeroes `pending_bytes`, and only then can the ring reach
//! the boundary with records still uncheckpointed instead of being refused by the
//! `pending_bytes + entry_size > region_size` guard in `append_entry`.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use memvid_core::constants::{MAGIC, SPEC_VERSION, WAL_OFFSET};
use memvid_core::io::header::HeaderCodec;
use memvid_core::io::wal::{EmbeddedWal, WalRecord};
use memvid_core::types::Header;
use tempfile::TempDir;

/// `[seq: u64][len: u32][reserved: 4][checksum: 32]`, duplicated from `src/io/wal.rs` so this
/// file reads the bytes a scanner would, without reaching into private constants.
const ENTRY_HEADER_SIZE: u64 = 48;
const ENTRY_HEADER_BYTES: usize = ENTRY_HEADER_SIZE as usize;

/// Non-zero, varying bytes, so no record boundary ever looks like an all-zero terminator.
fn payload_for(len: usize, i: usize) -> Vec<u8> {
    let mut p = vec![0u8; len];
    for (k, b) in p.iter_mut().enumerate() {
        *b = ((i.wrapping_mul(31).wrapping_add(k.wrapping_mul(7))) % 250 + 1) as u8;
    }
    p
}

/// The store lives inside the `TempDir` the calling test owns, so cleanup is RAII: a
/// panicking assertion still leaves nothing behind.
fn store_path(dir: &TempDir) -> PathBuf {
    dir.path().join("wal-boundary.mv2")
}

fn create_store(path: &Path, region: u64) -> (File, Header) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .expect("create scratch store");
    file.set_len(WAL_OFFSET + region + 4096)
        .expect("set_len scratch store");
    let header = Header {
        magic: MAGIC,
        version: SPEC_VERSION,
        footer_offset: WAL_OFFSET + region + 4096,
        wal_offset: WAL_OFFSET,
        wal_size: region,
        wal_checkpoint_pos: 0,
        wal_sequence: 0,
        toc_checksum: [0u8; 32],
    };
    HeaderCodec::write(&mut file, &header).expect("write header");
    file.sync_all().expect("sync header");
    (file, header)
}

/// Reopen through a *fresh* handle so the assertions describe what a reader recovers from the
/// bytes on disk, not what an in-memory object happens to remember.
fn reopen(path: &Path, header: &Header) -> EmbeddedWal {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("reopen scratch store");
    EmbeddedWal::open(&file, header).expect("reopen wal")
}

/// The header as it is encoded on disk, not the caller's in-memory copy.
fn persisted_header(path: &Path) -> Header {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open scratch store to read the header");
    HeaderCodec::read(&mut file).expect("decode persisted header")
}

/// The 48-byte entry header sitting at `position` inside the region, read straight off the
/// file. This is what `scan_records` looks at before it decides whether the chain continues.
fn read_entry_header(path: &Path, position: u64) -> [u8; ENTRY_HEADER_BYTES] {
    let mut file = OpenOptions::new()
        .read(true)
        .open(path)
        .expect("open scratch store for a raw entry read");
    file.seek(SeekFrom::Start(WAL_OFFSET + position))
        .expect("seek to entry header");
    let mut buf = [0u8; ENTRY_HEADER_BYTES];
    file.read_exact(&mut buf).expect("read entry header");
    buf
}

fn entry_header_fields(header: &[u8; ENTRY_HEADER_BYTES]) -> (u64, u32) {
    let sequence = u64::from_le_bytes(header[..8].try_into().expect("8 bytes of sequence"));
    let length = u32::from_le_bytes(header[8..12].try_into().expect("4 bytes of length"));
    (sequence, length)
}

fn sequences(records: &[WalRecord]) -> Vec<u64> {
    records.iter().map(|r| r.sequence).collect()
}

fn missing_sequences(want_from: u64, want_upto: u64, records: &[WalRecord]) -> Vec<u64> {
    let present: BTreeSet<u64> = records.iter().map(|r| r.sequence).collect();
    (want_from..=want_upto)
        .filter(|s| !present.contains(s))
        .collect()
}

/// `entry = 48 + 976 = 1_024` divides `region = 65_536` exactly, so the 64th append is an
/// exact fit: `write_head + entry_size == region_size`, the state that used to fold to 0.
const BOUNDARY_PAYLOAD: usize = 976;
const BOUNDARY_ENTRY: u64 = ENTRY_HEADER_SIZE + BOUNDARY_PAYLOAD as u64;
const BOUNDARY_REGION: u64 = 64 * BOUNDARY_ENTRY;
/// Checkpointed first so that `pending_bytes` is zeroed and the ring can fill the remaining
/// 16 slots without tripping the "region full" growth guard. Same call `Memvid::commit` makes
/// at 75% occupancy.
const CHECKPOINTED: usize = 48;
/// Records left uncheckpointed on the exactly full ring - the ones at risk.
const PENDING: usize = 16;
const TOTAL: usize = CHECKPOINTED + PENDING;

fn pending_bytes_expected() -> u64 {
    u64::try_from(PENDING).expect("fits") * BOUNDARY_ENTRY
}

fn first_pending_seq() -> u64 {
    u64::try_from(CHECKPOINTED + 1).expect("fits")
}

fn last_seq() -> u64 {
    u64::try_from(TOTAL).expect("fits")
}

fn slot_zero_expectation() -> (u64, u32) {
    (1, u32::try_from(BOUNDARY_PAYLOAD).expect("fits"))
}

/// The fixture arithmetic, asserted rather than assumed: if this stops holding, the tests
/// below are no longer driving the ring to an exact fit and their green means nothing.
fn assert_fixture_geometry() {
    assert_eq!(
        BOUNDARY_REGION % BOUNDARY_ENTRY,
        0,
        "fixture: the *entry* size (48 + payload = {BOUNDARY_ENTRY}) must divide the region \
         {BOUNDARY_REGION} exactly, or the last append is not an exact fit and the boundary \
         state is never reached (remainder {})",
        BOUNDARY_REGION % BOUNDARY_ENTRY,
    );
    assert_eq!(
        TOTAL,
        usize::try_from(BOUNDARY_REGION / BOUNDARY_ENTRY).expect("fits"),
        "fixture: {CHECKPOINTED} checkpointed + {PENDING} pending slots must be exactly the \
         {bound} slots the region holds",
        bound = BOUNDARY_REGION / BOUNDARY_ENTRY,
    );
    assert!(
        PENDING > 0,
        "fixture: with no uncheckpointed records there is nothing for a misplaced terminator to hide"
    );
}

/// Drive the ring to the boundary: 48 appends, a checkpoint, then 16 more so the ring is
/// exactly full with 16 records still uncheckpointed.
fn drive_to_boundary(wal: &mut EmbeddedWal, header: &mut Header, file: &mut File) {
    for i in 0..CHECKPOINTED {
        wal.append_entry(&payload_for(BOUNDARY_PAYLOAD, i))
            .unwrap_or_else(|e| panic!("append #{i} on an empty ring must succeed: {e:?}"));
    }
    assert!(
        wal.should_checkpoint(),
        "fixture: the 75% occupancy rule (`WAL_CHECKPOINT_THRESHOLD`) must fire at record \
         {CHECKPOINTED} (pending_bytes={} of region={BOUNDARY_REGION}); without that checkpoint \
         `append_entry` refuses record {} with `embedded WAL region full` and the ring never \
         reaches the boundary while records are pending",
        wal.stats().pending_bytes,
        CHECKPOINTED + 1,
    );
    wal.record_checkpoint(header)
        .expect("checkpoint the first 48 records");
    HeaderCodec::write(&mut *file, header).expect("persist the mid-ring checkpoint");
    file.sync_all().expect("sync the mid-ring checkpoint");
    assert_eq!(
        header.wal_checkpoint_pos,
        u64::try_from(CHECKPOINTED).expect("fits") * BOUNDARY_ENTRY,
        "fixture: the mid-ring checkpoint must have advanced the persisted checkpoint position \
         to the end of record {CHECKPOINTED}"
    );

    for i in CHECKPOINTED..TOTAL {
        wal.append_entry(&payload_for(BOUNDARY_PAYLOAD, i))
            .unwrap_or_else(|e| panic!("append #{i} filling the ring must succeed: {e:?}"));
    }
    assert_eq!(
        wal.stats().pending_bytes,
        pending_bytes_expected(),
        "fixture: the ring must be exactly full with {PENDING} records uncheckpointed \
         ({PENDING} x {BOUNDARY_ENTRY} bytes), not {} accounted as pending",
        wal.stats().pending_bytes,
    );
}

/// A checkpoint taken while `write_head == region_size`.
///
/// Two things are at stake. `record_checkpoint` copies the cursor into
/// `header.wal_checkpoint_pos`, so unless the cursor is reduced it stores `region_size` in a
/// field whose consumers all do `region_offset + pos` and would land outside the region. And
/// `maybe_write_sentinel` runs at the end of every checkpoint: if it takes the boundary cursor
/// modulo the region it writes its 48-byte terminator on offset 0, which is the oldest live
/// record, and `scan_records` - which always starts at cursor 0 and breaks on the first
/// all-zero header - then reports an empty WAL.
#[test]
fn checkpoint_at_the_exact_fit_boundary_stays_in_region_and_keeps_every_record() {
    assert_fixture_geometry();

    let dir = TempDir::new().expect("scratch dir for the boundary checkpoint test");
    let path = store_path(&dir);
    let (mut file, mut header) = create_store(&path, BOUNDARY_REGION);
    let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");
    drive_to_boundary(&mut wal, &mut header, &mut file);

    // (1) Before anything else: the handle that wrote the run must be able to hand it back.
    // On the unfixed tree this is already the failure, because the 64th append folded the
    // cursor to 0 and `maybe_write_sentinel` terminated the chain on top of record 1.
    let before = wal
        .pending_records()
        .expect("pending_records before the boundary checkpoint");
    let missing_before = missing_sequences(first_pending_seq(), last_seq(), &before);
    assert!(
        missing_before.is_empty(),
        "(1) {}/{} of the {PENDING} records left uncheckpointed on an exactly full ring were \
         already unreachable from the writing handle, before any boundary checkpoint ran (first \
         missing {:?}, last {:?}, recovered seq {:?}..{:?}); region={BOUNDARY_REGION} \
         entry={BOUNDARY_ENTRY} payload={BOUNDARY_PAYLOAD}",
        missing_before.len(),
        PENDING,
        missing_before.first(),
        missing_before.iter().next_back(),
        sequences(&before).first(),
        sequences(&before).last(),
    );

    // The state under test: `write_head` sits at `region_size` here.
    wal.record_checkpoint(&mut header)
        .expect("record_checkpoint at the exact-fit boundary");
    assert_eq!(
        header.wal_sequence,
        last_seq(),
        "the boundary checkpoint must have recorded every one of the {TOTAL} appended records"
    );
    assert!(
        header.wal_checkpoint_pos < BOUNDARY_REGION,
        "record_checkpoint taken at the exact-fit boundary persisted \
         wal_checkpoint_pos={} for a region of {BOUNDARY_REGION}. That field is an in-region \
         offset: `region_offset + wal_checkpoint_pos` is how a consumer reaches the byte it \
         names, so an out-of-region value points one entry past the ring. The cursor is at \
         {BOUNDARY_REGION} only because record {TOTAL} ended flush with the region end \
         ({TOTAL} x {BOUNDARY_ENTRY} = {BOUNDARY_REGION}).",
        header.wal_checkpoint_pos,
    );

    wal.flush().expect("flush wal");
    HeaderCodec::write(&mut file, &header).expect("persist the boundary checkpoint header");
    file.sync_all()
        .expect("sync the boundary checkpoint header");
    drop(wal);
    drop(file);

    // (2) The same claim about the *bytes*, not the caller's copy of the struct.
    let on_disk = persisted_header(&path);
    assert!(
        on_disk.wal_checkpoint_pos < BOUNDARY_REGION,
        "(2) the header on disk encodes wal_checkpoint_pos={} for region {BOUNDARY_REGION}; a \
         reader computing `region_offset + pos` would land outside the region",
        on_disk.wal_checkpoint_pos,
    );
    assert_eq!(
        on_disk.wal_sequence,
        last_seq(),
        "(2) the header on disk must carry the checkpoint sequence {TOTAL}, got {}",
        on_disk.wal_sequence,
    );

    // (3) No terminator may have been planted on the one header the scanner reads first.
    let slot_zero = entry_header_fields(&read_entry_header(&path, 0));
    assert_eq!(
        slot_zero,
        slot_zero_expectation(),
        "(3) the boundary checkpoint planted an end-of-chain terminator inside the live span: \
         slot 0 (byte 0 of the region) reads sequence {} length {}, but it holds the oldest \
         record - sequence 1, payload {BOUNDARY_PAYLOAD}. `maybe_write_sentinel` took the \
         boundary cursor modulo the region and wrote {} zero bytes there, and `scan_records` \
         starts at cursor 0 and stops at the first all-zero header, so every record behind it \
         became invisible; region={BOUNDARY_REGION} entry={BOUNDARY_ENTRY} records={TOTAL}",
        slot_zero.0,
        slot_zero.1,
        ENTRY_HEADER_SIZE,
    );

    // (4) A fresh reader must still recover the whole ring, including the {PENDING} that were
    // uncheckpointed when the checkpoint was taken.
    let mut reader = reopen(&path, &on_disk);
    let all = reader
        .records_after(0)
        .expect("records_after(0) after the boundary checkpoint");
    let missing_all = missing_sequences(1, last_seq(), &all);
    assert!(
        missing_all.is_empty(),
        "(4) a reader after the boundary checkpoint recovers {}/{} of the {TOTAL} records that \
         are still on disk ({} missing, first {:?}, last {:?}, recovered seq {:?}..{:?}); the \
         checkpoint at `write_head == region_size` hid records rather than recording them; \
         region={BOUNDARY_REGION} entry={BOUNDARY_ENTRY}",
        TOTAL - missing_all.len(),
        TOTAL,
        missing_all.len(),
        missing_all.first(),
        missing_all.iter().next_back(),
        sequences(&all).first(),
        sequences(&all).last(),
    );
    let missing_pending = missing_sequences(first_pending_seq(), last_seq(), &all);
    assert!(
        missing_pending.is_empty(),
        "(4) specifically, {}/{} of the records that were uncheckpointed when the boundary \
         checkpoint was taken are unrecoverable (first missing {:?}, last {:?}); those are the \
         records a sentinel on offset 0 hides first, because they sit at the top of the ring",
        missing_pending.len(),
        PENDING,
        missing_pending.first(),
        missing_pending.iter().next_back(),
    );

    // (5) And the checkpoint must be honoured: nothing left to replay.
    let replayed = reader
        .pending_records()
        .expect("pending_records after reopen at the boundary");
    assert!(
        replayed.is_empty(),
        "(5) the boundary checkpoint was not honoured: a fresh handle still replays {} records \
         (seq {:?}..{:?}) although the persisted checkpoint sequence {} covers all {TOTAL}; \
         region={BOUNDARY_REGION} entry={BOUNDARY_ENTRY}",
        replayed.len(),
        sequences(&replayed).first(),
        sequences(&replayed).last(),
        on_disk.wal_sequence,
    );
    assert_eq!(
        reader.stats().pending_bytes,
        0,
        "(5) a reopened handle must account nothing as pending once the ring is checkpointed, \
         got {} bytes",
        reader.stats().pending_bytes,
    );

    // (6) `records_after` re-runs `initialise_sentinel` on the way out; it must still not have
    // written a terminator at offset 0.
    drop(replayed);
    drop(all);
    drop(reader);
    let slot_zero_after = entry_header_fields(&read_entry_header(&path, 0));
    assert_eq!(
        slot_zero_after,
        slot_zero_expectation(),
        "(6) the read path wrote a terminator at offset 0: slot 0 reads sequence {} length {:?} \
         instead of sequence 1 with payload {BOUNDARY_PAYLOAD}. `open_internal`/`records_after` \
         folded the scan cursor down from the region end to 0 and `initialise_sentinel` then \
         zeroed the header there; region={BOUNDARY_REGION} entry={BOUNDARY_ENTRY}",
        slot_zero_after.0,
        slot_zero_after.1,
    );
}

/// Reopening an exactly full ring that still has uncheckpointed records must not write a
/// terminator at offset 0.
///
/// `scan_records` returns the cursor at the region end when the ring is exactly full. If the
/// reopen path folds that to 0, `EmbeddedWal::open` calls `initialise_sentinel` with
/// `write_head == 0`, which writes a 48-byte all-zero header on offset 0 - the oldest live
/// record - and the scan from cursor 0 then stops immediately. Nothing distinguishes that from
/// a healthy WAL, because every append before it returned `Ok`.
#[test]
fn reopen_at_the_exact_fit_boundary_writes_no_terminator_at_offset_zero() {
    assert_fixture_geometry();

    let dir = TempDir::new().expect("scratch dir for the boundary reopen test");
    let path = store_path(&dir);
    let (mut file, mut header) = create_store(&path, BOUNDARY_REGION);
    let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");
    drive_to_boundary(&mut wal, &mut header, &mut file);
    // Deliberately no checkpoint here: {PENDING} records are still uncheckpointed and the
    // cursor is sitting on the boundary.
    wal.flush().expect("flush wal");
    drop(wal);
    drop(file);

    // (1) The writer itself must not have terminated the chain on its way out.
    let slot_zero_before = entry_header_fields(&read_entry_header(&path, 0));
    assert_eq!(
        slot_zero_before,
        slot_zero_expectation(),
        "(1) closing an exactly full ring planted a terminator at offset 0: slot 0 reads \
         sequence {} length {:?} instead of sequence 1 with payload {BOUNDARY_PAYLOAD}. The \
         64th append put the cursor at 0 rather than at the region end, so the \
         `maybe_write_sentinel` call at the end of `append_entry` wrote {} zero bytes over the \
         oldest record; region={BOUNDARY_REGION} entry={BOUNDARY_ENTRY} records={TOTAL}",
        slot_zero_before.0,
        slot_zero_before.1,
        ENTRY_HEADER_SIZE,
    );

    // (2) Reopen: the scan cursor arrives at the region end, and `initialise_sentinel` must
    // leave the region alone.
    let mut reader = reopen(&path, &header);
    let recovered = reader
        .pending_records()
        .expect("pending_records after reopening an exactly full ring");
    let missing = missing_sequences(first_pending_seq(), last_seq(), &recovered);
    assert!(
        missing.is_empty(),
        "(2) reopening an exactly full ring hid {}/{} uncheckpointed records ({} missing, \
         first {:?}, last {:?}, recovered seq {:?}..{:?}); `open_internal` folds the scan cursor \
         modulo the region and `initialise_sentinel` writes its terminator wherever the cursor \
         lands, and a scan starting at cursor 0 stops at the first all-zero header; \
         region={BOUNDARY_REGION} entry={BOUNDARY_ENTRY} payload={BOUNDARY_PAYLOAD}",
        missing.len(),
        PENDING,
        missing.len(),
        missing.first(),
        missing.iter().next_back(),
        sequences(&recovered).first(),
        sequences(&recovered).last(),
    );
    assert_eq!(
        sequences(&recovered),
        (first_pending_seq()..=last_seq()).collect::<Vec<u64>>(),
        "(2) the reopened handle must hand back sequences {}..={TOTAL} in order, got {:?}",
        first_pending_seq(),
        sequences(&recovered),
    );

    // (3) Its accounting must agree with what it hands back.
    assert_eq!(
        reader.stats().pending_bytes,
        pending_bytes_expected(),
        "(3) the reopened handle must account {PENDING} x {BOUNDARY_ENTRY} = {} bytes as \
         pending on an exactly full ring, got {} - a handle that cannot see the records must \
         not claim them, and one that claims them must return them",
        pending_bytes_expected(),
        reader.stats().pending_bytes,
    );

    // (4) And offset 0 must still be the oldest record once the reopen path has run.
    drop(recovered);
    drop(reader);
    let slot_zero_after = entry_header_fields(&read_entry_header(&path, 0));
    assert_eq!(
        slot_zero_after,
        slot_zero_expectation(),
        "(4) `open_internal` wrote a terminator at offset 0 while reopening an exactly full \
         ring: slot 0 reads sequence {} length {:?} instead of sequence 1 with payload \
         {BOUNDARY_PAYLOAD}, which truncates every scan that starts at cursor 0; \
         region={BOUNDARY_REGION} entry={BOUNDARY_ENTRY}",
        slot_zero_after.0,
        slot_zero_after.1,
    );
}
