//! Regression tests for replay segment persistence/integrity.
//!
//! The replay segment is written to the `.mv2` file when ending a recording session.
//! Historically this could corrupt Tantivy (lexical) segments by writing replay bytes
//! at an incorrect offset and moving the TOC footer backwards.

#[cfg(all(feature = "lex", feature = "replay"))]
use memvid_core::{Memvid, PutOptions, SearchRequest};
// Ungated on purpose: the WAL tests below are themselves ungated so they run under the
// default feature set that CI builds, and each one owns a `TempDir` for its scratch store.
use tempfile::TempDir;

#[test]
#[cfg(all(feature = "lex", feature = "replay"))]
fn replay_save_does_not_corrupt_lex_index() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("replay_integrity.mv2");

    // Create a memory with a Tantivy-backed lexical index.
    {
        let mut mem = Memvid::create(&path).unwrap();
        mem.enable_lex().unwrap();

        let opts = PutOptions {
            uri: Some("mv2://doc/0".to_string()),
            title: Some("Doc".to_string()),
            search_text: Some("Climate change and sustainability are important.".to_string()),
            ..Default::default()
        };
        mem.put_bytes_with_options(b"climate", opts).unwrap();
        mem.commit().unwrap();

        // Record a session that includes a FIND action.
        mem.start_session(Some("Test".to_string()), None).unwrap();
        let _ = mem
            .search(SearchRequest {
                query: "climate".to_string(),
                top_k: 5,
                snippet_chars: 120,
                uri: None,
                scope: None,
                cursor: None,
                #[cfg(feature = "temporal_track")]
                temporal: None,
                as_of_frame: None,
                as_of_ts: None,
                no_sketch: false,
                acl_context: None,
                acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
            })
            .unwrap();
        mem.end_session().unwrap();

        // Mirror CLI behaviour: finalize, save replay, then commit again.
        mem.commit().unwrap();
        mem.save_replay_sessions().unwrap();
        mem.commit().unwrap();
    }

    // Reopen and ensure Tantivy loads and lexical search still works.
    let mut reopened = Memvid::open_read_only(&path).unwrap();
    let results = reopened
        .search(SearchRequest {
            query: "climate".to_string(),
            top_k: 5,
            snippet_chars: 120,
            uri: None,
            scope: None,
            cursor: None,
            #[cfg(feature = "temporal_track")]
            temporal: None,
            as_of_frame: None,
            as_of_ts: None,
            no_sketch: false,
            acl_context: None,
            acl_enforcement_mode: memvid_core::types::AclEnforcementMode::Audit,
        })
        .unwrap();

    assert!(
        !results.hits.is_empty(),
        "expected lexical search to work after saving replay sessions"
    );

    // Also ensure replay sessions can be loaded (read-only is fine).
    reopened.load_replay_sessions().unwrap();
    let sessions = reopened.list_sessions();
    assert_eq!(sessions.len(), 1, "expected one recorded replay session");
}

// ---------------------------------------------------------------------------
// Embedded WAL replay integrity (D4).
//
// The regression test above is gated on `all(feature = "lex", feature =
// "replay")`, so it compiles to zero tests under the default feature set that
// CI builds. The tests below need no features: they drive `EmbeddedWal`
// directly and assert a recovery invariant - every record appended since the
// last checkpoint is handed back - rather than the absence of an error,
// because on the unfixed tree every append returns `Ok` while records are
// already unreachable.
//
// Two independent triggers:
//
// `pending_records()` is called here from the checkpoint decision point, which is what
// the production caller does (`Memvid::commit`/flush call it at mutation.rs:761, :857 and
// :1020; `doctor` and `maintenance` also call it). The loss is therefore not an artefact
// of a diagnostic probe: the same configuration loses records through the public
// `Memvid` API (51 of 1001 with unique uris, 18 of 1001 with a shared uri), where no WAL
// internals are touched at all.
//
// * `sentinel_at_region_end_hides_uncheckpointed_records` drives
//   `EmbeddedWal::write_zero_header` into its `remaining < ENTRY_HEADER_SIZE`
//   branch, where it zeroes the tail, rewinds `pos` to 0, zeroes the header at
//   offset 0 and returns that 0, which `maybe_write_sentinel` assigns to
//   `write_head`.
// * `exact_divisor_record_size_loses_records` uses a record size that divides
//   the region exactly, so `append_entry` wraps to offset 0 without ever
//   tripping the protective "region full" error. The grow-by-doubling policy
//   `Memvid` uses is modelled here, but the measured failure happens with
//   `grows = 0`, i.e. before any growth.
// ---------------------------------------------------------------------------

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memvid_core::constants::{MAGIC, SPEC_VERSION, WAL_OFFSET};
use memvid_core::error::MemvidError;
use memvid_core::io::header::HeaderCodec;
use memvid_core::io::wal::{EmbeddedWal, WalRecord};
use memvid_core::types::Header;

const ENTRY_HEADER_SIZE: u64 = 48;

/// Non-zero, varying bytes: a real WAL record never contains a long zero run at a record
/// boundary, and we do not want a false "end of chain" sentinel.
fn payload_for(len: usize, i: usize) -> Vec<u8> {
    let mut p = vec![0u8; len];
    for (k, b) in p.iter_mut().enumerate() {
        *b = ((i.wrapping_mul(31).wrapping_add(k.wrapping_mul(7))) % 250 + 1) as u8;
    }
    p
}

/// Path of a test's scratch store inside the `TempDir` that test owns.
///
/// The directory, not the file, is the unit of cleanup: `TempDir::drop` removes it when the
/// test leaves scope, *including through a panicking assertion*. The previous scheme could not
/// promise that - it put the file directly in `env::temp_dir()` under a name built from the
/// tag and the pid, and deleted it with an inline `remove_file` sitting before the final
/// assertion. The first assertion to fire therefore orphaned the file, and nothing could ever
/// revisit the orphan because the pid baked into its name had died with the run.
fn store_path(dir: &TempDir) -> PathBuf {
    dir.path().join("wal-store.mv2")
}

fn create_store(path: &Path, region: u64) -> (File, Header) {
    let _ = std::fs::remove_file(path);
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .expect("open scratch store");
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

/// Reopen through a *fresh* handle so the assertions describe what a reader recovers from
/// the bytes on disk, not what an in-memory object happens to remember.
fn reopen(path: &Path, header: &Header) -> EmbeddedWal {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("reopen scratch store");
    EmbeddedWal::open(&file, header).expect("reopen wal")
}

fn missing_sequences(want_from: u64, want_upto: u64, records: &[WalRecord]) -> Vec<u64> {
    let present: BTreeSet<u64> = records.iter().map(|r| r.sequence).collect();
    (want_from..=want_upto)
        .filter(|s| !present.contains(s))
        .collect()
}

/// The region holds `TRIGGER_A_RECORDS` records of `TRIGGER_A_ENTRY` bytes and is then `20` bytes
/// short of a full record header, which is exactly the condition `write_zero_header` handles by
/// rewinding `pos` to 0 and falling through to write a 48-byte zero header there.
///
/// Measured with `examples/d4_slots.rs` on the pristine tree: the destruction happens on the **last
/// append** (`maybe_write_sentinel` is called from `append_entry`, and `open_internal` calls the
/// same code via `initialise_sentinel`), so slot 0 already reads `ZERO_HEADER` before any reopen
/// while the other 161 records are byte-for-byte intact. The frames are therefore *hidden* behind a
/// terminator the scanner (`scan_records`, which starts at cursor 0 and breaks on a zero header)
/// will never cross, not overwritten.
const TRIGGER_A_PAYLOAD: usize = 756;
const TRIGGER_A_ENTRY: u64 = ENTRY_HEADER_SIZE + TRIGGER_A_PAYLOAD as u64; // 804
const TRIGGER_A_RECORDS: usize = 162; // 162 * 804 = 130_248
const TRIGGER_A_TAIL: u64 = 20;
const TRIGGER_A_REGION: u64 = TRIGGER_A_ENTRY * TRIGGER_A_RECORDS as u64 + TRIGGER_A_TAIL; // 130_268

/// The region is deliberately a few bytes short of holding one more record header. Computed
/// through a local binding so the fixture property is exercised at runtime instead of being a
/// constant the compiler folds away.
fn tail_below_sentinel() -> bool {
    let tail = TRIGGER_A_REGION - TRIGGER_A_ENTRY * TRIGGER_A_RECORDS as u64;
    tail < ENTRY_HEADER_SIZE
}

#[test]
fn sentinel_at_region_end_hides_uncheckpointed_records() {
    assert_eq!(
        TRIGGER_A_ENTRY * TRIGGER_A_RECORDS as u64 + TRIGGER_A_TAIL,
        TRIGGER_A_REGION,
        "fixture must leave exactly TRIGGER_A_TAIL bytes in the region"
    );
    assert!(
        tail_below_sentinel(),
        "the leftover tail must be smaller than one record header"
    );

    let dir = TempDir::new().expect("scratch dir for trigger A");
    let path = store_path(&dir);
    let (file, header) = create_store(&path, TRIGGER_A_REGION);
    let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");

    for i in 0..TRIGGER_A_RECORDS {
        wal.append_entry(&payload_for(TRIGGER_A_PAYLOAD, i))
            .unwrap_or_else(|e| panic!("append #{i} must succeed on a fresh region: {e:?}"));
    }
    assert_eq!(
        usize::try_from(wal.stats().sequence).expect("sequence fits usize"),
        TRIGGER_A_RECORDS,
        "every append reported Ok, so the sequence must have advanced"
    );
    let pending_bytes = wal.stats().pending_bytes;
    assert_eq!(
        pending_bytes,
        TRIGGER_A_ENTRY * TRIGGER_A_RECORDS as u64,
        "nothing was checkpointed, so all {TRIGGER_A_RECORDS} records are still pending"
    );
    wal.flush().expect("flush wal");
    drop(wal);
    drop(file);

    let mut reader = reopen(&path, &header);
    let recovered = reader
        .pending_records()
        .expect("pending_records after reopen");
    drop(reader);
    let missing = missing_sequences(1, TRIGGER_A_RECORDS as u64, &recovered);

    assert!(
        missing.is_empty(),
        "{} of {TRIGGER_A_RECORDS} uncheckpointed records are unrecoverable after reopen: \
         payload_len={TRIGGER_A_PAYLOAD}, entry={TRIGGER_A_ENTRY}, region={TRIGGER_A_REGION}, \
         write_head sits {} bytes from the region end which is below the {ENTRY_HEADER_SIZE}-byte \
         sentinel size, pending_bytes at close = {pending_bytes}; \
         recovered {} records (seq {:?}..{:?}); first missing seq {:?}, last {:?}",
        missing.len(),
        TRIGGER_A_REGION - TRIGGER_A_ENTRY * TRIGGER_A_RECORDS as u64,
        recovered.len(),
        recovered.first().map(|r| r.sequence),
        recovered.last().map(|r| r.sequence),
        missing.first(),
        missing.iter().next_back(),
    );
}

/// The second, independent lie: `pending_bytes` keeps counting records the scan can no
/// longer see, so `should_checkpoint()` under-reports occupancy and the caller never
/// forces a checkpoint in time. The WAL's own accounting must agree with what it hands back.
#[test]
fn pending_byte_accounting_agrees_with_recoverable_records() {
    let dir = TempDir::new().expect("scratch dir for the pending-byte accounting test");
    let path = store_path(&dir);
    let (file, header) = create_store(&path, TRIGGER_A_REGION);
    let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");
    for i in 0..TRIGGER_A_RECORDS {
        wal.append_entry(&payload_for(TRIGGER_A_PAYLOAD, i))
            .expect("append");
    }
    let pending_bytes = wal.stats().pending_bytes;
    wal.flush().expect("flush");
    let in_memory = wal
        .pending_records()
        .expect("pending_records in memory")
        .len();
    drop(wal);
    drop(file);

    let mut reader = reopen(&path, &header);
    let on_disk = reader
        .pending_records()
        .expect("pending_records after reopen")
        .len();
    drop(reader);

    let claimed = usize::try_from(pending_bytes / TRIGGER_A_ENTRY).expect("fits usize");
    assert_eq!(
        claimed, on_disk,
        "the WAL claims {claimed} pending records from pending_bytes={pending_bytes} \
         ({pending_bytes} bytes / {TRIGGER_A_ENTRY} bytes per record) but hands back \
         {on_disk} after reopen; in-memory pending_records() returned {in_memory}"
    );
}

/// The policy `Memvid` applies when the embedded WAL reports itself full: double the
/// region, relocate the existing bytes, re-zero the new tail, persist the header, reopen.
fn grow_region(file: &mut File, header: &mut Header, wal: &mut EmbeddedWal) {
    let mut new_size = header.wal_size;
    let target = header.wal_size + 1;
    while new_size <= target {
        new_size *= 2;
    }
    let delta = new_size - header.wal_size;
    let original_len = file.metadata().expect("metadata").len();
    let data_start = header.wal_offset + header.wal_size;

    file.set_len(original_len + delta).expect("extend file");
    let mut remaining = original_len.saturating_sub(data_start);
    let mut buf = vec![0u8; 64 * 1024];
    while remaining > 0 {
        let chunk = remaining.min(buf.len() as u64);
        let src = data_start + remaining - chunk;
        file.seek(SeekFrom::Start(src)).expect("seek src");
        file.read_exact(&mut buf[..chunk as usize])
            .expect("read src");
        file.seek(SeekFrom::Start(src + delta)).expect("seek dst");
        file.write_all(&buf[..chunk as usize]).expect("write dst");
        remaining -= chunk;
    }
    file.seek(SeekFrom::Start(data_start)).expect("seek tail");
    let zeros = vec![0u8; 64 * 1024];
    let mut rem = delta;
    while rem > 0 {
        let w = rem.min(zeros.len() as u64) as usize;
        file.write_all(&zeros[..w]).expect("zero tail");
        rem -= w as u64;
    }
    header.wal_size = new_size;
    header.footer_offset += delta;
    HeaderCodec::write(&mut *file, header).expect("persist grown header");
    file.sync_all().expect("sync grow");
    *wal = EmbeddedWal::open(&*file, header).expect("reopen after grow");
}

/// Record size divides the region exactly, so the ring cursor wraps to 0 on a legitimate
/// append. Exact arithmetic: `entry_size = ENTRY_HEADER_SIZE + payload = 48 + 976 = 1024`
/// (`wal.rs:111`), `region_size = 65_536 = 64 * 1024` (`:10`), i.e. **64 records per lap** and
/// `65_536 % 1_024 == 0`. Note `976` is the *payload*; as an entry size it would not divide the
/// region (`65_536 % 976 = 144`) and the geometry would exercise nothing.
///
/// This trigger is **not reachable from a clean ring**. With `pending_bytes` accounting for every
/// live byte, the 65th append is refused by the `pending_bytes + entry_size > region_size` guard
/// at `wal.rs:117` - the designed "region full" growth path. The loss needs `pending_bytes` to
/// under-report first, which the WAL's own checkpoint policy does at 75% occupancy
/// (`should_checkpoint` at `:158-165`, `WAL_CHECKPOINT_THRESHOLD` in `constants.rs:35`):
/// `record_checkpoint` (`:167-176`) sets `pending_bytes = 0` and moves `checkpoint_head` while
/// `write_head` keeps its ring position. That is why this test checkpoints mid-stream - the same
/// call sequence `Memvid::commit` makes - and why it asserts the checkpoint actually happened.
/// A test of this shape that is green before the fix is testing the `:117` guard, not trigger B.
///
/// `grows` is reported, not assumed: the loss appears with `grows = 0`, i.e. with no region
/// doubling and no relocation, so no growth machinery is involved. A fix limited to the
/// `remaining < ENTRY_HEADER_SIZE` branch does not repair this.
///
/// Also measured with `examples/d4_slots.rs`: after the loss every one of the 64 slots still holds
/// its record with a matching payload - slot 32 carries a legitimate end-of-chain sentinel and the
/// 16 "lost" records (`seq 49..=64`) sit intact at slots 48..=63. Because `scan_records` walks from
/// cursor 0 and stops at that sentinel, the loss is visibility, not destruction, which is what makes
/// the write-side fix sufficient rather than a recovery exercise.
const TRIGGER_B_PAYLOAD: usize = 976;
const TRIGGER_B_ENTRY: u64 = ENTRY_HEADER_SIZE + TRIGGER_B_PAYLOAD as u64; // 1_024
const TRIGGER_B_REGION: u64 = 64 * TRIGGER_B_ENTRY; // 65_536, an exact multiple
const TRIGGER_B_RECORDS: usize = 1200;

#[test]
fn exact_divisor_entry_loses_records_after_checkpoint_resets_pending_bytes() {
    assert_eq!(
        TRIGGER_B_REGION % TRIGGER_B_ENTRY,
        0,
        "fixture must make the *entry* size (48 + payload = {TRIGGER_B_ENTRY}) divide the region exactly"
    );
    assert_eq!(
        usize::try_from(TRIGGER_B_REGION / TRIGGER_B_ENTRY).expect("fits usize"),
        64,
        "the ring must hold exactly 64 records per lap"
    );

    let dir = TempDir::new().expect("scratch dir for trigger B");
    let path = store_path(&dir);
    let (mut file, mut header) = create_store(&path, TRIGGER_B_REGION);
    let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");

    let mut appended_since = 0usize;
    let mut grows = 0usize;
    let mut first_break: Option<String> = None;

    for i in 0..TRIGGER_B_RECORDS {
        let p = payload_for(TRIGGER_B_PAYLOAD, i);
        loop {
            match wal.append_entry(&p) {
                Ok(_) => break,
                Err(MemvidError::CheckpointFailed { ref reason })
                    if reason.as_str() == "embedded WAL region full"
                        || reason.as_str() == "embedded WAL region too small for entry" =>
                {
                    grow_region(&mut file, &mut header, &mut wal);
                    grows += 1;
                }
                Err(e) => panic!("append #{i} failed unexpectedly: {e:?}"),
            }
        }
        appended_since += 1;

        if wal.should_checkpoint() {
            // Same call `Memvid` makes on its way to a checkpoint.
            let recs = wal.pending_records().expect("pending_records");
            if recs.len() != appended_since && first_break.is_none() {
                let missing =
                    missing_sequences(header.wal_sequence + 1, wal.stats().sequence, &recs);
                first_break = Some(format!(
                    "at append #{i}: {appended_since} records appended since the last checkpoint, \
                     pending_records() returned {} (seq {:?}..{:?}), {} sequences in ({}, {}] are \
                     unrecoverable; region={} entry={TRIGGER_B_ENTRY} payload={TRIGGER_B_PAYLOAD} \
                     (divides region exactly: {}), grows={grows}, pending_bytes={}, \
                     appends_since_checkpoint={}",
                    recs.len(),
                    recs.first().map(|r| r.sequence),
                    recs.last().map(|r| r.sequence),
                    missing.len(),
                    header.wal_sequence,
                    wal.stats().sequence,
                    TRIGGER_B_REGION,
                    TRIGGER_B_REGION % TRIGGER_B_ENTRY == 0,
                    wal.stats().pending_bytes,
                    wal.stats().appends_since_checkpoint,
                ));
            }
            if !hold_checkpoint_for_diagnosis() {
                wal.record_checkpoint(&mut header)
                    .expect("record_checkpoint");
                HeaderCodec::write(&mut file, &header).expect("persist header");
                file.sync_all().expect("sync checkpoint");
                appended_since = 0;
            }
        }
    }

    if let Some(msg) = first_break {
        assert!(
            header.wal_sequence > 0,
            "trigger B needs the checkpoint policy to have reset pending_bytes; no checkpoint \
             ran, so this failure would be the :117 region-full guard rather than trigger B"
        );
        assert_eq!(
            grows, 0,
            "no growth may be needed for this repro to be meaningful"
        );
        panic!("silent record loss in the exact-divisor configuration: {msg}");
    }
}

/// `D4_HOLD_CHECKPOINT=1` leaves every round uncheckpointed, which turns a visibility bug
/// into an accumulating one. Investigation aid only; never set in CI.
fn hold_checkpoint_for_diagnosis() -> bool {
    std::env::var_os("D4_HOLD_CHECKPOINT").is_some()
}

/// Write a raw 48-byte entry header straight into a slot, bypassing `append_entry`, so a test can
/// put the scanner in front of bytes the writer would never produce.
fn write_raw_header(file: &mut File, slot: u64, sequence: u64, length: u32) {
    let mut header = [0u8; 48];
    header[..8].copy_from_slice(&sequence.to_le_bytes());
    header[8..12].copy_from_slice(&length.to_le_bytes());
    let offset = WAL_OFFSET + slot * u64::from(length) + slot * 48;
    file.seek(SeekFrom::Start(offset)).expect("seek raw header");
    file.write_all(&header).expect("write raw header");
    file.sync_all().expect("sync raw header");
}

/// The sentinel fixes make an in-band zero terminator *rarer*; they must not turn the scanner into
/// something that tolerates junk. `scan_records` raises `WalCorruption` when a length is zero or a
/// record would overrun the region, and a "skip zero runs" refactor is exactly how that guard gets
/// traded away for something that looks like an improvement. Pinned here from both sides.
#[test]
fn bogus_record_length_is_still_reported_as_corruption() {
    let dir = TempDir::new().expect("scratch dir for the bogus-length test");
    let path = store_path(&dir);
    let (mut file, header) = create_store(&path, TRIGGER_B_REGION);

    // A length that pushes the record past the end of the region.
    write_raw_header(
        &mut file,
        0,
        1,
        u32::try_from(TRIGGER_B_REGION).expect("fits"),
    );
    let err = EmbeddedWal::open(&file, &header).expect_err("an over-running record must not open");
    assert!(
        matches!(err, MemvidError::WalCorruption { .. }),
        "expected WalCorruption for an over-running record length, got {err:?}"
    );

    // A zero length with a live sequence is junk too, not an end-of-log marker.
    let (mut file, header) = create_store(&path, TRIGGER_B_REGION);
    write_raw_header(&mut file, 0, 7, 0);
    let err = EmbeddedWal::open(&file, &header).expect_err("a zero length must not open");
    assert!(
        matches!(err, MemvidError::WalCorruption { .. }),
        "expected WalCorruption for a zero record length, got {err:?}"
    );
}

/// The other side of the same coin: a genuine end-of-log zero header must terminate the chain
/// silently rather than raise, otherwise "make corruption loud again" is paid for by refusing to
/// open every healthy WAL.
#[test]
fn zero_terminator_ends_the_chain_without_error() {
    let dir = TempDir::new().expect("scratch dir for the zero-terminator test");
    let path = store_path(&dir);
    let (file, header) = create_store(&path, TRIGGER_B_REGION);
    {
        let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");
        wal.append_entry(&payload_for(TRIGGER_B_PAYLOAD, 0))
            .expect("append");
        wal.flush().expect("flush");
    }

    let mut reader = reopen(&path, &header);
    let records = reader.pending_records().expect("scan must succeed");
    assert_eq!(
        records.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        vec![1],
        "one appended record must be recoverable across a zero terminator"
    );
    drop(reader);
}

/// The reopen path carries its **own** copy of the boundary collapse. `scan_records` returns the
/// cursor at the region end when the ring is exactly full, and `open_internal` folded that with
/// `% region_size` down to 0, so the first write after reopen lands on offset 0 - inside the live
/// span, because the 75% checkpoint had already reset `pending_bytes` while `write_head` kept its
/// ring position. `EmbeddedWal::open` then calls `initialise_sentinel`, which is what physically
/// writes the zero header there. No test in the lib suite reopens at the exact-fill point, so
/// without this test nothing distinguishes that hunk from inert decoration.
///
/// Shape: `entry_size = 1024`, `region_size = 65_536` (64 slots). Append 48 records, checkpoint,
/// append 16 more so the ring is exactly full with 16 records still uncheckpointed. Then:
/// (1) the in-memory handle must hand those 16 back, (2) a *fresh* handle must hand back the same
/// 16, and (3) one further append must either succeed without destroying anything or raise the
/// designed `region full` error - never silently clobber.
const EXACT_FILL_CHECKPOINTED: usize = 48;
const EXACT_FILL_PENDING: usize = 16;

#[test]
fn reopen_at_exact_fill_preserves_pending_records() {
    assert_eq!(
        EXACT_FILL_CHECKPOINTED + EXACT_FILL_PENDING,
        usize::try_from(TRIGGER_B_REGION / TRIGGER_B_ENTRY).expect("fits usize"),
        "the fixture must fill the ring exactly: 48 checkpointed + 16 pending slots"
    );

    let dir = TempDir::new().expect("scratch dir for the exact-fill reopen test");
    let path = store_path(&dir);
    let (mut file, mut header) = create_store(&path, TRIGGER_B_REGION);
    let mut wal = EmbeddedWal::open(&file, &header).expect("open wal");

    for i in 0..EXACT_FILL_CHECKPOINTED {
        wal.append_entry(&payload_for(TRIGGER_B_PAYLOAD, i))
            .unwrap_or_else(|e| panic!("append #{i} on an empty ring must succeed: {e:?}"));
    }
    assert!(
        wal.should_checkpoint(),
        "the 75% occupancy rule (`WAL_CHECKPOINT_THRESHOLD`) must fire at record \
         {EXACT_FILL_CHECKPOINTED}; without that checkpoint the ring is clean and :117 refuses \
         further appends instead of losing anything"
    );
    wal.record_checkpoint(&mut header)
        .expect("record_checkpoint");
    HeaderCodec::write(&mut file, &header).expect("persist checkpoint");
    file.sync_all().expect("sync checkpoint");

    let pending_start = EXACT_FILL_CHECKPOINTED;
    let pending_end = EXACT_FILL_CHECKPOINTED + EXACT_FILL_PENDING;
    for i in pending_start..pending_end {
        wal.append_entry(&payload_for(TRIGGER_B_PAYLOAD, i))
            .unwrap_or_else(|e| panic!("append #{i} filling the ring must succeed: {e:?}"));
    }

    let want: Vec<u64> = ((pending_start + 1)..=pending_end)
        .map(|i| i as u64)
        .collect();
    let sequences =
        |records: &[WalRecord]| -> Vec<u64> { records.iter().map(|r| r.sequence).collect() };

    let expected_pending = u64::try_from(EXACT_FILL_PENDING).expect("fits") * TRIGGER_B_ENTRY;
    assert_eq!(
        wal.stats().pending_bytes,
        expected_pending,
        "16 uncheckpointed records must still be accounted for on an exactly full ring"
    );
    let in_memory = wal.pending_records().expect("pending in memory");
    assert_eq!(
        sequences(&in_memory),
        want,
        "(1) the in-memory handle must hand back the {EXACT_FILL_PENDING} uncheckpointed records \
         on an exactly full ring; the ring cursor wrapped to 0 at the last append, so \
         `maybe_write_sentinel` wrote a zero header on offset 0 and truncated the scan"
    );
    drop(in_memory);
    drop(wal);

    let mut reader = reopen(&path, &header);
    let after_reopen = reader.pending_records().expect("pending after reopen");
    assert_eq!(
        sequences(&after_reopen),
        want,
        "(2) a fresh handle at the exact-fill boundary must hand back the same \
         {EXACT_FILL_PENDING} uncheckpointed records; `scan_records` returns the cursor at the \
         region end and `open_internal` folded it with `% region_size` to 0, then \
         `initialise_sentinel` wrote a zero header there"
    );
    drop(after_reopen);

    // (3) The next append must either be refused by the designed growth guard or succeed without
    // destroying a single still-recoverable record.
    let mut writer = reopen(&path, &header);
    let outcome = match writer.append_entry(&payload_for(TRIGGER_B_PAYLOAD, pending_end)) {
        Ok(sequence) => format!("appended sequence {sequence}"),
        Err(MemvidError::CheckpointFailed { ref reason })
            if reason.contains("region full") || reason.contains("too small") =>
        {
            format!("refused with {reason:?} (designed growth path)")
        }
        Err(e) => panic!("append after exact fill failed unexpectedly: {e:?}"),
    };
    writer.flush().expect("flush");
    drop(writer);

    let mut checker = reopen(&path, &header);
    let still = checker
        .pending_records()
        .expect("pending after append attempt");
    let missing = missing_sequences(
        u64::try_from(pending_start + 1).expect("fits"),
        u64::try_from(pending_end).expect("fits"),
        &still,
    );
    drop(still);
    drop(checker);

    assert!(
        missing.is_empty(),
        "(3) append after exact fill ({outcome}) destroyed {} of the {EXACT_FILL_PENDING} \
         uncheckpointed records (first missing {:?}, last {:?}); region={TRIGGER_B_REGION} \
         entry={TRIGGER_B_ENTRY} payload={TRIGGER_B_PAYLOAD}",
        missing.len(),
        missing.first(),
        missing.iter().next_back(),
    );
}
