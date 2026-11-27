//! Read-side contract for [`Memvid::lex_engine_report`].
//!
//! The report exists so a caller that serves queries can decide whether an
//! embedded lexical index is trustworthy **without opening it** — the same decision
//! `verify_lex_index` makes before it will let a commit through. Two
//! independent rules for one defect is how the two drift apart, so what these tests
//! pin is not the shape of the report but its *agreement* with the gate: on every
//! state the consumer's conjunction can speak to, the gate refuses exactly when the
//! rule derived from the report refuses.
//!
//! The consumer's rule is `gate_refuses` below, restated from `require_lexical_leg`
//! in the `memvid-kb-rs` service. It is a conjunction, and the accessor's most
//! important property is that everything outside it reads as *no opinion*: the
//! frame count is withheld rather than computed, and a withheld or failed count is
//! `None`, never `Some(0)` — because `None` is what the consumer abstains on.
//!
//! Agreement is total except for one state, which `assert_gate_agrees` cannot be used
//! on: a store whose engine is standing and non-empty while the catalog claims a
//! different count. The gate refuses that, the consumer abstains, and
//! `lex_engine_report_diverges_from_the_gate_on_a_diverged_count_by_design` pins why
//! that is the intended reading rather than a hole.
//!
//! The last three tests are about cost rather than agreement. The frame pass clones
//! the frame table and extracts a payload, and the accessor runs it on a state that
//! is permanently true for image-only stores, so it may run at most once per memo key
//! — `(generation, toc.toc_checksum)` — and a pass that fails has to be remembered as
//! a failure. `passes`
//! counts fresh passes, which makes "that second read answered from the memo" a
//! measurement rather than a statement of intent.

use crate::search::TantivyEngine;
use crate::types::{DocMetadata, LexIndexManifest};
use crate::{LexEngineReport, Memvid, PutOptions};
use tempfile::TempDir;

/// Frames carry a token this distinctive so a hit can only have come from the
/// frame this test wrote.
fn text_options(uri: &str, token: &str) -> PutOptions {
    PutOptions {
        uri: Some(uri.to_owned()),
        title: Some(format!("frame {token}")),
        search_text: Some(format!("the {token} lexical marker for report testing")),
        ..Default::default()
    }
}

/// A frame that declares no text at all and a content type nothing tokenises:
/// the closest thing to "opaque" a store can hold.
fn opaque_options(uri: &str) -> PutOptions {
    PutOptions {
        uri: Some(uri.to_owned()),
        metadata: Some(DocMetadata {
            mime: Some("application/octet-stream".to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// The consumer's rule, restated: an engine that is standing and advertising
/// segments, holds nothing, while frames that belong in it do.
///
/// `unwrap_or(0)` is the whole contract — a withheld or failed frame pass abstains.
/// It is safe only because the accessor never substitutes a computed zero for a
/// pass it did not run, which is what `lex_engine_report_never_fabricates_a_document_claim`
/// pins.
fn gate_refuses(report: LexEngineReport) -> bool {
    report.loaded
        && report.segments_advertised
        && report.docs == 0
        && report.indexable_frames.unwrap_or(0) > 0
}

/// The load-bearing assertion of this module: the report a reader sees and the
/// gate a writer faces make the same call about the same store.
fn assert_gate_agrees(mem: &mut Memvid, state: &str) -> LexEngineReport {
    let report = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    let refused = mem.verify_lex_index().is_err();
    assert_eq!(
        refused,
        gate_refuses(report),
        "state {state}: the commit gate refused={refused} but the report {report:?} \
         drives the consumer to refuse={}",
        gate_refuses(report),
    );
    report
}

/// The store the defect produces: the catalog still advertises its segments and
/// the engine behind them holds nothing. Injected directly, because with the trust
/// rule fixed the open path rebuilds such an index instead of reaching this state.
fn emptied_engine(mem: &mut Memvid) {
    mem.tantivy = Some(TantivyEngine::create().unwrap());
    mem.tantivy_dirty = false;
}

fn text_store(path: &std::path::Path, tokens: &[&str]) -> Memvid {
    let mut mem = Memvid::create(path).unwrap();
    mem.enable_lex().unwrap();
    for token in tokens {
        mem.put_bytes_with_options(
            format!("payload for the {token} marker").as_bytes(),
            text_options(&format!("mv2://lex-report/{token}"), token),
        )
        .unwrap();
    }
    mem.commit().unwrap();
    mem
}

/// How many *fresh* frame passes this store has run.
///
/// The accessor must drive this to one and hold it there; only the unmemoised pass
/// a commit performs may move it further.
fn passes(mem: &Memvid) -> u32 {
    mem.lex_indexable_pass_count
        .load(std::sync::atomic::Ordering::Relaxed)
}

/// The accessor must never bring an engine into being.
///
/// A process that has not opened the engine yet reports `loaded == false`, which a
/// naive implementation would "helpfully" repair by opening it — and then a healthy
/// store would look empty, and a store whose open fails would have the very empty
/// engine installed that the commit gate exists to refuse. `Memvid::open` opens the
/// engine eagerly today, so the cold state is reached by clearing the field: the
/// point is not how the process got there but that reporting cannot change it.
#[test]
fn lex_engine_report_never_opens_the_engine() {
    let dir = TempDir::new().unwrap();
    let mut mem = text_store(&dir.path().join("cold.mv2"), &["zephyrolux", "quixotarial"]);
    assert!(
        mem.tantivy.is_some(),
        "fixture should start with an open engine"
    );

    mem.tantivy = None;
    mem.tantivy_dirty = false;

    let report = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert!(!report.loaded, "nothing opened an engine to answer");
    assert_eq!(report.docs, 0, "no engine standing can only report zero");
    assert!(
        report.indexable_frames.is_none(),
        "a cold process must not pay for a frame pass, got {:?}",
        report.indexable_frames
    );
    assert!(
        report.segments_advertised,
        "the catalog still advertises what it committed"
    );
    assert_eq!(
        report.expected_docs,
        Some(2),
        "the catalog's claim survives the engine being absent"
    );
    assert!(mem.tantivy.is_none(), "the accessor materialised an engine");
    assert!(
        mem.verify_lex_index().is_ok(),
        "a store with no engine standing is not a refused store"
    );
    assert!(
        !gate_refuses(report),
        "an engine this process never opened is not an empty index: {report:?}"
    );
}

/// `None` from the accessor means "this store makes no claim", never "refuse".
#[test]
fn lex_engine_report_is_silent_without_a_lexical_engine() {
    let dir = TempDir::new().unwrap();
    let mut mem = text_store(&dir.path().join("no-lex.mv2"), &["nimbostratus"]);
    mem.lex_enabled = false;
    assert!(
        mem.lex_engine_report().is_none(),
        "a store with no lexical engine must report nothing at all"
    );
    assert!(
        mem.verify_lex_index().is_ok(),
        "and the gate must not refuse it either"
    );
}

/// The cost property the fork refuses to give up, on the read path.
///
/// Counting indexable frames walks the whole frame table and reads a payload for
/// every frame whose text has to be derived. The commit gate pays it only after the
/// catalog has disagreed with the engine — "a store the catalog agrees with pays
/// nothing" — and this accessor is meant to run while serving a query, so it withholds
/// the count in every state the gate declines to inspect.
#[test]
fn lex_engine_report_withholds_the_frame_count_from_a_healthy_store() {
    let dir = TempDir::new().unwrap();
    let mut mem = text_store(
        &dir.path().join("healthy.mv2"),
        &["vermiliontide", "ferrulox"],
    );

    let report = assert_gate_agrees(&mut mem, "healthy");
    assert!(report.loaded && report.segments_advertised);
    assert_eq!(report.docs, 2);
    assert_eq!(report.expected_docs, Some(2));
    assert!(
        report.indexable_frames.is_none(),
        "the catalog and the engine agree, so no frame pass may have run: got {:?}",
        report.indexable_frames
    );

    // And the same store, once its engine has lost what it advertises, does pay:
    // that is the state where the answer changes the decision.
    emptied_engine(&mut mem);
    let report = assert_gate_agrees(&mut mem, "healthy-then-emptied");
    assert_eq!(
        report.indexable_frames,
        Some(2),
        "the defect state must report the frames that should be indexed"
    );
}

/// Every state a store can be reached in, both rule sources, one verdict.
///
/// The states are chosen so that each arm of the consumer's conjunction is the one
/// that decides: a healthy store (non-zero docs), a store whose engine lost its
/// documents (the refusal), an opaque-only store (a frame pass that finds nothing
/// to index must never refuse), and a store that advertises no segments at all.
#[test]
fn lex_engine_report_and_the_commit_gate_agree_on_every_state() {
    let dir = TempDir::new().unwrap();

    // A healthy store the catalog agrees with.
    let mut healthy = text_store(&dir.path().join("agree-healthy.mv2"), &["zephyrolux"]);
    let report = assert_gate_agrees(&mut healthy, "healthy");
    assert_eq!(report.docs, 1);
    assert!(report.indexable_frames.is_none());

    // The defect: the same store, engine emptied under an advertising catalog.
    emptied_engine(&mut healthy);
    let report = assert_gate_agrees(&mut healthy, "advertised-empty-with-indexable");
    assert_eq!(report.indexable_frames, Some(1));

    // An opaque-only store. Measured, and worth pinning: an octet-stream frame that
    // carries a uri is still indexable today, because the caller's text is stored
    // *augmented* (`title:`/`uri:` folded in) and any frame with a uri reads as
    // explicit at the commit and rebuild sites. The residual that would let a frame
    // carry the caller's own instruction is the `search_text_explicit` follow-up
    // recorded on `LexFramePlan`; until then "opaque" cannot mean "not indexable"
    // here, and pretending otherwise would make this test lie about the rule.
    let mut opaque = Memvid::create(dir.path().join("agree-opaque.mv2")).unwrap();
    opaque.enable_lex().unwrap();
    opaque
        .put_bytes_with_options(
            b"opaque blob bytes with no declared type".as_slice(),
            opaque_options("mv2://lex-report/opaque-only"),
        )
        .unwrap();
    opaque.commit().unwrap();
    let report = assert_gate_agrees(&mut opaque, "opaque-only");
    assert_eq!(report.docs, 1, "the opaque frame is indexed, see above");
    assert_eq!(report.expected_docs, Some(1));
    assert!(report.indexable_frames.is_none());
    emptied_engine(&mut opaque);
    let report = assert_gate_agrees(&mut opaque, "opaque-only-emptied");
    assert_eq!(report.indexable_frames, Some(1));

    // A store with nothing to index is healthy, not faulty: the pass runs here,
    // finds nothing, and both rule sources must abstain.
    let mut empty = Memvid::create(dir.path().join("agree-empty.mv2")).unwrap();
    empty.enable_lex().unwrap();
    empty.commit().unwrap();
    let report = assert_gate_agrees(&mut empty, "nothing-indexable");
    assert_eq!(
        report.indexable_frames,
        Some(0),
        "a real pass ran and found nothing, which is not the same as no pass"
    );

    // A store that advertises no segments claims no index, so it cannot fail to
    // account for one.
    let mut uncommitted = Memvid::create(dir.path().join("agree-zero-segments.mv2")).unwrap();
    uncommitted.enable_lex().unwrap();
    uncommitted
        .put_bytes_with_options(
            b"payload for the plenipotentius marker".as_slice(),
            text_options("mv2://lex-report/uncommitted", "plenipotentius"),
        )
        .unwrap();
    let report = assert_gate_agrees(&mut uncommitted, "zero-advertised-segments");
    assert!(
        !report.segments_advertised,
        "nothing advertised: {report:?}"
    );
    assert!(report.indexable_frames.is_none());
}

/// The one state where the two rule sources deliberately disagree, pinned so it stays
/// a decision and cannot drift into an accident.
///
/// The gate runs its frame pass for any catalog it can disprove: the short-circuit is
/// `actual_docs != 0 && expected_docs == Some(actual_docs)`, so a store holding two
/// documents under a catalog claim of nine falls through that guard, pays the pass,
/// and is refused with `CountDiverged`. The consumer's conjunction additionally
/// requires `docs == 0`, so on the identical store the report withholds the frame
/// count and the read path answers anyway. A partially diverged index is caught at
/// commit, not at query time — the caller's rule, not a gap in the report: widening
/// the accessor's pass to `docs != 0` would put a payload-reading scan of the frame
/// table in front of every query against a store whose catalog merely miscounts, for
/// a refusal that caller never makes.
#[test]
fn lex_engine_report_diverges_from_the_gate_on_a_diverged_count_by_design() {
    use crate::error::{LexIndexFault, MemvidError};

    let dir = TempDir::new().unwrap();
    let mut mem = text_store(
        &dir.path().join("diverged.mv2"),
        &["zephyrolux", "quixotarial"],
    );
    assert!(
        mem.verify_lex_index().is_ok(),
        "fixture: catalog and engine agree"
    );

    // The catalog now claims nine documents where the engine holds two — the one
    // symptom neither `emptied_engine` nor a failed open produces, because the engine
    // is standing and demonstrably not empty.
    mem.toc.indexes.lex = Some(LexIndexManifest {
        doc_count: 9,
        generation: 0,
        bytes_offset: 0,
        bytes_length: 0,
        checksum: [0; 32],
    });

    let report = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(report.docs, 2, "the engine is standing and non-empty");
    assert_eq!(
        report.expected_docs,
        Some(9),
        "the disagreement is visible in the report itself"
    );
    assert!(
        report.indexable_frames.is_none(),
        "docs != 0 is outside the one state a frame pass runs for, got {:?}",
        report.indexable_frames
    );
    assert!(
        !gate_refuses(report),
        "so the consumer abstains on a store the gate refuses: {report:?}"
    );

    let err = mem
        .verify_lex_index()
        .expect_err("the gate abstains on no catalog it can disprove");
    assert!(
        matches!(
            err,
            MemvidError::LexIndexIncomplete {
                expected_docs: Some(9),
                actual_docs: 2,
                fault: LexIndexFault::CountDiverged,
                ..
            }
        ),
        "the divergence is diagnosed as a divergence, not as an empty index: {err}"
    );
}

/// Neither a missing claim nor a missing pass may arrive as a zero.
///
/// `expected_docs == None` is "the catalog never recorded a claim", which is
/// ordinary for a store that carries Tantivy segments without a manifest — and
/// `None != Some(n)` is true for every `n`, so a consumer that read the absence as
/// zero would refuse stores the gate accepts. The same holds for `indexable_frames`.
#[test]
fn lex_engine_report_never_fabricates_a_document_claim() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no-claim.mv2");
    let mem = text_store(&path, &["obsidiancrest"]);
    drop(mem);

    // Reopening is the route to "segments advertised, engine full, no manifest
    // claim": the catalog advertises what it committed without restating a count.
    let mut reopened = Memvid::open(&path).unwrap();
    let report = assert_gate_agrees(&mut reopened, "reopened-without-a-claim");
    assert_eq!(
        report.expected_docs, None,
        "a store with no recorded claim must not be handed one"
    );
    assert_eq!(
        report.docs, 1,
        "and the engine really does hold the frame, so a consumer that turned the \
         missing claim into Some(0) would refuse a healthy store"
    );
    assert!(
        !gate_refuses(report),
        "an absent claim abstains: {report:?}"
    );

    // A pass that cannot run abstains the same way, rather than reporting an empty
    // index: a cold process has no engine to ask, and reporting the frames it could
    // not look at as a computed zero would refuse a store the gate accepts.
    let mut cold = text_store(&dir.path().join("no-pass.mv2"), &["septentrional"]);
    cold.tantivy = None;
    cold.tantivy_dirty = false;
    let report = assert_gate_agrees(&mut cold, "cold-no-pass");
    assert_eq!(report.indexable_frames, None);
    assert!(!gate_refuses(report));
}

/// One key, one sweep.
///
/// The accessor's entry condition — an engine standing, segments advertised, engine
/// empty — is not rare: it is permanently true for exactly the stores that keep
/// serving queries anyway, an image-only knowledge base or one text frame over
/// `max_index_payload()`. Those stores pay for `/find` and `/ask` traffic under the
/// service's single mutex, so a second report under an unchanged key may not touch the
/// frame table again.
#[test]
fn the_frame_pass_is_memoised_within_one_generation() {
    let dir = TempDir::new().unwrap();
    let mut mem = text_store(
        &dir.path().join("memo-hit.mv2"),
        &["heliotrope", "latticework"],
    );
    emptied_engine(&mut mem);
    mem.lex_indexable_pass_count
        .store(0, std::sync::atomic::Ordering::Relaxed);

    let first = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(first.docs, 0);
    assert_eq!(
        first.indexable_frames,
        Some(2),
        "both frames carry text the rule wants indexed"
    );
    assert!(
        gate_refuses(first),
        "this is the state the consumer is meant to refuse on"
    );
    assert_eq!(passes(&mem), 1, "the first read has to look");

    let second = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(second, first);
    assert_eq!(
        passes(&mem),
        1,
        "the second read answered from the memo; a sweep here is the defect this test exists for"
    );
}

/// A commit moves the memo's key.
///
/// Invalidating on the pair over-invalidates — a commit that never touched a frame
/// still moves `generation` and costs one sweep — and that is the accepted direction:
/// an under-invalidating cache would serve a count of frames that are no longer there.
#[test]
fn a_commit_invalidates_the_frame_pass_memo() {
    let dir = TempDir::new().unwrap();
    let mut mem = text_store(&dir.path().join("memo-invalidate.mv2"), &["heliotrope"]);
    emptied_engine(&mut mem);
    let memoised = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(memoised.indexable_frames, Some(1));

    // A real commit: the path that bumps `generation` and rewrites the footer.
    mem.put_bytes_with_options(
        b"payload for the underdrawing marker".as_slice(),
        text_options("mv2://lex-report/underdrawing", "underdrawing"),
    )
    .unwrap();
    mem.commit().unwrap();

    // The commit brought the engine up to date, so return the store to the state the
    // memo guards: advertised segments, an engine standing, nothing in it.
    emptied_engine(&mut mem);
    mem.lex_indexable_pass_count
        .store(0, std::sync::atomic::Ordering::Relaxed);
    let report = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(report.docs, 0);
    assert_eq!(
        report.indexable_frames,
        Some(2),
        "the committed frame is visible to a fresh pass"
    );
    assert_eq!(
        passes(&mem),
        1,
        "both halves of the key moved, so the cached verdict could not have answered"
    );

    let again = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(again, report);
    assert_eq!(
        passes(&mem),
        1,
        "and the recomputed verdict is what the next read pays"
    );
}

/// A pass that cannot finish is a memoised answer too.
///
/// One frame whose payload will not decode makes the whole pass return `Err`, and the
/// accessor reports that as `None` — no opinion, never `Some(0)`, which is the
/// fork's own abstain signal and would arm a false negative. Reporting it must also
/// not cost a sweep per request: a store that can never answer would otherwise pay
/// for it on every query.
#[test]
fn a_failed_frame_pass_is_memoised_as_no_opinion() {
    use crate::error::MemvidError;

    let dir = TempDir::new().unwrap();
    let mut mem = Memvid::create(dir.path().join("memo-poison.mv2")).unwrap();
    mem.enable_lex().unwrap();
    // No explicit text and no declared content type: the rule wants this frame's text
    // derived from its payload, which is the step that can fail.
    mem.put_bytes_with_options(
        b"the quaternary lexical marker".as_slice(),
        PutOptions {
            uri: Some("mv2://lex-report/quaternary".to_owned()),
            ..Default::default()
        },
    )
    .unwrap();
    mem.commit().unwrap();

    // Control, on the identical fixture with nothing poisoned: the frame has no text
    // of its own and no declared content type, so the rule must derive its text from
    // the payload. Proving it counts here is what makes the offset below the only
    // variable between `Some(n)` and `None` — otherwise a future change to the
    // selection rule could make this frame simply unindexable, and the failure
    // assertion would pass for the wrong reason.
    let control = mem
        .toc
        .frames
        .iter_mut()
        .find(|frame| frame.status == crate::types::FrameStatus::Active)
        .expect("the committed frame is active");
    control.search_text = None;
    emptied_engine(&mut mem);
    let report = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(report.docs, 0);
    assert!(
        report.indexable_frames.is_some_and(|count| count >= 1),
        "control: the unpoisoned frame is indexable by the rule, got {:?}",
        report.indexable_frames
    );

    // Poison it: the payload no longer sits where the frame says it does.
    // `payload_length` stays small so the selection rule still wants the frame.
    let frame = mem
        .toc
        .frames
        .iter_mut()
        .find(|frame| frame.status == crate::types::FrameStatus::Active)
        .expect("the committed frame is active");
    // No text of its own, so the selection rule sends this frame down the one arm
    // that can fail: derive the text from the payload.
    frame.search_text = None;
    frame.payload_offset = u64::MAX / 4;

    // The cause, asserted rather than assumed. `read_frame_payload_bytes`
    // (src/memvid/frame.rs:625) starts with `validate_frame_bounds`
    // (src/memvid/frame.rs:635-676), which refuses a range running past `data_end`
    // with `InvalidFrame { reason: "payload extends past data region" }`; that is what
    // the derive arm's payload read returns here. Naming the variant is what proves the
    // failure comes from the unreadable payload and not from any other error the pass
    // could stumble into — `Failed` alone is reachable from any error at all.
    let err = mem
        .lex_indexable_frame_count_fresh()
        .expect_err("the poisoned payload must make the pass fail");
    assert!(
        matches!(
            err,
            MemvidError::InvalidFrame {
                reason: "payload extends past data region",
                ..
            }
        ),
        "expected the bounds check to refuse the poisoned offset, got {err:?}"
    );

    emptied_engine(&mut mem);
    // The store changed underneath the cache, exactly as it would after a repair
    // tool edited the frame table; clear the memo so this assertion measures the
    // pass itself rather than a verdict recorded before the poison.
    mem.lex_indexable_memo = None;
    mem.lex_indexable_pass_count
        .store(0, std::sync::atomic::Ordering::Relaxed);

    let first = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(first.docs, 0);
    assert!(
        first.indexable_frames.is_none(),
        "an unreadable payload is no opinion, never a fabricated zero"
    );
    assert!(!gate_refuses(first), "no opinion never refuses");
    assert_eq!(passes(&mem), 1, "the pass ran, failed, and said so");

    let second = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    assert_eq!(second, first);
    assert_eq!(
        passes(&mem),
        1,
        "the failure is memoised as `Failed`; retrying it per request is the defect this test exists for"
    );
}

/// The three public spellings of the type name the same type, so the service can
/// reach it however it imports the crate.
#[test]
fn lex_engine_report_is_exported_where_the_service_can_see_it() {
    let dir = TempDir::new().unwrap();
    let mut mem = text_store(&dir.path().join("exported.mv2"), &["quixotarial"]);
    let report = mem
        .lex_engine_report()
        .expect("a lex-enabled store has an engine report");
    let via_root: crate::LexEngineReport = report;
    let via_memvid: crate::memvid::LexEngineReport = report;
    let via_search: crate::memvid::search::LexEngineReport = report;
    assert_eq!(via_root, via_memvid);
    assert_eq!(via_memvid, via_search);
    assert_eq!(via_search, report);
}
