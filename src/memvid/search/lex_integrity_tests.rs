//! End-to-end tests for the permanent-empty-lexical-index defect.
//!
//! Every assertion here is expressible against the code *before* the fix, so the
//! suite is a red-before witness: run it on the base revision and it fails.
//! What the fix guarantees:
//!
//! 1. An embedded index that cannot be opened is never laundered into a successful
//!    search with too few rows — the store rebuilds it from its own frames.
//! 2. Every indexing site applies one selection rule, so a frame cannot be
//!    findable after `put` and permanently gone after `commit`.
//! 3. A store that advertises an index it cannot account for refuses `commit`
//!    loudly, and `vacuum` is the recovery.

use crate::search::TantivyEngine;
use crate::types::{AclEnforcementMode, DocMetadata, SearchRequest};
use crate::{Memvid, PutOptions};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use tempfile::TempDir;

/// Frames carry a token this distinctive so a hit can only have come from the
/// frame this test wrote.
fn text_frame_options(uri: &str, token: &str) -> PutOptions {
    PutOptions {
        uri: Some(uri.to_owned()),
        title: Some(format!("frame {token}")),
        search_text: Some(format!("the {token} lexical marker for integrity testing")),
        ..Default::default()
    }
}

/// Reachability of `token` through the engine that the defect made silently
/// short. `search`, not `find`: `find` is the legacy artifact-index surface,
/// which a Tantivy-backed store does not have.
fn hits(mem: &mut Memvid, token: &str, phase: &str) -> usize {
    match run_search(mem, token) {
        Ok(response) => response.hits.len(),
        // A Tantivy-only store has no legacy artifact index to fall back to, and
        // the fallback arm surfaces that as `LexNotEnabled` instead of an empty
        // result. That is a separate, pre-existing rough edge; here it means the
        // same thing as no hits: the token is not reachable through the lexical
        // index. Any other error is a real failure and panics.
        Err(crate::MemvidError::LexNotEnabled) => 0,
        Err(err) => panic!(
            "search({token:?}) errored at {phase} (lex_enabled={}, tantivy={:?}): {err}",
            mem.lex_enabled,
            mem.tantivy.as_ref().map(|_| "some"),
        ),
    }
}

fn run_search(mem: &mut Memvid, token: &str) -> crate::Result<crate::types::SearchResponse> {
    mem.search(SearchRequest {
        query: token.to_owned(),
        top_k: 10,
        snippet_chars: 200,
        uri: None,
        scope: None,
        cursor: None,
        #[cfg(feature = "temporal_track")]
        temporal: None,
        as_of_frame: None,
        as_of_ts: None,
        no_sketch: true,
        acl_context: None,
        acl_enforcement_mode: AclEnforcementMode::Audit,
    })
}

fn advertised_segments(mem: &Memvid) -> usize {
    mem.toc.segment_catalog.tantivy_segments.len()
}

/// A store whose embedded Tantivy segments cannot be opened must not answer a
/// search with `Ok` and nothing.
///
/// Before the fix, `init_tantivy` swallowed the failed open, kept the empty
/// engine, and then the trust rule forced `needs_rebuild = false` because
/// segments were advertised. The empty index was therefore permanent and the
/// search answered `Ok(0 hits)` for the life of the store — a wrong answer, not
/// a missing log line.
#[test]
fn an_unopenable_embedded_index_is_not_reported_as_working() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("unopenable.mv2");

    let (segment_offset, segment_length) = {
        let mut mem = Memvid::create(&path).unwrap();
        mem.enable_lex().unwrap();
        mem.put_bytes_with_options(
            b"payload for the zephyrolux marker".as_slice(),
            text_frame_options("mv2://lex-integrity/1", "zephyrolux"),
        )
        .unwrap();
        mem.put_bytes_with_options(
            b"payload for the quixotarial marker".as_slice(),
            text_frame_options("mv2://lex-integrity/2", "quixotarial"),
        )
        .unwrap();
        mem.commit().unwrap();
        assert_eq!(
            hits(&mut mem, "zephyrolux", "fixture"),
            1,
            "fixture did not index"
        );
        assert!(
            advertised_segments(&mem) > 0,
            "fixture advertised no embedded Tantivy segments, so it cannot exercise \
             the unopenable-index path"
        );
        let descriptor = &mem.toc.segment_catalog.tantivy_segments[0];
        (
            descriptor.common.bytes_offset,
            descriptor.common.bytes_length,
        )
    };
    assert!(segment_length > 0, "advertised segment is empty");

    // Scribble over the advertised segment bytes: this is the failed open, not a
    // simulation of one — the same read `init_tantivy` performs now fails.
    {
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(segment_offset)).unwrap();
        let garbage = vec![0xA5_u8; usize::try_from(segment_length).unwrap()];
        file.write_all(&garbage).unwrap();
        file.sync_all().unwrap();
    }

    // Opening stays possible, and the search stays truthful: the frames are still
    // in the store, so the index is rebuilt from them rather than trusted empty.
    let mut mem = Memvid::open(&path).unwrap();
    assert_eq!(
        hits(&mut mem, "zephyrolux", "after-corruption"),
        1,
        "an advertised-but-unopenable index answered with too few rows"
    );
    assert_eq!(
        hits(&mut mem, "quixotarial", "after-corruption"),
        1,
        "an advertised-but-unopenable index answered with too few rows"
    );
    // The store is still writable: the failure is loud, not terminal.
    mem.commit().unwrap();
}

/// One selection rule, everywhere.
///
/// Put-time indexing used to be gated only on `search_text`, while both commit
/// paths and the rebuild also excluded non-text content types and oversized
/// payloads. A frame whose payload merely decoded as text could therefore be
/// carried into the committed index by the pending entry the put created, and
/// then vanish for good the first time the index was rebuilt from the frames —
/// after which nothing ever rebuilt it again, because advertised segments
/// suppressed the rebuild.
///
/// Pending frames are not yet resolvable by `search`, so findability is observed
/// after `commit`, and then across a full rebuild (`vacuum`) and a reopen. All
/// three must agree, and agree with the rule.
#[test]
fn put_and_commit_agree_on_which_frames_are_findable() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("agreement.mv2");
    let mut mem = Memvid::create(&path).unwrap();
    mem.enable_lex().unwrap();

    // (a) opaque content type with a text payload and no explicit search text: the
    // rule excludes it, so it may never be findable, at any point in the store's
    // life. This is the frame that used to be indexed at put and dropped at commit.
    let opaque = PutOptions {
        uri: Some("mv2://lex-integrity/opaque".to_owned()),
        metadata: Some(DocMetadata {
            mime: Some("application/octet-stream".to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    };
    // (b) no declared content type and a text payload: the payload is the text, so
    // the rule indexes it.
    let undeclared = PutOptions {
        uri: Some("mv2://lex-integrity/undeclared".to_owned()),
        ..Default::default()
    };
    // (c) an explicit search text is an instruction and outranks the content-type
    // gate, so this frame stays findable despite being declared opaque.
    let explicit = PutOptions {
        uri: Some("mv2://lex-integrity/explicit".to_owned()),
        metadata: Some(DocMetadata {
            mime: Some("application/octet-stream".to_owned()),
            ..Default::default()
        }),
        search_text: Some("ferrulox is the explicit-text marker".to_owned()),
        ..Default::default()
    };

    mem.put_bytes_with_options(b"vermiliontide payload text".as_slice(), opaque)
        .unwrap();
    mem.put_bytes_with_options(b"obsidiancrest payload text".as_slice(), undeclared)
        .unwrap();
    mem.put_bytes_with_options(b"ignored payload".as_slice(), explicit)
        .unwrap();

    // Put-time indexing must already cover every frame the rule will keep, so no
    // frame enters the index through one path and is skipped by another. This is
    // the only place the put-time gate is directly observable: pending frames are
    // not resolvable by `search` until commit. Before the fix the put gate looked
    // only for an explicit `search_text`, so the payload-derived frame was missing
    // here and reached the index for the first time at commit.
    let indexed_at_put = mem
        .tantivy
        .as_ref()
        .expect("lexical index enabled")
        .num_docs();
    assert_eq!(
        indexed_at_put, 2,
        "put must have indexed the text-payload frame and the explicit-search-text \
         frame, and only those"
    );

    mem.commit().unwrap();
    let expected = (0, 1, 1);
    let after_commit = probe(&mut mem, "after-commit");
    assert_eq!(
        after_commit, expected,
        "the selection rule must skip the opaque payload and keep the text payload \
         plus the frame with explicit search text"
    );

    // A full rebuild from the frames must reach the same answer the incremental
    // commit produced. Before the fix the put-time gate indexed the opaque frame,
    // the commit kept it, and the rebuild dropped it — permanently.
    mem.vacuum().unwrap();
    let after_rebuild = probe(&mut mem, "after-rebuild");
    assert_eq!(
        after_rebuild, after_commit,
        "a frame's findability changed when the lexical index was rebuilt from the \
         frames: (vermiliontide, obsidiancrest, ferrulox) was {after_commit:?} after \
         commit and is {after_rebuild:?} after rebuild"
    );

    // And a reopen must reach it too.
    drop(mem);
    let mut reopened = Memvid::open(&path).unwrap();
    let after_reopen = probe(&mut reopened, "after-reopen");
    assert_eq!(
        after_reopen, after_commit,
        "findability changed again across reopen"
    );
    reopened.commit().unwrap();
}

fn probe(mem: &mut Memvid, phase: &str) -> (usize, usize, usize) {
    (
        hits(mem, "vermiliontide", phase),
        hits(mem, "obsidiancrest", phase),
        hits(mem, "ferrulox", phase),
    )
}

/// A store that advertises an index which cannot account for its own frames
/// refuses `commit`, and `vacuum` repairs it.
///
/// The empty engine is injected directly, because with the trust rule fixed the
/// open path now rebuilds such an index instead of reaching this state: this test
/// pins the gate itself, not one particular route into it.
#[test]
fn commit_refuses_a_store_that_cannot_account_for_its_advertised_index() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("refusal.mv2");
    let mut mem = Memvid::create(&path).unwrap();
    mem.enable_lex().unwrap();
    mem.put_bytes_with_options(
        b"payload for the plenipotentius marker".as_slice(),
        text_frame_options("mv2://lex-integrity/a", "plenipotentius"),
    )
    .unwrap();
    mem.put_bytes_with_options(
        b"payload for the nimbostratus marker".as_slice(),
        text_frame_options("mv2://lex-integrity/b", "nimbostratus"),
    )
    .unwrap();
    mem.commit().unwrap();
    assert_eq!(hits(&mut mem, "plenipotentius", "fixture"), 1);
    assert!(
        advertised_segments(&mem) > 0,
        "fixture advertised no embedded Tantivy segments"
    );

    // The defect state: the catalog still advertises its segments, the engine
    // behind them holds nothing, and nothing is pending that would repair it.
    mem.tantivy = Some(TantivyEngine::create().unwrap());
    mem.tantivy_dirty = false;

    let err = mem
        .commit()
        .expect_err("commit accepted a store whose advertised index is empty");
    let message = err.to_string();
    assert!(
        message.starts_with("lexical index incomplete:"),
        "refusal must be the lexical-index fault, got {message:?}"
    );
    assert!(
        message.contains("hold 0 reachable document(s)"),
        "refusal must report the reachable-document count, got {message:?}"
    );

    // Reads stay available on a refused store, and the documented recovery is
    // `vacuum`: it rebuilds the index from the frames and verifies the result.
    mem.vacuum().unwrap();
    assert_eq!(
        hits(&mut mem, "plenipotentius", "after-vacuum"),
        1,
        "vacuum did not rebuild the lexical index"
    );
    assert_eq!(
        hits(&mut mem, "nimbostratus", "after-vacuum"),
        1,
        "vacuum lost a frame"
    );
    mem.commit().unwrap();

    drop(mem);
    let mut reopened = Memvid::open(&path).unwrap();
    assert_eq!(
        hits(&mut reopened, "plenipotentius", "after-vacuum-reopen"),
        1
    );
    assert_eq!(
        hits(&mut reopened, "nimbostratus", "after-vacuum-reopen"),
        1
    );
    reopened.commit().unwrap();
}

/// A store with nothing to index is healthy, not faulty: the gate must not
/// refuse it, and `vacuum` on it must succeed.
#[test]
fn a_store_with_nothing_to_index_is_never_refused() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no-index.mv2");
    let mut mem = Memvid::create(&path).unwrap();
    mem.enable_lex().unwrap();
    let opaque = PutOptions {
        uri: Some("mv2://lex-integrity/opaque-only".to_owned()),
        metadata: Some(DocMetadata {
            mime: Some("application/octet-stream".to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    };
    mem.put_bytes_with_options(
        b"opaque blob bytes with no declared type".as_slice(),
        opaque,
    )
    .unwrap();
    mem.commit().unwrap();
    mem.vacuum().unwrap();
    mem.commit().unwrap();
    assert_eq!(hits(&mut mem, "plenipotentius", "nothing-to-index"), 0);
}

/// A blank `search_text` is an absent `search_text` in **both** text accessors.
///
/// Before this change the four text readers disagreed about blank. At 9228579
/// `frame_preview`, `frame_canonical_text` and `frame_search_text` each accepted
/// any `Some(text)` — `Some("   ")` included — and only `frame_content` had a
/// blank check at all, `!search.is_empty()`, which catches `""` but not
/// whitespace. So a frame holding `Some("   ")` got `frame_content` to fall
/// through to `frame_canonical_text`, which handed the blank straight back, while
/// the indexing gates refused to tokenise whitespace: such a frame belonged to no
/// index at all — put, commit, or rebuild — while the frame listing went on
/// advertising it. The fix is one predicate,
/// [`crate::memvid::lifecycle::search_text_is_present`], that all four consult.
///
/// This is asserted on the accessors rather than through `find`, and the reason
/// is worth keeping: the blank state is not reachable end-to-end through `put`.
/// `put` augments the caller's text before storing it, and `augment_search_text`
/// drops a blank base, so a labelled frame ends up storing non-blank
/// `title:`/`uri:` segments (which then outrank the content gates — the residual
/// recorded as the `search_text_explicit` follow-up) and an unlabelled frame ends
/// up storing `None`, which takes the derive branch whether or not the predicate
/// is correct. Only a directly held frame can be blank, which is exactly what the
/// accessors have to cope with.
#[test]
fn both_text_accessors_read_a_blank_search_text_from_the_payload() {
    const PAYLOAD: &str = "septentrional payload text";
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("blank-accessors.mv2");
    let mut mem = Memvid::create(&path).unwrap();
    mem.enable_lex().unwrap();
    // A real put, so the payload exists in storage and canonical text can be read.
    mem.put_bytes_with_options(PAYLOAD.as_bytes(), PutOptions::default())
        .unwrap();
    // Frames reach `toc.frames` at commit; before it they live only in the WAL.
    mem.commit().unwrap();
    let mut frame = mem
        .toc
        .frames
        .first()
        .cloned()
        .expect("the store holds one frame");
    frame.search_text = Some("   ".to_owned());

    assert!(
        !crate::memvid::lifecycle::search_text_is_present(Some("   ")),
        "blank must count as absent"
    );
    assert!(
        !crate::memvid::lifecycle::search_text_is_present(None),
        "no search text at all is absent, by definition"
    );
    assert!(
        !crate::memvid::lifecycle::search_text_is_present(Some("   \n\t ")),
        "whitespace with newlines and tabs is still absent"
    );
    assert!(
        crate::memvid::lifecycle::search_text_is_present(Some("text")),
        "text that carries a token is present"
    );

    let content = mem
        .frame_content(&frame)
        .expect("content of a frame with a stored payload");
    assert!(
        content.contains("septentrional"),
        "frame_content must derive from the payload when search_text is blank, got {content:?}"
    );
    let search = mem
        .frame_search_text(&frame)
        .expect("search text of a frame with a stored payload");
    assert!(
        search.contains("septentrional"),
        "frame_search_text must derive from the payload when search_text is blank, \
         got {search:?}"
    );
}

/// A frame with no declared content type is text-indexable, and stays that way
/// through the shape an upgrade takes.
///
/// The old put path had no content-type gate at all, so stores written before
/// this change already hold mime-less frames in the embedded index and their
/// manifests advertise them. Were the shared selection rule to read a *missing*
/// content type as opaque, the first rebuild after the upgrade would drop those
/// frames below the count the store itself claims, and the commit gate would
/// refuse a store that was healthy before the upgrade — the one outcome this
/// change must not produce. A declared non-text type is still excluded, which
/// `put_and_commit_agree_on_which_frames_are_findable` pins. Reopen-then-commit is
/// the upgrade shape, so it must return `Ok`.
#[test]
fn a_frame_with_no_declared_content_type_survives_reopen_and_commit() {
    const TOKEN: &str = "hypsometrical";
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("undeclared-mime.mv2");
    let mut mem = Memvid::create(&path).unwrap();
    mem.enable_lex().unwrap();
    mem.put_bytes_with_options(
        format!("{TOKEN} payload text").as_bytes(),
        PutOptions::default(),
    )
    .unwrap();
    let indexed_at_put = mem
        .tantivy
        .as_ref()
        .expect("lexical index enabled")
        .num_docs();
    assert!(
        indexed_at_put >= 1,
        "a frame with no declared content type was not indexed at put, so a rebuild \
         after an upgrade would drive the engine below the count its own manifest \
         advertises"
    );
    mem.commit().unwrap();
    assert_eq!(
        hits(&mut mem, TOKEN, "after-commit"),
        1,
        "a frame with no declared content type must be findable after commit"
    );
    mem.vacuum().unwrap();
    assert_eq!(
        hits(&mut mem, TOKEN, "after-rebuild"),
        1,
        "a full rebuild dropped the mime-less frame the store advertises"
    );
    drop(mem);
    let mut reopened = Memvid::open(&path).unwrap();
    assert_eq!(
        hits(&mut reopened, TOKEN, "after-reopen"),
        1,
        "the mime-less frame was not findable after reopening"
    );
    reopened
        .commit()
        .expect("reopen-then-commit of a healthy mime-less store must not fault");
}

/// A frame put with a blank `search_text` is findable from its payload after
/// `commit` and after a full rebuild plus a reopen — both halves, because the
/// original defect was one path writing a frame another path dropped.
///
/// This is a **guard, not a witness**: `put` augments the caller's text before
/// storing it, and `augment_search_text` drops a blank base, so an unlabelled
/// frame reaches the store with `search_text: None` and takes the derivation
/// branch with or without `search_text_is_present`. The assertion that actually
/// pins the fix is
/// [`both_text_accessors_read_a_blank_search_text_from_the_payload`], which holds
/// a blank frame directly. What this test adds is the end-to-end guarantee that
/// the population agrees across `commit`, `vacuum`, and reopen for a frame that
/// carries no content of its own beyond the payload.
#[test]
fn a_frame_put_with_a_blank_search_text_is_findable_from_its_payload() {
    const TOKEN: &str = "circumvolant";
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("blank-search-text.mv2");
    let mut mem = Memvid::create(&path).unwrap();
    mem.enable_lex().unwrap();
    let blank = PutOptions {
        search_text: Some("   ".to_owned()),
        ..Default::default()
    };
    mem.put_bytes_with_options(format!("{TOKEN} payload text").as_bytes(), blank)
        .unwrap();
    mem.commit().unwrap();
    assert_eq!(
        hits(&mut mem, TOKEN, "after-commit"),
        1,
        "a frame whose search_text is blank must be indexed from its payload at commit"
    );
    mem.vacuum().unwrap();
    assert_eq!(
        hits(&mut mem, TOKEN, "after-rebuild"),
        1,
        "a full rebuild from the frames lost the frame whose search_text is blank"
    );
    drop(mem);
    let mut reopened = Memvid::open(&path).unwrap();
    assert_eq!(
        hits(&mut reopened, TOKEN, "after-reopen"),
        1,
        "the blank-search-text frame was not findable after reopening the store"
    );
    reopened.commit().unwrap();
}
