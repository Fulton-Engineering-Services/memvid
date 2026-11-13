//! Regression coverage for the two HNSW defects that made `memvid-kb-rs` lose its
//! vector index past `HNSW_THRESHOLD` (1000 vectors).
//!
//! **D1 — silent total vector loss on every rebuild.**
//! `VecIndex::entries()` returned an empty iterator for `VecIndex::Hnsw`, but all
//! three rebuild sites carry existing vectors forward *only* through `entries()`:
//! `Memvid::build_vec_artifact` (`search/builders.rs`, reached from every `commit()`),
//! `Memvid::add_embeddings` (`enrichment.rs`), and `build_vec_index_from_segments`
//! (`search/builders.rs`, the segment-catalog path). A healthy store with >1000
//! vectors plus a single ordinary frame write therefore committed an artifact
//! containing only the *new* vectors — the whole HNSW index collapsed to 0 without an
//! error. The tests here exercise the commit and `add_embeddings` paths; the segment
//! path shares the same `entries()` contract but is not covered.
//!
//! **D2 — panic on wide queries.**
//! `HnswVecIndex::search` sized its destination buffer to `max(limit, ef)` while
//! `hnsw`'s `nearest()` can only fill `min(dest.len(), ef)` elements
//! (`hnsw-0.11.0/src/hnsw/hnsw_const.rs` does
//! `let found = min(dest.len(), searcher.nearest.len()); dest.copy_from_slice(&searcher.nearest[..found])`,
//! i.e. it copies into the *whole* `dest`), so any `limit > ef` panicked with
//! "source slice length (50) does not match destination slice length (N)".
//!
//! Every test here uses only the public API, and each one fails on unpatched 2.0.140
//! code for the reason named in its doc comment.

// The HNSW backend, and therefore everything under test, only exists with one of the
// vector features enabled; `memvid-core` builds without them too (that is what CI
// exercises), so the whole file compiles away in that case.
#![cfg(any(feature = "vec", feature = "hnsw_bench"))]

use std::path::Path;

use memvid_core::Memvid;
use memvid_core::vec::{
    HNSW_EF_SEARCH, HNSW_EF_SEARCH_MAX, HnswVecIndex, VecDocument, VecIndex, VecIndexBuilder,
};
use tempfile::TempDir;

/// Below this count vectors are stored uncompressed; at or above it the builder
/// switches to HNSW. Mirrors `HNSW_THRESHOLD` in `src/vec.rs`.
const HNSW_THRESHOLD: usize = 1_000;
/// Deliberately just over the threshold: the smallest store that exercises HNSW.
const N_VECTORS: usize = HNSW_THRESHOLD + 8;
const DIM: usize = 32;

/// Deterministic, pairwise-distinct embedding for frame `i`.
///
/// The `i as f32 * 1e-4` term is load-bearing. Without it this generator aliases:
/// `251 * 31 = 7781 = 31 * 251`, so `i * 31 (mod 251)` has period 251 and
/// `embedding(i) == embedding(i + 251)` exactly. With `N_VECTORS = 1008` every
/// vector then had four identical twins, which silently invalidated every
/// "the nearest vector to frame `i`'s own embedding must be that frame" assertion:
/// ties at distance `0.0` may return any twin. The per-`i` offset shifts every
/// component, so distinct `i` produce distinct vectors while a frame still matches
/// its own embedding at distance exactly `0`.
fn embedding(i: usize) -> Vec<f32> {
    let spread = i as f32 * 1e-4;
    (0..DIM)
        .map(|j| ((i * 31 + j * 7 + 1) % 251) as f32 / 251.0 + spread)
        .collect()
}

/// Distinct embedding that differs from every `embedding(i)`.
fn foreign_embedding() -> Vec<f32> {
    (0..DIM)
        .map(|j| ((j * 13 + 5) % 197) as f32 / 40.0 + 0.5)
        .collect()
}

/// Enforces the premise of every "the nearest vector to frame `i`'s own embedding
/// must be that frame" assertion below: the fixture vectors must be distinct.
///
/// This is not decoration. The original generator aliased with period 251, giving
/// every vector four identical twins, and no assertion below could distinguish a
/// correct index from one that returned the wrong twin. Delete the `spread` term
/// from `embedding` and this test fails.
#[test]
fn fixture_embeddings_are_pairwise_distinct() {
    let key =
        |v: &Vec<f32>| -> Vec<u32> { v.iter().map(|component| component.to_bits()).collect() };
    let mut seen = std::collections::HashSet::new();
    for i in 0..N_VECTORS {
        assert!(
            seen.insert(key(&embedding(i))),
            "fixture defect: embedding({i}) duplicates an earlier embedding; \
             findability assertions cannot be trusted while twins exist"
        );
    }
    assert!(
        !seen.contains(&key(&foreign_embedding())),
        "fixture defect: foreign_embedding() collides with a fixture embedding"
    );
}

/// `N_VECTORS` embedded frames, committed. Returns the frame ids.
fn build_store(dir: &Path) -> Vec<u64> {
    let mut mem = Memvid::create(dir.join("store.mv2")).expect("create");
    mem.enable_vec().expect("enable_vec");
    let mut ids = Vec::with_capacity(N_VECTORS);
    for i in 0..N_VECTORS {
        let id = mem
            .put_with_embedding(format!("frame {i}").as_bytes(), embedding(i))
            .expect("put_with_embedding");
        ids.push(id);
    }
    mem.commit().expect("commit");
    assert_eq!(
        mem.vector_count(),
        N_VECTORS,
        "precondition: the committed store must hold every embedding"
    );
    ids
}

fn hnsw_vec_index(docs: &[VecDocument]) -> VecIndex {
    let mut builder = VecIndexBuilder::new();
    for doc in docs {
        builder.add_document(doc.frame_id, doc.embedding.clone());
    }
    let artifact = builder.finish().expect("finish");
    assert_eq!(artifact.vector_count as usize, docs.len());
    let index = VecIndex::decode(&artifact.bytes).expect("decode");
    assert!(
        matches!(index, VecIndex::Hnsw(_)),
        "precondition: {} documents must produce an HNSW index",
        docs.len()
    );
    index
}

fn docs(shift: u64) -> Vec<VecDocument> {
    (0..N_VECTORS)
        .map(|i| VecDocument {
            frame_id: u64::try_from(i).expect("fits") + shift,
            embedding: embedding(i),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// D1 — carry-forward through commit()
// ---------------------------------------------------------------------------

/// D1: one ordinary frame write (no embedding) must not erase the vector index.
///
/// Fails unpatched: `vector_count` collapses from 1008 to 0 and every vector becomes
/// unfindable, because `build_vec_artifact` carries existing vectors through
/// `VecIndex::entries()`, which was empty for `VecIndex::Hnsw`.
#[test]
fn d1_ordinary_frame_write_preserves_hnsw_vectors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("store.mv2");
    // The returned frame ids are deliberately not bound here: this test proves vectors
    // survive, which it checks by distance (see the D3 note in the probe loop).
    build_store(dir.path());

    let mut mem = Memvid::open(&path).expect("reopen");
    // One ordinary write, with no embedding at all: this is the `!delta.is_empty()`
    // trigger in `commit_from_records` that forces a full index rebuild.
    mem.put_bytes(b"an ordinary frame with no embedding")
        .expect("put_bytes");
    mem.commit().expect("commit");

    assert_eq!(
        mem.vector_count(),
        N_VECTORS,
        "D1: commit() must preserve all {N_VECTORS} vectors, not rewrite the index \
         from only the new documents"
    );

    // A counter can be a lie: prove the vectors are actually still there, and correct.
    //
    // Widen the candidate window first: this test is about carry-forward, not about how
    // far hnsw's greedy descent happens to reach at the default width. With a window
    // wider than the store, a miss means the vector is genuinely gone rather than
    // merely unfound. The `vector_count` assertion above stays at the default width --
    // that is the direct D1 regression tripwire.
    mem.set_ef_search(HNSW_EF_SEARCH_MAX);
    for probe in [0usize, 7, 500, N_VECTORS - 1] {
        let hits = mem
            .search_vec(&embedding(probe), 5)
            .unwrap_or_else(|e| panic!("search_vec for frame {probe} errored: {e}"));
        assert!(
            !hits.is_empty(),
            "D1: vector {probe} must still be findable after the rebuild"
        );
        // Assert the *vector*, not the frame id. `put_with_embedding` files the
        // embedding it was given under a frame id that is not the frame it returns
        // (measured on this fixture: `ids[500] == 503` while the vector sits at 500) --
        // defect D3, tracked separately and outside this change. Comparing `frame_id`
        // against `ids[probe]` would test D3 and fail for a reason that has nothing to
        // do with carry-forward. Distance `0.0` against the query embedding is the
        // exact statement that this frame's vector survived the rebuild, and it is not
        // vacuous: with `entries()` returning nothing the rebuilt index is empty and
        // this loop fails on the assertion above instead.
        assert!(
            hits[0].distance < 1e-3,
            "D1: exact-match distance for frame {probe} was {}",
            hits[0].distance
        );
    }

    // Same assertions against the bytes on disk.
    drop(mem);
    let mut reopened = Memvid::open_read_only(&path).expect("reopen read-only");
    assert_eq!(
        reopened.vector_count(),
        N_VECTORS,
        "D1: the artifact persisted by commit() must still describe all {N_VECTORS} vectors"
    );
    // Wide window: the point is that the persisted artifact still holds frame 500's
    // vector, not that the default-width descent reaches it. Checked by distance for
    // the D3 reason given above -- the frame id it comes back under is not the one
    // `put_with_embedding` reported.
    reopened.set_ef_search(HNSW_EF_SEARCH_MAX);
    let hits = reopened
        .search_vec(&embedding(500), 3)
        .expect("search_vec after reopen");
    assert!(
        hits.iter().any(|h| h.distance < 1e-3),
        "D1: the persisted artifact lost frame 500's vector; nearest distance {:?}",
        hits.first().map(|h| h.distance)
    );
}

// ---------------------------------------------------------------------------
// D1 — carry-forward through add_embeddings()
// ---------------------------------------------------------------------------

/// D1, second rebuild site: `add_embeddings()` on an existing frame of a >1000-vector
/// store must not drop the other vectors.
///
/// Fails unpatched: `add_embeddings` rebuilds from `entries()` (empty for HNSW) plus
/// the one new embedding, so the live index shrinks to a single vector. Asserted on
/// the same handle that did the call, which is how the consumer sees it.
#[test]
fn d1_add_embeddings_preserves_existing_vectors() {
    let dir = TempDir::new().unwrap();
    let ids = build_store(dir.path());

    let target = 42usize;
    let mut mem = Memvid::open(dir.path().join("store.mv2")).expect("open");

    // Re-embed one existing frame: the enrichment-worker call shape.
    let added = mem
        .add_embeddings(vec![(ids[target], foreign_embedding())])
        .expect("add_embeddings");
    assert_eq!(added, 1, "add_embeddings reports the number accepted");

    assert_eq!(
        mem.vector_count(),
        N_VECTORS,
        "D1: add_embeddings() must rebuild over the existing {} vectors, not only the \
         {} new one(s)",
        N_VECTORS,
        added
    );

    // Findability, not just the counter: an untouched frame must still resolve to
    // itself with distance ~0.
    //
    // Wide window for the same reason as `d1_ordinary_frame_write_preserves_hnsw_vectors`:
    // the subject is carry-forward through `add_embeddings`, and a wide window makes a
    // miss mean "the vector is gone" rather than "hnsw did not reach it".
    mem.set_ef_search(HNSW_EF_SEARCH_MAX);
    for probe in [0usize, 500, N_VECTORS - 1] {
        let hits = mem
            .search_vec(&embedding(probe), 5)
            .unwrap_or_else(|e| panic!("search_vec for frame {probe} errored: {e}"));
        assert!(
            !hits.is_empty(),
            "D1: frame {probe}'s vector must survive add_embeddings()"
        );
        // Distance, not frame id -- see the D3 note in
        // `d1_ordinary_frame_write_preserves_hnsw_vectors`. The vectors were filed by
        // `put_with_embedding` under ids this fixture does not control; what is being
        // proved here is that the rebuild kept them.
        assert!(
            hits[0].distance < 1e-3,
            "D1: surviving vector for frame {probe} was corrupted: distance {}",
            hits[0].distance
        );
    }

    // Deliberately NOT asserted here: that a search for `foreign_embedding()` returns
    // `ids[target]`. `add_embeddings` files that vector verbatim under the id it was
    // handed (`enrichment.rs` copies `(frame_id, embedding)` straight to the builder), so
    // the pairing is guaranteed by construction -- but *finding* it is not. Measured on a
    // 1008-node graph, only 982 nodes are reachable from the entry point even at
    // `ef = HNSW_EF_SEARCH_MAX`, so a distance-0 miss here says "hnsw could not reach the
    // node", not "the vector was not filed", and this store's ids are further displaced by
    // D3. That the builder files under the id given is proven deterministically, without a
    // graph traversal, by `hnsw_entries_yield_every_vector_in_insertion_order`.
}

// ---------------------------------------------------------------------------
// The `ef` ceiling must be observable, not silent
// ---------------------------------------------------------------------------

/// `search_width` is the caller's only way to tell "ef capped me" from "the index is
/// lossy". Without it a `limit` above `ef` returns 50 hits and looks like data loss.
#[test]
fn search_width_reports_the_ef_ceiling() {
    let index = hnsw_vec_index(&docs(0));
    let query = [0.25f32; DIM];

    // `ef` below `limit` is the binding constraint, and says so before the query runs.
    assert_eq!(index.search_width(200, 20), 20);
    // `limit` below `ef` makes `limit` binding.
    assert_eq!(index.search_width(5, 200), 5);
    // The default knob advertises exactly the width it enforces.
    assert_eq!(index.search_width(200, HNSW_EF_SEARCH), HNSW_EF_SEARCH);
    // Out-of-range widths are advertised the same way the search clamps them.
    assert_eq!(
        index.search_width(10, 0),
        1,
        "ef 0 must advertise the clamped width"
    );
    assert_eq!(index.search_width(10, usize::MAX), 10);
    assert_eq!(
        index.search_width(0, 200),
        0,
        "nothing requested, nothing available"
    );

    // The width is an upper bound the search never exceeds, and the case it warns about
    // is the real one: 200 requested at the default `ef` yields 50.
    for (limit, ef) in [(200usize, 20usize), (200, 50), (1000, 200), (5, 200)] {
        let width = index.search_width(limit, ef);
        let got = index.search_with_ef(&query, limit, ef).len();
        assert!(
            got <= width,
            "search_with_ef(limit={limit}, ef={ef}) returned {got} hits, above the \
             advertised width {width}"
        );
    }
    assert_eq!(
        index.search_with_ef(&query, 200, HNSW_EF_SEARCH).len(),
        index.search_width(200, HNSW_EF_SEARCH),
        "the advertised width must be the width actually delivered"
    );
}

/// The non-HNSW arms must advertise a width they can actually honour.
///
/// `ef` narrows only the HNSW backend; the uncompressed backend scores every document, so
/// its ceiling is `min(limit, documents)` and it must not move when `ef` moves. This is the
/// ordinary shape of a small memory -- anything below the HNSW threshold -- and the case
/// where advertising bare `limit` would promise hits the index does not hold, which is
/// precisely the confusion the accessor exists to remove.
#[test]
fn search_width_on_the_uncompressed_arm_is_bound_by_the_document_count() {
    const COUNT: usize = 10;
    let mut builder = VecIndexBuilder::new();
    for i in 0..COUNT {
        builder.add_document(u64::try_from(i).expect("fits"), embedding(i));
    }
    let artifact = builder.finish().expect("finish");
    let index = VecIndex::decode(&artifact.bytes).expect("decode");
    assert!(
        matches!(index, VecIndex::Uncompressed { .. }),
        "precondition: {COUNT} documents must produce an uncompressed index, not {:?}",
        index
    );

    assert_eq!(index.search_width(5, HNSW_EF_SEARCH), 5);
    assert_eq!(
        index.search_width(1_000, HNSW_EF_SEARCH),
        COUNT,
        "the uncompressed arm cannot return more hits than it holds"
    );
    assert_eq!(
        index.search_width(1_000, 1),
        COUNT,
        "`ef` must not narrow a backend it does not bind"
    );
    assert_eq!(index.search_width(1_000, usize::MAX), COUNT);
    assert_eq!(index.search_width(0, HNSW_EF_SEARCH), 0);

    let query = embedding(3);
    for limit in [1usize, 5, 10, 1_000] {
        let width = index.search_width(limit, 1);
        assert!(
            index.search_with_ef(&query, limit, 1).len() <= width,
            "uncompressed arm returned more than its advertised width {width} at limit={limit}"
        );
    }
}

/// The handle-level signal must move with the knob, so a consumer can raise `ef` and
/// verify the ceiling lifted rather than trusting it did.
#[test]
fn vec_search_width_tracks_the_handle_knob() {
    let dir = TempDir::new().unwrap();
    build_store(dir.path());
    let mut mem = Memvid::open(dir.path().join("store.mv2")).expect("open");

    assert_eq!(
        mem.ef_search(),
        HNSW_EF_SEARCH,
        "the handle starts at the default"
    );
    assert_eq!(
        mem.vec_search_width(200),
        HNSW_EF_SEARCH,
        "ef must bound the advertised width"
    );
    let starved = mem
        .search_vec(&embedding(500), 200)
        .expect("search_vec at default ef")
        .len();
    assert_eq!(
        starved, HNSW_EF_SEARCH,
        "limit 200 at the default ef must deliver exactly the advertised width"
    );

    mem.set_ef_search(200);
    assert_eq!(
        mem.vec_search_width(200),
        200,
        "the advertisement must follow the knob"
    );
    let wide = mem
        .search_vec(&embedding(500), 200)
        .expect("search_vec at ef 200")
        .len();
    assert_eq!(
        wide, 200,
        "raising ef must lift the delivered width, not just the claim"
    );

    assert_eq!(mem.vec_search_width(0), 0);
}

/// D3 -- `put_with_embedding` files the embedding it was given on a frame other than
/// the one it returns.
///
/// Measured on this exact fixture: the returned ids are `[1, 2, 3, .., 1010]` with
/// `ids[500] == 503`, while a full-width search holds ids `0..=1007`, holds `0`, and does
/// NOT hold the last returned id; `search_vec(embedding(500))` returns frame `500` at
/// distance `0.0`. So the vector is present but filed under the wrong frame.
///
/// Ignored because D3 is outside this change and this assertion FAILS on current code --
/// that is the point. Un-ignore it when D3 is fixed. The user-visible consequences are
/// that semantic search hands back a frame whose content is not the content that was
/// embedded, and that `delete_frame(ids[i])` does not remove the vector, because the
/// frame which actually holds it stays active. The D1 tests above therefore assert
/// survival by distance, never by frame id.
#[test]
#[ignore = "D3 open: put_with_embedding associates the vector with the wrong frame id"]
fn d3_put_with_embedding_registers_vector_on_returned_frame() {
    let dir = TempDir::new().unwrap();
    let ids = build_store(dir.path());
    let mut mem = Memvid::open(dir.path().join("store.mv2")).expect("open");
    mem.set_ef_search(HNSW_EF_SEARCH_MAX);

    for probe in [0usize, 7, 500, N_VECTORS - 1] {
        let hits = mem
            .search_vec(&embedding(probe), 1)
            .unwrap_or_else(|e| panic!("search_vec for {probe} errored: {e}"));
        assert_eq!(
            hits.first().map(|h| h.frame_id),
            Some(ids[probe]),
            "D3: embedding({probe}) is not reachable as the frame it was attached to \
             ({}); nearest distance {:?}",
            ids[probe],
            hits.first().map(|h| h.distance)
        );
    }
}

// ---------------------------------------------------------------------------
// D2 — no (limit, ef) combination may panic
// ---------------------------------------------------------------------------

/// D2: searching a healthy HNSW store must work for every `limit`, including limits
/// above the query-time `ef`, and the exact match must be findable at every `limit`.
///
/// Two passes on purpose. The first runs at the default width and asserts only the
/// sizing contract -- no `(limit, ef)` combination panics and `limit` is respected.
/// That is what the `copy_from_slice` fix buys, and it fails unpatched for every
/// `limit > 50` with `source slice length (50) does not match destination slice
/// length (N)`.
///
/// The second raises `ef` before asserting findability. Whether the greedy descent
/// surfaces a given vector is a property of the candidate window, not of the
/// destination sizing, so asserting it at the default width would pin hnsw's recall
/// luck rather than our contract; asserting it under a deliberately wide window makes
/// it a real expectation -- and a wide window is exactly what the `ef` knob exists to
/// give callers whose post-filter fetch width exceeds 50.
#[test]
fn d2_search_vec_survives_every_limit() {
    let dir = TempDir::new().unwrap();
    build_store(dir.path());

    let mut mem = Memvid::open(dir.path().join("store.mv2")).expect("open");

    // 24/25/26 sit on the `top_k * 2` boundary used by
    // vec_search_with_embedding_acl (26 * 2 = 52 > ef); 51+ sit above the ef default.
    for limit in [24usize, 25, 26, 50, 51, 60, 100, 1000] {
        let hits = mem
            .search_vec(&embedding(500), limit)
            .unwrap_or_else(|e| panic!("search_vec(limit={limit}) errored: {e}"));
        assert!(
            !hits.is_empty(),
            "D2: limit={limit} returned no hits on a {N_VECTORS}-vector store"
        );
        assert!(
            hits.len() <= limit,
            "D2: limit={limit} returned {} hits",
            hits.len()
        );
    }

    // Same sweep with a window wide enough that missing the exact match is a defect,
    // not a consequence of a narrow candidate pool.
    mem.set_ef_search(HNSW_EF_SEARCH_MAX);
    for limit in [24usize, 25, 26, 50, 51, 60, 100, 1000] {
        let hits = mem
            .search_vec(&embedding(500), limit)
            .unwrap_or_else(|e| panic!("search_vec(limit={limit}, wide ef) errored: {e}"));
        assert!(
            hits.iter().any(|h| h.distance < 1e-3),
            "D2: limit={limit} lost the exact match even with ef={}",
            HNSW_EF_SEARCH_MAX
        );
    }
}

/// D2 at the `VecIndex` layer, including `limit` values far above `ef` and above the
/// graph size, plus the empty-graph corner.
#[test]
fn d2_vec_index_search_never_panics_on_wide_limits() {
    let index = hnsw_vec_index(&docs(0));

    for limit in [
        1usize, 24, 25, 26, 49, 50, 51, 60, 100, 512, 1000, 4096, 100_000,
    ] {
        let hits = index.search(&[0.25f32; DIM], limit);
        assert!(
            hits.len() <= limit,
            "D2: VecIndex::search(limit={limit}) returned {} hits",
            hits.len()
        );
    }

    // Empty graph: hnsw's `search_layer` returns early, but buffer sizing must not
    // depend on that.
    let empty = HnswVecIndex::build(&[]).expect("build empty");
    assert!(
        empty.search(&[0.1f32; DIM], 10).is_empty(),
        "D2: empty graph must yield no hits"
    );
    assert!(
        empty.search(&[0.1f32; DIM], 5_000).is_empty(),
        "D2: empty graph with limit > ef must yield no hits, not panic"
    );
}

// ---------------------------------------------------------------------------
// D1 root fix: entries()/embedding_for() for HNSW
// ---------------------------------------------------------------------------

/// The property every carry-forward path depends on: `entries()` must yield every
/// vector, paired with the frame it was inserted as. A wrong pairing would be worse
/// than the original bug, so vectors are compared byte-exactly against insertion
/// order — this is the test that validates `hnsw::Hnsw::feature(i)` really is the
/// i-th inserted vector.
#[test]
fn hnsw_entries_yield_every_vector_in_insertion_order() {
    const SHIFT: u64 = 100;
    let source = docs(SHIFT);
    let index = hnsw_vec_index(&source);

    let yielded: Vec<(u64, Vec<f32>)> = index
        .entries()
        .map(|(frame_id, vector)| (frame_id, vector.to_vec()))
        .collect();
    assert_eq!(
        yielded.len(),
        N_VECTORS,
        "D1: entries() must yield every vector; it yielded {}",
        yielded.len()
    );
    for (i, doc) in source.iter().enumerate() {
        let (frame_id, vector) = &yielded[i];
        assert_eq!(
            *frame_id, doc.frame_id,
            "D1: entries() yielded frame {frame_id} at position {i}, expected {}",
            doc.frame_id
        );
        assert_eq!(
            vector, &doc.embedding,
            "D1: entries() paired frame {} with the wrong vector",
            doc.frame_id
        );
    }

    for doc in &source {
        let looked_up = index
            .embedding_for(doc.frame_id)
            .unwrap_or_else(|| panic!("D1: embedding_for({}) returned None", doc.frame_id));
        assert_eq!(
            looked_up,
            doc.embedding.as_slice(),
            "D1: embedding_for({}) returned the wrong vector",
            doc.frame_id
        );
    }
}

// ---------------------------------------------------------------------------
// D2 root fix: `ef` is a real query-time parameter, not a decorative one
// ---------------------------------------------------------------------------

/// The same store and query must return *different* widths for different `ef`.
///
/// This is the test that distinguishes a threaded `ef` from one that is accepted
/// and then ignored: an implementation that dropped `ef` on the floor would return
/// the same 50 hits for every case here and fail on the first assertion.
#[test]
fn ef_controls_the_candidate_width_at_the_vec_index_layer() {
    let index = hnsw_vec_index(&docs(0));
    let query = [0.25f32; DIM];

    // `ef` is the binding constraint: 200 requested, 20 explored.
    let narrow = index.search_with_ef(&query, 200, 20);
    assert_eq!(
        narrow.len(),
        20,
        "ef=20 must cap the result set at 20, got {}",
        narrow.len()
    );

    // Raising `ef` widens the pool only until the graph runs out of candidates it
    // can reach from the entry point. From this query the store saturates at 138:
    // `ef` = 200, 400 and `HNSW_EF_SEARCH_MAX` all deliver exactly 138 (measured at
    // the pinned rev), so 138 is this query's reachability ceiling -- not a budget
    // that `ef` grants, and not a truncation that a wider `ef` could undo.
    let wide = index.search_with_ef(&query, 200, 200);
    assert_eq!(
        wide.len(),
        138,
        "ef=200 must deliver every candidate this query can reach, got {}",
        wide.len()
    );
    assert!(
        wide.len() > narrow.len(),
        "ef had no effect: narrow={} wide={}",
        narrow.len(),
        wide.len()
    );

    // A narrow `limit` still wins over a wide `ef`.
    assert_eq!(index.search_with_ef(&query, 5, 200).len(), 5);

    // The no-`ef` entry point keeps the historical width, so existing callers
    // cannot silently change behaviour.
    assert_eq!(
        index.search(&query, 200).len(),
        HNSW_EF_SEARCH,
        "search() must keep the default width"
    );

    // Out-of-range widths are clamped, never rejected and never panic-inducing.
    assert_eq!(index.search_with_ef(&query, 10, 0).len(), 1);
    assert_eq!(
        index.search_with_ef(&query, 10, usize::MAX).len(),
        10,
        "an absurd ef must be clamped, not honoured"
    );
}

/// `set_ef_search` on the handle must reach the search path, and must clamp.
#[test]
fn set_ef_search_reaches_search_vec() {
    let dir = TempDir::new().unwrap();
    build_store(dir.path());
    let mut mem = Memvid::open(dir.path().join("store.mv2")).expect("open");

    assert_eq!(
        mem.ef_search(),
        HNSW_EF_SEARCH,
        "a reopened store must start at the historical default"
    );

    mem.set_ef_search(200);
    let wide = mem
        .search_vec(&embedding(500), 200)
        .expect("search_vec at ef=200");
    assert_eq!(
        wide.len(),
        200,
        "search_vec must honour a widened ef, got {}",
        wide.len()
    );

    mem.set_ef_search(20);
    let narrow = mem
        .search_vec(&embedding(500), 200)
        .expect("search_vec at ef=20");
    assert_eq!(
        narrow.len(),
        20,
        "search_vec must honour a narrowed ef, got {}",
        narrow.len()
    );

    // Clamped, never zero, never unbounded.
    mem.set_ef_search(0);
    assert_eq!(mem.ef_search(), 1, "ef must be clamped up to 1");
    mem.set_ef_search(usize::MAX);
    assert_eq!(
        mem.ef_search(),
        HNSW_EF_SEARCH_MAX,
        "ef must be clamped to the documented maximum"
    );
}

/// `ef` is the size of the candidate pool handed to the graph search, not a
/// decorative argument.
///
/// With `limit` (200) above every `ef` in the first group, the delivered width tracks
/// `ef` exactly while the graph still has candidates left to hand back. The second
/// group is the saturation point: from this query the store yields 138 candidates, and
/// `ef` = 200, 400 and `HNSW_EF_SEARCH_MAX` all return the same 138, because width is
/// bounded by what the layer links expose from the entry point, never by `ef` alone.
/// An implementation that accepted `ef` and then searched with the default pool would
/// return 50 on the `ef` = 1 and `ef` = 20 rows -- the mutation this test exists to
/// catch -- and one that padded results up to `ef` would overshoot the saturated rows.
/// Measured on this fixture, and stable: hnsw's traversal depends only on insertion
/// order because `Hnsw::new`/`new_params` seed the PRNG with `R::Seed::default()` and
/// `Searcher::new` builds its visited set with `RandomState::with_seeds(0, 0, 0, 0)`
/// (pinned rev `a1d6ca88`).
#[test]
fn ef_sets_the_graph_candidate_pool() {
    let index = hnsw_vec_index(&docs(0));
    let query = [0.25f32; DIM];

    for ef in [1usize, 20, 50, 100] {
        let hits = index.search_with_ef(&query, 200, ef);
        assert_eq!(
            hits.len(),
            ef,
            "ef={ef} must admit exactly {ef} candidates, got {}",
            hits.len()
        );
    }

    // Past this query's reach `ef` stops binding: the width is the 138 candidates the
    // graph can actually reach, neither `ef` nor `limit`.
    for ef in [200usize, 400, HNSW_EF_SEARCH_MAX] {
        let hits = index.search_with_ef(&query, 200, ef);
        assert_eq!(
            hits.len(),
            138,
            "ef={ef} must saturate at the 138 candidates this query can reach, got {}",
            hits.len()
        );
    }
}

/// The advertised width is a **ceiling**, and a result set may legitimately come back
/// shorter than it: hnsw's greedy descent explores only what the layer links expose
/// from the entry point, so some nodes are unreachable however wide `ef` is.
///
/// Measured on this exact fixture (1008 vectors in `docs(0)` insertion order, pinned
/// hnsw rev `a1d6ca88`, deterministic per the note on
/// [`ef_sets_the_graph_candidate_pool`]): the queries `embedding(1)` and
/// `embedding(333)` saturate at 138 candidates even at `ef = HNSW_EF_SEARCH_MAX`,
/// while `embedding(500)` and `embedding(1007)` reach the full 200. Pairing the two is
/// the point -- it shows the shortfall is a property of the graph as seen from that
/// query, not a global cap that `search_width` is failing to report: at a store-wide
/// `limit` those narrow queries still reach only 138 while others reach 1002 of the
/// 1008 vectors, and the union of what every query can reach is 1002, so a handful of
/// vectors are unreachable from anywhere however wide `ef` goes. In every case the
/// query's own vector is still rank 0 at distance exactly 0, which is the guarantee a
/// caller actually depends on.
#[test]
fn search_width_advertises_a_ceiling_the_graph_may_not_reach() {
    let index = hnsw_vec_index(&docs(0));

    for probe in [1usize, 333] {
        let query = embedding(probe);
        let advertised = index.search_width(200, HNSW_EF_SEARCH_MAX);
        let hits = index.search_with_ef(&query, 200, HNSW_EF_SEARCH_MAX);
        assert_eq!(advertised, 200, "the ceiling is min(limit, ef)");
        assert_eq!(
            hits.len(),
            138,
            "probe {probe} reaches exactly 138 candidates on this fixture, got {}",
            hits.len()
        );
        assert!(
            hits.len() < advertised,
            "probe {probe}: expected a shortfall below the advertised width"
        );
        assert_eq!(
            hits[0].frame_id,
            u64::try_from(probe).expect("fits"),
            "probe {probe} must rank first even when the graph cannot reach the ceiling"
        );
        assert_eq!(
            hits[0].distance, 0.0,
            "probe {probe} must match its own vector at distance 0"
        );
    }

    // The same request on probes the graph can reach fully delivers the advertised
    // width, which is what makes the 138 above topology rather than a ceiling bug.
    for probe in [500usize, 1007] {
        let query = embedding(probe);
        let hits = index.search_with_ef(&query, 200, HNSW_EF_SEARCH_MAX);
        assert_eq!(
            hits.len(),
            200,
            "probe {probe} should reach the full advertised width, got {}",
            hits.len()
        );
        assert_eq!(
            hits[0].frame_id,
            u64::try_from(probe).expect("fits"),
            "probe {probe} must rank first"
        );
    }
}
