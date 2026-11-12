use blake3::hash;
use serde::{Deserialize, Serialize};

use crate::{MemvidError, Result, types::FrameId};

#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
use hnsw::{Hnsw, Params, Searcher};
#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
use rand_pcg::Pcg64;
#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
use space::Metric;

fn vec_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian()
}

#[allow(clippy::cast_possible_truncation)]
const VEC_DECODE_LIMIT: usize = crate::MAX_INDEX_BYTES as usize;

#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
const HNSW_THRESHOLD: usize = 1000;
/// Fixed-point scaling factor for HNSW distances.
/// Necessary because `space::Metric` requires `Unit: Unsigned`, but we use f32 L2 distances.
/// 100,000.0 gives 1e-5 precision and max distance ~42,000 (enough for high-dim embeddings).
#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
const HNSW_DISTANCE_SCALE: f32 = 100_000.0;

/// Default query-time candidate pool width (`ef`) for HNSW search.
///
/// hnsw explores `ef` candidates per query and hands back at most
/// `min(ef, graph_len)` of them, so this value also bounds how many hits an
/// HNSW-backed search can return regardless of the `limit` a caller passes.
pub const HNSW_EF_SEARCH: usize = 50;

/// Upper bound for `ef`. The pool is explored on every query, so an unbounded
/// knob would let a mistyped value pin a core; callers are clamped, not rejected.
pub const HNSW_EF_SEARCH_MAX: usize = 4096;

/// The single definition of how wide an HNSW query is allowed to be, shared by
/// [`HnswVecIndex::search_with_ef`] (which sizes the destination buffer) and
/// [`HnswVecIndex::search_width`] (which advertises it).
///
/// Both must agree exactly: if the advertised width drifts above the buffer width,
/// callers are told "not truncated" while hits were dropped, and if it drifts below,
/// they are told "truncated" for complete results. `dest.len() <= ef` is the invariant
/// the patched hnsw relies on to fill the destination it is handed.
#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
fn target_width(limit: usize, ef: usize, graph_len: usize) -> usize {
    if limit == 0 {
        return 0;
    }
    let ef = ef.clamp(1, HNSW_EF_SEARCH_MAX);
    limit.min(ef).min(graph_len).max(1)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecDocument {
    pub frame_id: FrameId,
    pub embedding: Vec<f32>,
}

#[derive(Default)]
pub struct VecIndexBuilder {
    documents: Vec<VecDocument>,
}

impl VecIndexBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_document<I>(&mut self, frame_id: FrameId, embedding: I)
    where
        I: Into<Vec<f32>>,
    {
        self.documents.push(VecDocument {
            frame_id,
            embedding: embedding.into(),
        });
    }

    pub fn finish(self) -> Result<VecIndexArtifact> {
        #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
        if self.documents.len() >= HNSW_THRESHOLD {
            return self.finish_hnsw();
        }

        let bytes = bincode::serde::encode_to_vec(&self.documents, vec_config())?;

        let checksum = *hash(&bytes).as_bytes();
        let dimension = self
            .documents
            .first()
            .map_or(0, |doc| u32::try_from(doc.embedding.len()).unwrap_or(0));
        #[cfg(feature = "parallel_segments")]
        let bytes_uncompressed = self
            .documents
            .iter()
            .map(|doc| doc.embedding.len() * std::mem::size_of::<f32>())
            .sum::<usize>() as u64;
        Ok(VecIndexArtifact {
            bytes,
            vector_count: self.documents.len() as u64,
            dimension,
            checksum,
            #[cfg(feature = "parallel_segments")]
            bytes_uncompressed,
        })
    }

    #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
    #[allow(clippy::cast_possible_truncation)]
    fn finish_hnsw(self) -> Result<VecIndexArtifact> {
        let count = self.documents.len() as u64;
        let dimension = self
            .documents
            .first()
            .map(|d| d.embedding.len() as u32)
            .unwrap_or(0);

        #[cfg(feature = "parallel_segments")]
        let bytes_uncompressed = self
            .documents
            .iter()
            .map(|doc| doc.embedding.len() * std::mem::size_of::<f32>())
            .sum::<usize>() as u64;

        let index = HnswVecIndex::build(&self.documents)?;
        let bytes = bincode::serde::encode_to_vec(&index, vec_config())?;
        let checksum = *hash(&bytes).as_bytes();

        Ok(VecIndexArtifact {
            bytes,
            vector_count: count,
            dimension,
            checksum,
            #[cfg(feature = "parallel_segments")]
            bytes_uncompressed,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VecIndexArtifact {
    pub bytes: Vec<u8>,
    pub vector_count: u64,
    pub dimension: u32,
    pub checksum: [u8; 32],
    #[cfg(feature = "parallel_segments")]
    pub bytes_uncompressed: u64,
}

#[derive(Debug, Clone)]
pub enum VecIndex {
    Uncompressed {
        documents: Vec<VecDocument>,
    },
    Compressed(crate::vec_pq::QuantizedVecIndex),
    #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
    Hnsw(HnswVecIndex),
}

impl VecIndex {
    /// Decode vector index from bytes
    /// For backward compatibility, defaults to uncompressed if no manifest provided
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        Self::decode_with_compression(bytes, crate::VectorCompression::None)
    }

    /// Decode vector index with compression mode from manifest
    ///
    /// ALWAYS tries uncompressed format first, regardless of compression flag.
    /// This is necessary because `MIN_VECTORS_FOR_PQ` threshold (100 vectors)
    /// causes most segments to be stored as uncompressed even when Pq96 is requested.
    /// Falls back to PQ format for true compressed segments.
    pub fn decode_with_compression(
        bytes: &[u8],
        _compression: crate::VectorCompression,
    ) -> Result<Self> {
        // Try uncompressed format first, regardless of compression flag.
        // This is necessary because MIN_VECTORS_FOR_PQ threshold (100 vectors)
        // causes most segments to be stored as uncompressed even when Pq96 is requested.
        match bincode::serde::decode_from_slice::<Vec<VecDocument>, _>(
            bytes,
            bincode::config::standard()
                .with_fixed_int_encoding()
                .with_little_endian()
                .with_limit::<VEC_DECODE_LIMIT>(),
        ) {
            Ok((documents, read)) if read == bytes.len() => {
                tracing::debug!(
                    bytes_len = bytes.len(),
                    docs_count = documents.len(),
                    "decoded as uncompressed"
                );
                return Ok(Self::Uncompressed { documents });
            }
            Ok((_, read)) => {
                tracing::debug!(
                    bytes_len = bytes.len(),
                    read = read,
                    "uncompressed decode partial read, trying HNSW/PQ"
                );
            }
            Err(err) => {
                tracing::debug!(
                    error = %err,
                    bytes_len = bytes.len(),
                    "uncompressed decode failed, trying HNSW/PQ"
                );
            }
        }

        #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
        {
            match bincode::serde::decode_from_slice::<HnswVecIndex, _>(
                bytes,
                bincode::config::standard()
                    .with_fixed_int_encoding()
                    .with_little_endian()
                    .with_limit::<VEC_DECODE_LIMIT>(),
            ) {
                Ok((index, _)) => {
                    tracing::debug!(bytes_len = bytes.len(), "decoded as HNSW");
                    return Ok(Self::Hnsw(index));
                }
                Err(err) => {
                    tracing::debug!(
                        error = %err,
                        bytes_len = bytes.len(),
                        "HNSW decode failed, trying PQ"
                    );
                }
            }
        }

        // Try Product Quantization format
        match crate::vec_pq::QuantizedVecIndex::decode(bytes) {
            Ok(quantized_index) => {
                tracing::debug!(bytes_len = bytes.len(), "decoded as PQ");
                Ok(Self::Compressed(quantized_index))
            }
            Err(err) => {
                tracing::debug!(
                    error = %err,
                    bytes_len = bytes.len(),
                    "PQ decode also failed"
                );
                Err(MemvidError::InvalidToc {
                    reason: "unsupported vector index encoding".into(),
                })
            }
        }
    }

    #[must_use]
    pub fn search(&self, query: &[f32], limit: usize) -> Vec<VecSearchHit> {
        self.search_with_ef(query, limit, HNSW_EF_SEARCH)
    }

    /// Nearest-neighbour search with an explicit query-time candidate width.
    ///
    /// `ef` is the candidate pool the HNSW backend explores per query. It only
    /// affects that backend: the uncompressed backend scores every document and
    /// the compressed backend searches exhaustively, so for both `limit` alone
    /// applies. An HNSW search returns at most `min(limit, ef, graph_len)` hits,
    /// so a caller needing `limit` results needs `ef >= limit`.
    #[must_use]
    pub fn search_with_ef(&self, query: &[f32], limit: usize, ef: usize) -> Vec<VecSearchHit> {
        if query.is_empty() {
            return Vec::new();
        }
        // `ef` is only meaningful for the HNSW backend. Without `vec`/`hnsw_bench`
        // that arm is compiled out -- the uncompressed backend scores every document
        // and the compressed one searches exhaustively -- so the parameter would be
        // unused and `cargo clippy -D warnings` (which CI runs on default features)
        // would fail the build.
        #[cfg(not(any(feature = "vec", feature = "hnsw_bench")))]
        let _ = ef;
        match self {
            VecIndex::Uncompressed { documents } => {
                let mut hits: Vec<VecSearchHit> = documents
                    .iter()
                    .map(|doc| {
                        let distance = l2_distance(query, &doc.embedding);
                        VecSearchHit {
                            frame_id: doc.frame_id,
                            distance,
                        }
                    })
                    .collect();
                hits.sort_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                hits.truncate(limit);
                hits
            }
            VecIndex::Compressed(quantized) => quantized.search(query, limit),
            #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
            VecIndex::Hnsw(index) => index.search_with_ef(query, limit, ef),
        }
    }

    /// Maximum number of hits [`search_with_ef`](Self::search_with_ef) can return for
    /// this `(limit, ef)` pair.
    ///
    /// Only the HNSW backend can be narrowed by `ef`; the uncompressed and compressed
    /// backends score every stored document, so `limit` binds them -- and, for the
    /// uncompressed arm, the number of documents actually stored. A caller that gets
    /// fewer hits than this width was asked either hit the `ef` ceiling or found
    /// genuinely fewer matching vectors.
    #[must_use]
    pub fn search_width(&self, limit: usize, ef: usize) -> usize {
        // Only the HNSW arm below consumes `ef`; without either vector feature it is
        // compiled out, and CI's default-features `clippy -D warnings` would fail on
        // the unused parameter.
        #[cfg(not(any(feature = "vec", feature = "hnsw_bench")))]
        let _ = ef;
        match self {
            #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
            VecIndex::Hnsw(index) => index.search_width(limit, ef),
            // Scoring every document still cannot yield more hits than there are.
            VecIndex::Uncompressed { documents } => limit.min(documents.len()),
            // `QuantizedIndex` exposes no document count, so `limit` is the only bound
            // that can be advertised honestly here.
            VecIndex::Compressed(_) => limit,
        }
    }

    #[must_use]
    pub fn entries(&self) -> Box<dyn Iterator<Item = (FrameId, &[f32])> + '_> {
        match self {
            VecIndex::Uncompressed { documents } => Box::new(
                documents
                    .iter()
                    .map(|doc| (doc.frame_id, doc.embedding.as_slice())),
            ),
            VecIndex::Compressed(_) => {
                // Compressed vectors don't have direct f32 access
                Box::new(std::iter::empty())
            }
            #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
            VecIndex::Hnsw(index) => Box::new(
                // The hnsw graph keeps every feature it was given, and
                // `feature(i)` for `i in 0..len()` yields the i-th insertion --
                // which is the order `HnswVecIndex::build` pushes into `ids`.
                // Without this, rebuilding an HNSW-backed store drops every
                // existing vector; see the integration test
                // `hnsw_entries_yield_every_vector_in_insertion_order`.
                // `ids` and the graph are appended together in `build`, so these
                // lengths agree by construction; the bound keeps a corrupt or
                // future-divergent artifact from panicking in a rebuild path.
                (0..index.graph.len().min(index.ids.len()))
                    .map(|i| (index.ids[i], index.graph.feature(i).as_slice())),
            ),
        }
    }

    #[must_use]
    pub fn embedding_for(&self, frame_id: FrameId) -> Option<&[f32]> {
        match self {
            VecIndex::Uncompressed { documents } => documents
                .iter()
                .find(|doc| doc.frame_id == frame_id)
                .map(|doc| doc.embedding.as_slice()),
            VecIndex::Compressed(_) => {
                // Compressed vectors don't have direct f32 access
                None
            }
            #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
            VecIndex::Hnsw(index) => index
                .ids
                .iter()
                .position(|&id| id == frame_id)
                .filter(|&i| i < index.graph.len())
                .map(|i| index.graph.feature(i).as_slice()),
        }
    }

    pub fn remove(&mut self, frame_id: FrameId) {
        match self {
            VecIndex::Uncompressed { documents } => {
                documents.retain(|doc| doc.frame_id != frame_id);
            }
            VecIndex::Compressed(_quantized) => {
                // Compressed indices are immutable
            }
            #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
            VecIndex::Hnsw(_) => {
                // HNSW indices are immutable in this implementation
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VecSearchHit {
    pub frame_id: FrameId,
    pub distance: f32,
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::l2_distance_simd(a, b)
}

#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Euclidean;

#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
impl Metric<Vec<f32>> for Euclidean {
    type Unit = u32;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u32 {
        let d = l2_distance(a, b);
        // Saturating cast prevents overflow for huge distances (though unlikely for embeddings)
        (d * HNSW_DISTANCE_SCALE).min(u32::MAX as f32) as u32
    }
}

#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
#[derive(Clone, Serialize, Deserialize)]
#[allow(clippy::unsafe_derive_deserialize)]
pub struct HnswVecIndex {
    graph: Hnsw<Euclidean, Vec<f32>, Pcg64, 16, 32>,
    ids: Vec<FrameId>,
    dimension: u32,
}

#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
impl std::fmt::Debug for HnswVecIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswVecIndex")
            .field("dimension", &self.dimension)
            .field("vector_count", &self.ids.len())
            .finish_non_exhaustive()
    }
}

#[cfg(any(feature = "vec", feature = "hnsw_bench"))]
impl HnswVecIndex {
    #[allow(clippy::cast_possible_truncation)]
    pub fn build(documents: &[VecDocument]) -> Result<Self> {
        let params = Params::new().ef_construction(100);
        let mut graph = Hnsw::new_params(Euclidean, params);
        let mut ids = Vec::with_capacity(documents.len());
        let mut searcher = Searcher::default();

        for doc in documents {
            graph.insert(doc.embedding.clone(), &mut searcher);
            ids.push(doc.frame_id);
        }

        Ok(Self {
            graph,
            ids,
            dimension: documents
                .first()
                .map(|d| d.embedding.len() as u32)
                .unwrap_or(0),
        })
    }

    #[must_use]
    pub fn search(&self, query: &[f32], limit: usize) -> Vec<VecSearchHit> {
        self.search_with_ef(query, limit, HNSW_EF_SEARCH)
    }

    /// Nearest-neighbour search with an explicit query-time candidate width.
    ///
    /// `hnsw::nearest` fills at most `min(ef, graph_len)` neighbours and then
    /// copies them with `copy_from_slice` into the *whole* destination slice, so a
    /// destination longer than it can fill panics. `want` below is the largest
    /// width both sides agree on, which makes every `(limit, ef)` pair safe --
    /// including `limit > ef` (the shape that used to panic) and `ef > graph_len`.
    #[must_use]
    pub fn search_with_ef(&self, query: &[f32], limit: usize, ef: usize) -> Vec<VecSearchHit> {
        if limit == 0 || query.is_empty() || self.ids.is_empty() {
            return Vec::new();
        }
        // Use thread-local searcher and dest buffer to avoid per-query allocations
        thread_local! {
            static SEARCHER: std::cell::RefCell<Searcher<u32>> = std::cell::RefCell::new(Searcher::new());
            static DEST: std::cell::RefCell<Vec<space::Neighbor<u32>>> = const { std::cell::RefCell::new(Vec::new()) };
        }

        // Clamped here as well as inside `target_width`: this `ef` is the *candidate
        // pool* handed to `hnsw::nearest`, which allocates and explores it, so it must
        // never see a raw value. `target_width` clamps for its own sake (and for the
        // accessor, which has no other clamp); the two must agree so that
        // `dest.len() <= ef` holds, which is what the patched `nearest` relies on to
        // fill the destination slice it is handed.
        let ef = ef.clamp(1, HNSW_EF_SEARCH_MAX);
        let want = target_width(limit, ef, self.graph.len());

        SEARCHER.with(|searcher_cell| {
            DEST.with(|dest_cell| {
                let mut searcher = searcher_cell.borrow_mut();
                let mut dest = dest_cell.borrow_mut();

                // The thread-local only ever grows; slice it down to `want` so hnsw
                // never receives a destination wider than it can fill.
                if dest.len() < want {
                    dest.resize(
                        want,
                        space::Neighbor {
                            index: !0,
                            distance: 0,
                        },
                    );
                }

                // Convert query slice to Vec for the graph
                let query_vec: Vec<f32> = query.to_vec();

                let found = self
                    .graph
                    .nearest(&query_vec, ef, &mut searcher, &mut dest[..want]);

                found
                    .iter()
                    .take(limit)
                    .map(|neighbor| VecSearchHit {
                        frame_id: self.ids[neighbor.index],
                        distance: (neighbor.distance as f32) / HNSW_DISTANCE_SCALE,
                    })
                    .collect()
            })
        })
    }

    /// Maximum number of hits [`search_with_ef`](Self::search_with_ef) can return for
    /// this `(limit, ef)` pair.
    ///
    /// `ef` is a **ceiling** on returned width, not a floor on exploration: asking for
    /// more hits than `ef` silently yields `ef`, which is easy to mistake for an empty
    /// or lossy index. Compare a result's length against this width to tell the two
    /// apart. A shortfall below it means the graph could not reach that many nodes from
    /// its entry point -- hnsw's descent explores only what the layer links expose, and
    /// a 1008-node store measured here reachable to 982 -- which no caller can observe
    /// through the public API without this.
    #[must_use]
    pub fn search_width(&self, limit: usize, ef: usize) -> usize {
        if limit == 0 || self.ids.is_empty() {
            return 0;
        }
        target_width(limit, ef, self.graph.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_roundtrip() {
        let mut builder = VecIndexBuilder::new();
        builder.add_document(1, vec![0.0, 1.0, 2.0]);
        builder.add_document(2, vec![1.0, 2.0, 3.0]);
        let artifact = builder.finish().expect("finish");
        assert_eq!(artifact.vector_count, 2);
        assert_eq!(artifact.dimension, 3);

        let index = VecIndex::decode(&artifact.bytes).expect("decode");
        let hits = index.search(&[0.0, 1.0, 2.0], 10);
        assert_eq!(hits[0].frame_id, 1);
    }

    #[test]
    fn l2_distance_behaves() {
        let d = l2_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 1e-6);
    }

    /// Test that HNSW is used for indices with >1000 vectors (HNSW_THRESHOLD)
    #[test]
    #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
    fn hnsw_threshold_triggers_hnsw_index() {
        use super::HNSW_THRESHOLD;

        // Create index with exactly HNSW_THRESHOLD vectors
        let mut builder = VecIndexBuilder::new();
        let dim = 32;
        for i in 0..HNSW_THRESHOLD {
            let embedding: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 / 1000.0).collect();
            builder.add_document(i as FrameId, embedding);
        }

        let artifact = builder.finish().expect("finish hnsw");
        assert_eq!(artifact.vector_count, HNSW_THRESHOLD as u64);

        // Decode and verify it's an HNSW index
        let index = VecIndex::decode(&artifact.bytes).expect("decode");
        assert!(
            matches!(index, VecIndex::Hnsw(_)),
            "Expected HNSW index for {} vectors",
            HNSW_THRESHOLD
        );
    }

    /// Test that brute force is used for indices below threshold
    #[test]
    #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
    fn below_threshold_uses_brute_force() {
        use super::HNSW_THRESHOLD;

        // Create index with fewer than HNSW_THRESHOLD vectors
        let mut builder = VecIndexBuilder::new();
        let count = HNSW_THRESHOLD - 1;
        let dim = 32;
        for i in 0..count {
            let embedding: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 / 1000.0).collect();
            builder.add_document(i as FrameId, embedding);
        }

        let artifact = builder.finish().expect("finish brute force");
        assert_eq!(artifact.vector_count, count as u64);

        // Decode and verify it's NOT an HNSW index
        let index = VecIndex::decode(&artifact.bytes).expect("decode");
        assert!(
            matches!(index, VecIndex::Uncompressed { .. }),
            "Expected Uncompressed index for {} vectors",
            count
        );
    }

    /// Test HNSW search returns correct nearest neighbors
    #[test]
    #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
    fn hnsw_search_finds_nearest_neighbors() {
        use super::HNSW_THRESHOLD;

        let mut builder = VecIndexBuilder::new();
        let dim = 32;

        // Insert HNSW_THRESHOLD vectors with predictable embeddings
        for i in 0..HNSW_THRESHOLD {
            let embedding: Vec<f32> = (0..dim).map(|_| i as f32).collect();
            builder.add_document(i as FrameId, embedding);
        }

        let artifact = builder.finish().expect("finish");
        let index = VecIndex::decode(&artifact.bytes).expect("decode");

        // Query with a vector identical to frame_id=500
        let query: Vec<f32> = (0..dim).map(|_| 500.0_f32).collect();
        let hits = index.search(&query, 5);

        assert!(!hits.is_empty(), "Should find at least one hit");
        assert_eq!(
            hits[0].frame_id, 500,
            "Nearest neighbor should be exact match"
        );
        assert!(
            hits[0].distance < 0.001,
            "Distance to exact match should be near zero"
        );
    }

    /// Test HNSW serialization/deserialization roundtrip
    #[test]
    #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
    fn hnsw_serialization_roundtrip() {
        use super::HNSW_THRESHOLD;

        let mut builder = VecIndexBuilder::new();
        let dim = 64;

        for i in 0..HNSW_THRESHOLD {
            let embedding: Vec<f32> = (0..dim).map(|j| ((i + j) % 100) as f32 / 100.0).collect();
            builder.add_document(i as FrameId, embedding);
        }

        let artifact = builder.finish().expect("finish");
        let original_bytes = artifact.bytes.clone();

        // Decode
        let index = VecIndex::decode(&original_bytes).expect("decode");
        assert!(matches!(index, VecIndex::Hnsw(_)));

        // Search before any re-serialization
        let query: Vec<f32> = (0..dim).map(|j| (j % 100) as f32 / 100.0).collect();
        let hits_1 = index.search(&query, 10);

        // Decode again from same bytes (simulates loading from disk)
        let index_2 = VecIndex::decode(&original_bytes).expect("decode again");
        let hits_2 = index_2.search(&query, 10);

        // Results should be identical
        assert_eq!(hits_1.len(), hits_2.len());
        for (h1, h2) in hits_1.iter().zip(hits_2.iter()) {
            assert_eq!(h1.frame_id, h2.frame_id);
            assert!((h1.distance - h2.distance).abs() < 1e-6);
        }
    }

    /// Test HNSW with larger dataset to verify approximate search quality
    #[test]
    #[cfg(any(feature = "vec", feature = "hnsw_bench"))]
    fn hnsw_recall_quality() {
        use super::HNSW_THRESHOLD;

        let count = HNSW_THRESHOLD + 500; // 1500 vectors
        let dim = 32;

        // Build HNSW index
        let mut builder = VecIndexBuilder::new();
        let embeddings: Vec<Vec<f32>> = (0..count)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 7 + j * 13) % 1000) as f32 / 1000.0)
                    .collect()
            })
            .collect();

        for (i, emb) in embeddings.iter().enumerate() {
            builder.add_document(i as FrameId, emb.clone());
        }

        let artifact = builder.finish().expect("finish");
        let hnsw_index = VecIndex::decode(&artifact.bytes).expect("decode");

        // Also build brute force index for ground truth
        let brute_index = VecIndex::Uncompressed {
            documents: embeddings
                .iter()
                .enumerate()
                .map(|(i, emb)| VecDocument {
                    frame_id: i as FrameId,
                    embedding: emb.clone(),
                })
                .collect(),
        };

        // Query with vector similar to index 750
        let query = embeddings[750].clone();
        let k = 10;

        let hnsw_hits = hnsw_index.search(&query, k);
        let brute_hits = brute_index.search(&query, k);

        // HNSW should find the exact match first
        assert_eq!(hnsw_hits[0].frame_id, 750, "HNSW should find exact match");
        assert_eq!(
            brute_hits[0].frame_id, 750,
            "Brute force should find exact match"
        );

        // Calculate recall: how many of top-k from HNSW are in top-k from brute force
        let brute_set: std::collections::HashSet<_> =
            brute_hits.iter().map(|h| h.frame_id).collect();
        let recall = hnsw_hits
            .iter()
            .filter(|h| brute_set.contains(&h.frame_id))
            .count();
        let recall_ratio = recall as f32 / k as f32;

        // HNSW should achieve at least 80% recall on this simple dataset
        assert!(
            recall_ratio >= 0.8,
            "HNSW recall {} should be >= 0.8",
            recall_ratio
        );
    }
}
