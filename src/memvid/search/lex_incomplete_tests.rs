//! Unit tests for the typed lexical-index fault (`LexIndexFault`,
//! `MemvidError::LexIndexIncomplete`) and for the two decision tables that decide
//! when it fires.
//!
//! These cover *what the fault says*, not the end-to-end scenarios — those are in
//! [`lex_integrity_tests`](super::lex_integrity_tests).

use crate::error::{LexIndexFault, MemvidError};
use crate::memvid::lifecycle::{lex_commit_fault, lex_rebuild_required};

/// Rebuild the variant from its parts, so a change to the field list is a
/// compile error here rather than a silently weakened assertion.
fn incomplete(
    expected_docs: Option<u64>,
    actual_docs: u64,
    segments: u32,
    fault: LexIndexFault,
) -> MemvidError {
    MemvidError::LexIndexIncomplete {
        expected_docs,
        actual_docs,
        segments,
        fault,
    }
}

/// Read the four fields back out of the variant.
fn fields(err: &MemvidError) -> (Option<u64>, u64, u32, LexIndexFault) {
    match err {
        MemvidError::LexIndexIncomplete {
            expected_docs,
            actual_docs,
            segments,
            fault,
        } => (*expected_docs, *actual_docs, *segments, *fault),
        other => panic!("expected LexIndexIncomplete, got {other:?}"),
    }
}

/// Every fault carries all four fields through construction unchanged.
#[test]
fn every_fault_preserves_its_fields() {
    let faults = [
        LexIndexFault::OpenFailed,
        LexIndexFault::CountDiverged,
        LexIndexFault::PutTimeSkipped,
    ];
    for (i, fault) in faults.into_iter().enumerate() {
        let expected = Some(7 + i as u64);
        let actual = 3 + i as u64;
        let segments = 2 + i as u32;
        let built = incomplete(expected, actual, segments, fault);
        assert_eq!(
            fields(&built),
            (expected, actual, segments, fault),
            "fields changed through construction for {fault:?}"
        );
    }
}

/// `expected_docs` is `Option` because absent is real information. `None` and
/// `Some(0)` are different facts and must stay distinguishable on the variant.
#[test]
fn absent_expectation_is_distinct_from_a_zero_expectation() {
    let absent = incomplete(None, 0, 4, LexIndexFault::PutTimeSkipped);
    let zero = incomplete(Some(0), 0, 4, LexIndexFault::PutTimeSkipped);
    assert_eq!(fields(&absent).0, None);
    assert_eq!(fields(&zero).0, Some(0));
    assert_ne!(
        fields(&absent),
        fields(&zero),
        "None must not collapse into Some(0)"
    );
}

/// The three faults are distinct values, and so are their wire names: a caller
/// branching on the fault cannot conflate them.
#[test]
fn faults_are_distinct_values_and_names() {
    let faults = [
        LexIndexFault::OpenFailed,
        LexIndexFault::CountDiverged,
        LexIndexFault::PutTimeSkipped,
    ];
    for (i, a) in faults.into_iter().enumerate() {
        for (j, b) in faults.into_iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "{a:?} and {b:?} must not compare equal");
                assert_ne!(a.to_string(), b.to_string());
                assert_ne!(
                    serde_json::to_string(&a).unwrap(),
                    serde_json::to_string(&b).unwrap()
                );
            }
        }
    }
}

/// Display snapshot per fault: short, stable, field-derived. There is no prose
/// bag — the numbers are the message.
#[test]
fn display_snapshot_per_fault() {
    let cases = [
        (
            LexIndexFault::OpenFailed,
            incomplete(Some(3), 0, 2, LexIndexFault::OpenFailed),
            "lexical index incomplete: 2 advertised segment(s) hold 0 reachable document(s), expected 3",
        ),
        (
            LexIndexFault::CountDiverged,
            incomplete(Some(2), 1, 1, LexIndexFault::CountDiverged),
            "lexical index incomplete: 1 advertised segment(s) hold 1 reachable document(s), expected 2",
        ),
        (
            LexIndexFault::PutTimeSkipped,
            incomplete(None, 0, 4, LexIndexFault::PutTimeSkipped),
            "lexical index incomplete: 4 advertised segment(s) hold 0 reachable document(s), expected unknown",
        ),
    ];
    for (fault, err, expected) in cases {
        assert_eq!(err.to_string(), expected, "Display drifted for {fault:?}");
    }
}

/// Display is composed only from the four fields: the fault name never leaks into
/// it, and identical numbers produce identical text for every fault. A caller
/// that wants to distinguish the faults reads `fault`, not a sentence.
#[test]
fn display_carries_no_prose_beyond_the_fields() {
    let numbers_only = |fault: LexIndexFault| {
        incomplete(Some(5), 5, 3, fault)
            .to_string()
            .split(", expected ")
            .next()
            .unwrap_or_default()
            .to_owned()
    };
    let first = numbers_only(LexIndexFault::OpenFailed);
    for fault in [LexIndexFault::CountDiverged, LexIndexFault::PutTimeSkipped] {
        assert_eq!(numbers_only(fault), first);
    }
    let all = incomplete(Some(5), 5, 3, LexIndexFault::OpenFailed).to_string();
    for leaked in ["failed to open", "diverged", "never indexed"] {
        assert!(
            !all.contains(leaked),
            "Display leaked fault prose {leaked:?} into {all:?}"
        );
    }
    assert!(all.starts_with("lexical index incomplete: "));
}

/// Each fault round-trips through serde under its stable snake_case name. The
/// wire name is the stable identifier — `code()` does not exist in this crate.
#[test]
fn fault_serde_round_trip_and_wire_names() {
    let cases = [
        (LexIndexFault::OpenFailed, "\"open_failed\""),
        (LexIndexFault::CountDiverged, "\"count_diverged\""),
        (LexIndexFault::PutTimeSkipped, "\"put_time_skipped\""),
    ];
    for (fault, wire) in cases {
        let encoded = serde_json::to_string(&fault).unwrap();
        assert_eq!(encoded, wire, "wire name drifted for {fault:?}");
        let decoded: LexIndexFault = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, fault, "serde round-trip lost {fault:?}");
    }
}

/// The trust rule at open. The base defect is row 2: a catalog that claims three
/// documents while the engine holds none was trusted because segments existed.
#[test]
fn rebuild_required_decision_table() {
    let cases: &[(&str, Option<u64>, u64, bool, bool)] = &[
        ("catalog agrees: trusted", Some(3), 3, true, false),
        ("catalog disagrees: rebuilt", Some(3), 0, true, true),
        (
            "empty engine over a claim: rebuilt",
            Some(0),
            0,
            true,
            false,
        ),
        ("no claim, engine non-empty: trusted", None, 5, true, false),
        ("no claim, empty engine: rebuilt", None, 0, true, true),
        ("nothing advertised: rebuilt", None, 7, false, true),
        ("nothing advertised, empty: rebuilt", None, 0, false, true),
        ("claim without segments: honoured", Some(2), 2, false, false),
    ];
    for (name, expected, actual, advertised, want) in cases {
        assert_eq!(
            lex_rebuild_required(*expected, *actual, *advertised),
            *want,
            "{name}: expected/actual/advertised = {expected:?}/{actual}/{advertised}"
        );
    }
}

/// The refusal rule. Each row names the rule it pins; the `None` rows are the
/// ones that keep a store with no recorded expectation from being refused.
#[test]
fn commit_fault_decision_table() {
    use LexIndexFault::{CountDiverged, OpenFailed, PutTimeSkipped};
    /// `(name, expected_docs, actual_docs, segments, indexable_frames,
    /// open_failed, expected fault)`
    type Row = (
        &'static str,
        Option<u64>,
        u64,
        u32,
        u64,
        bool,
        Option<LexIndexFault>,
    );
    let cases: &[Row] = &[
        // R6: a store with nothing to index is healthy whatever it advertises.
        ("R6 nothing indexable", Some(9), 0, 5, 0, false, None),
        (
            "R6 nothing indexable, open failed",
            None,
            0,
            5,
            0,
            true,
            None,
        ),
        // Nothing advertised is nothing claimed.
        ("no segments advertised", Some(9), 0, 0, 4, false, None),
        // R7: a recorded failed open is the most specific explanation.
        (
            "R7 open failed, empty engine",
            None,
            0,
            3,
            2,
            true,
            Some(OpenFailed),
        ),
        (
            "R7 open failed outranks claim",
            Some(2),
            0,
            3,
            2,
            true,
            Some(OpenFailed),
        ),
        (
            "R7 open failed but engine non-empty",
            Some(2),
            2,
            3,
            2,
            true,
            None,
        ),
        // R4: segments advertised, engine demonstrably empty.
        (
            "R4 empty engine, no claim",
            None,
            0,
            3,
            2,
            false,
            Some(PutTimeSkipped),
        ),
        // A `Some(0)` claim must not launder an empty engine over indexable frames.
        (
            "R4 empty engine, claim agrees at zero",
            Some(0),
            0,
            3,
            2,
            false,
            Some(PutTimeSkipped),
        ),
        // Total order `OpenFailed` > `PutTimeSkipped` > `CountDiverged`: an empty
        // engine over indexable frames is the more specific diagnosis, so it wins
        // even when the catalog's claim also disagrees with it. The label differs,
        // never the refusal.
        (
            "R4 outranks R3: empty engine, claim disagrees",
            Some(5),
            0,
            3,
            2,
            false,
            Some(PutTimeSkipped),
        ),
        // R3: known disagreement, in both directions.
        (
            "R3 engine behind claim",
            Some(4),
            2,
            3,
            4,
            false,
            Some(CountDiverged),
        ),
        (
            "R3 engine ahead of claim",
            Some(2),
            4,
            3,
            4,
            false,
            Some(CountDiverged),
        ),
        // R2: a missing claim is no opinion, never a refusal.
        ("R2 no claim, engine non-empty", None, 4, 3, 5, false, None),
        ("healthy store", Some(4), 4, 3, 4, false, None),
    ];
    for (name, expected, actual, segments, indexable, open_failed, want) in cases {
        assert_eq!(
            lex_commit_fault(*expected, *actual, *segments, *indexable, *open_failed),
            *want,
            "{name}: expected/actual/segments/indexable/open_failed = \
             {expected:?}/{actual}/{segments}/{indexable}/{open_failed}"
        );
    }
}

/// Guard against the two shapes that would make the gate useless: a rule that
/// refuses every store, and a rule that refuses nothing.
#[test]
fn commit_fault_is_neither_always_on_nor_always_off() {
    let refuses = lex_commit_fault(None, 0, 3, 2, false).is_some();
    let abstains = lex_commit_fault(None, 4, 3, 5, false).is_none();
    let healthy = lex_commit_fault(Some(4), 4, 3, 4, false).is_none();
    assert!(refuses, "the gate never fires");
    assert!(abstains && healthy, "the gate fires on a healthy store");
}
