//! Corpus measurement against a real, pinned checkout.
//!
//! Slice 1's done-condition in
//! `docs/plans/2026-09-05-fleet-cpg-engine-implementation.md` requires the
//! engine to reproduce the counts recorded by the validated Python spike on
//! the pinned zod revision, "or document every divergence with a reason".
//! The unit tests in `src/lib.rs` cover lowering primitives on synthetic
//! input; this covers agreement with the corpus the nine Phase 0–4
//! experiments were actually run against.
//!
//! Ignored by default so `cargo test` stays hermetic and offline. To run:
//!
//! ```text
//! git clone https://github.com/colinhacks/zod && cd zod
//! git checkout 5ff9566508e6c95873d2648a5bdcc3a371f1b757
//! CPG_CORPUS=$PWD/packages/zod/src cargo test -p cpg-lowering-typescript -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Facts recorded by the Python spike at zod `5ff9566`, from
/// `experiments/evidence/fleet-cpg-phase4-second-language/` (TypeScript
/// variant, 3/3 repetitions). The spike lowered three node kinds
/// (function/class/method declarations) and identified every symbol as
/// `file:line:name`.
const SPIKE_FILES: usize = 324;
const SPIKE_DEFS: usize = 1388;
const SPIKE_CALLS: usize = 43888;

fn collect_sources(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(err) => panic!("reading {}: {err}", root.display()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_sources(&path, out);
            continue;
        }
        match path.extension().and_then(|e| e.to_str()) {
            Some("ts") | Some("tsx") => out.push(path),
            _ => {}
        }
    }
}

#[test]
#[ignore = "requires a pinned corpus checkout; set CPG_CORPUS"]
fn agrees_with_recorded_spike_facts() {
    let corpus = std::env::var("CPG_CORPUS")
        .expect("set CPG_CORPUS to a checkout of packages/zod/src at 5ff9566");
    let root = PathBuf::from(&corpus);

    let mut sources = Vec::new();
    collect_sources(&root, &mut sources);
    sources.sort();
    assert!(
        !sources.is_empty(),
        "no .ts/.tsx files under {}",
        root.display()
    );

    let mut defs_total = 0usize;
    let mut calls_total = 0usize;
    let mut failed: Vec<String> = Vec::new();
    // Caller identity is the diagnostic that matters for Slice 2: the blast
    // radius is a reverse-BFS over callers, so a caller key that is not
    // file-scoped collapses distinct symbols into one graph node.
    let mut callers: BTreeMap<String, usize> = BTreeMap::new();

    for path in &sources {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let source = fs::read_to_string(path).expect("reading source file");
        let tsx = path.extension().and_then(|e| e.to_str()) == Some("tsx");

        match cpg_lowering_typescript::lower(&source, rel.clone(), tsx) {
            Ok((defs, calls)) => {
                defs_total += defs.len();
                calls_total += calls.len();
                for call in &calls {
                    *callers.entry(call.caller.clone()).or_default() += 1;
                }
            }
            Err(err) => failed.push(format!("{rel}: {err}")),
        }
    }

    let module_scoped = callers.get("<module>").copied().unwrap_or(0);

    eprintln!("\n--- cpg-lowering-typescript vs recorded spike facts ---");
    eprintln!("corpus: {}", root.display());
    eprintln!(
        "  files   engine={:<7} spike={:<7} {}",
        sources.len(),
        SPIKE_FILES,
        verdict(sources.len(), SPIKE_FILES)
    );
    eprintln!(
        "  defs    engine={:<7} spike={:<7} {}",
        defs_total,
        SPIKE_DEFS,
        verdict(defs_total, SPIKE_DEFS)
    );
    eprintln!(
        "  calls   engine={:<7} spike={:<7} {}",
        calls_total,
        SPIKE_CALLS,
        verdict(calls_total, SPIKE_CALLS)
    );
    eprintln!("  distinct caller keys: {}", callers.len());
    eprintln!(
        "  calls attributed to the bare \"<module>\" key: {module_scoped} \
         ({:.1}% of all calls)",
        100.0 * module_scoped as f64 / calls_total.max(1) as f64
    );
    if let Some((name, count)) = callers.iter().max_by_key(|(_, c)| **c) {
        eprintln!("  most-attributed caller key: {name:?} with {count} calls");
    }
    eprintln!("  files that failed to lower: {}", failed.len());
    for failure in failed.iter().take(5) {
        eprintln!("    {failure}");
    }
    eprintln!();

    // The corpus must parse. Everything else is measurement, reported above
    // and interpreted in the plan rather than asserted here, because the
    // divergence is a design question and not something a test should decide.
    assert!(
        failed.is_empty(),
        "{} file(s) failed to lower; the engine must at least parse the corpus",
        failed.len()
    );
    assert_eq!(
        sources.len(),
        SPIKE_FILES,
        "corpus file count differs from the recorded spike: wrong revision or path?"
    );
}

fn verdict(engine: usize, spike: usize) -> String {
    if engine == spike {
        return "match".into();
    }
    let delta = engine as i64 - spike as i64;
    format!(
        "DIVERGES by {delta:+} ({:.1}x)",
        engine as f64 / spike.max(1) as f64
    )
}
