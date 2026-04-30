"""
Section 6 Runner — CLI Orchestrator
====================================
Coordinates the five-stage Section 6 pipeline:

  Stage A  verify     Data integrity check (~10s)
  Stage B  evidence   Evidence build (~25h at 3 workers)
  Stage C  materialise Preset filtering (~40s per preset)
  Stage D  analyse    Comparative metrics + stats (~2min)
  Stage E  export     Paper-ready JSON (part of analyse)

Usage:
  python section6_run.py --stage all
  python section6_run.py --stage evidence --workers 3
  python section6_run.py --stage materialise
  python section6_run.py --stage materialise --preset all_gates
  python section6_run.py --stage analyse
  python section6_run.py --stage verify
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ARTIFACT_RUNTIME = (REPO_ROOT / "runtime").resolve()
if str(ARTIFACT_RUNTIME) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_RUNTIME))

# Add Paper directory for section6_* imports
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from section6_manifest import PRESET_ORDER, STUDY_ID, now_iso
from section6_evidence_builder import EvidenceBuilder
from section6_preset_materializer import materialise_all_presets, PresetMaterializer
from section6_analysis import Section6Analysis

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "empirical_results_v2"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quiet noisy libraries
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_verify(frozen_dir: Path, output_dir: Path, **kwargs) -> None:
    """Stage A: verify frozen dataset integrity."""
    builder = EvidenceBuilder(frozen_dir, output_dir / "evidence_build")
    result = builder.verify_frozen_data()
    if result.get("passed"):
        print(f"PASS — dataset '{result.get('manifest_id', '?')}' verified")
    else:
        print("FAIL — dataset verification failed")
        sys.exit(1)


def run_evidence(
    frozen_dir: Path,
    output_dir: Path,
    workers: int = 3,
    resume: bool = True,
    config_path: Path = None,
    **kwargs,
) -> None:
    """Stage B: build evidence."""
    evidence_dir = output_dir / "evidence_build"
    builder = EvidenceBuilder(
        frozen_dir=frozen_dir,
        output_dir=evidence_dir,
        config_path=config_path,
        workers=workers,
        resume=resume,
    )
    t0 = time.time()
    manifest = builder.build_evidence()
    elapsed = time.time() - t0

    print(f"\nEvidence build complete:")
    print(f"  Slots: {manifest.completed_slots}/{manifest.total_slots}")
    print(f"  Failed: {len(manifest.failed_slots)}")
    print(f"  Duration: {elapsed/3600:.1f} hours")
    print(f"  Cache key: {manifest.cache_key[:16]}")

    if manifest.failed_slots:
        print(f"\nFailed slots:")
        for fs in manifest.failed_slots[:10]:
            print(f"  {fs['slot']}: {fs['error'][:80]}")


def run_materialise(
    output_dir: Path,
    preset: str = "",
    **kwargs,
) -> None:
    """Stage C: apply gate presets over frozen evidence."""
    evidence_dir = output_dir / "evidence_build"
    presets_dir = output_dir / "presets"

    if not (evidence_dir / "build_manifest.json").exists():
        print("ERROR: No evidence build found. Run --stage evidence first.")
        sys.exit(1)

    t0 = time.time()

    if preset:
        # Single preset
        from pm_research import ABLATION_PRESETS  # type: ignore
        if preset not in ABLATION_PRESETS:
            print(f"ERROR: Unknown preset '{preset}'")
            print(f"Available: {', '.join(PRESET_ORDER)}")
            sys.exit(1)
        gc = ABLATION_PRESETS[preset]
        m = PresetMaterializer(evidence_dir, presets_dir / preset, gc)
        result = m.materialise()
        print(f"Preset '{preset}': {result.filled_slots}/{result.total_slots} filled")
    else:
        # All presets
        results = materialise_all_presets(evidence_dir, presets_dir)
        print(f"\nMaterialisation complete ({time.time()-t0:.1f}s):")
        for name, manifest in results.items():
            print(f"  {name:25s}  {manifest.filled_slots:4d}/{manifest.total_slots} filled")


def run_analyse(output_dir: Path, **kwargs) -> None:
    """Stage D+E: comparative analysis and paper export."""
    presets_dir = output_dir / "presets"
    analysis_dir = output_dir / "analysis"

    if not presets_dir.exists():
        print("ERROR: No preset outputs found. Run --stage materialise first.")
        sys.exit(1)

    analyser = Section6Analysis(presets_dir, analysis_dir)
    t0 = time.time()
    result = analyser.run_analysis()
    elapsed = time.time() - t0

    print(f"\nAnalysis complete ({elapsed:.1f}s):")
    print(f"  Output: {analysis_dir}")

    # Quick summary
    fill_rates = result.get("fill_rates", [])
    if fill_rates:
        print(f"\n  {'Preset':25s} {'Fill Rate':>10s}")
        print(f"  {'-'*37}")
        for fr in fill_rates:
            print(f"  {fr['preset']:25s} {fr['fill_rate_pct']:>8.1f}%")


def run_all(
    frozen_dir: Path,
    output_dir: Path,
    workers: int = 3,
    resume: bool = True,
    config_path: Path = None,
    **kwargs,
) -> None:
    """Run all stages end-to-end."""
    print(f"{'='*60}")
    print(f"  Section 6 Full Pipeline — {STUDY_ID}")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {workers}")
    print(f"  Started: {now_iso()}")
    print(f"{'='*60}\n")

    total_t0 = time.time()

    print("[Stage A] Verifying frozen dataset...")
    run_verify(frozen_dir, output_dir)

    print("\n[Stage B] Building evidence...")
    run_evidence(frozen_dir, output_dir, workers, resume, config_path)

    print("\n[Stage C] Materialising presets...")
    run_materialise(output_dir)

    print("\n[Stage D+E] Running analysis...")
    run_analyse(output_dir)

    total = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  All stages complete in {total/3600:.1f} hours")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Frozen data auto-discovery
# ---------------------------------------------------------------------------

def find_frozen_dir(output_dir: Path, explicit: str = "") -> Path:
    """Locate the frozen dataset directory.

    Search order:
      1. Explicit --data-dir argument
      2. {output_dir}/frozen_dataset/
      3. Repository data/frozen_dataset/
    """
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified data dir not found: {p}")

    # Check output root
    candidate = output_dir / "frozen_dataset"
    if candidate.exists() and (candidate / "MANIFEST.json").exists():
        return candidate

    # Public repo layout. The repository includes the manifest and frozen CSVs.
    candidate = REPO_ROOT / "data" / "frozen_dataset"
    if candidate.exists() and (candidate / "MANIFEST.json").exists():
        return candidate

    raise FileNotFoundError(
        "Could not find frozen dataset. Use --data-dir to specify location."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Section 6 Empirical Ablation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  verify      Stage A: check frozen data integrity
  evidence    Stage B: build candidate evidence (~25h)
  materialise Stage C: apply gate presets (~40s each)
  analyse     Stage D+E: comparative metrics + paper export
  all         Run everything end-to-end
""",
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["verify", "evidence", "materialise", "analyse", "all"],
        help="Which stage(s) to run",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="Parallel workers for evidence build (default: 3)",
    )
    parser.add_argument(
        "--output-root", type=str, default="",
        help=f"Base output directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--data-dir", type=str, default="",
        help="Frozen dataset directory (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--config", type=str, default="",
        help="Pipeline config.json path (default: bundled runtime config)",
    )
    parser.add_argument(
        "--preset", type=str, default="",
        help="Materialise only this preset (with --stage materialise)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming interrupted builds",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would run without executing",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    output_dir = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config) if args.config else None

    if args.dry_run:
        print(f"[DRY RUN] Stage: {args.stage}")
        print(f"[DRY RUN] Output: {output_dir}")
        print(f"[DRY RUN] Workers: {args.workers}")
        return

    try:
        frozen_dir = find_frozen_dir(output_dir, args.data_dir)
    except FileNotFoundError as exc:
        if args.stage in ("materialise", "analyse"):
            frozen_dir = Path(".")  # Not needed for these stages
        else:
            print(f"ERROR: {exc}")
            sys.exit(1)

    dispatch = {
        "verify": run_verify,
        "evidence": run_evidence,
        "materialise": run_materialise,
        "analyse": run_analyse,
        "all": run_all,
    }

    fn = dispatch[args.stage]
    fn(
        frozen_dir=frozen_dir,
        output_dir=output_dir,
        workers=args.workers,
        resume=not args.no_resume,
        config_path=config_path,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
