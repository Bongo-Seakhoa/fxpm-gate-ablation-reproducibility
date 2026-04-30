"""
FXPM Artifact Version and Metadata
=================================

Single source of truth for artifact identity, schema versioning,
and environment fingerprinting.

This module is used by:
- Experiment manifests to stamp runs with the exact artifact state
- Configuration validation to detect schema drift
- Research tooling to record provenance

Academic context:
    Author: Bongo Bokoa Kosa (WD42M3)
    Institution: University of Debrecen, Hungary
    Programme: BSc Engineering Management
    Audit reference: academic/audit/fxpm_academic_audit.html

Created: 2026-03-10
Research Log Entry: R-002
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ARTIFACT_NAME = "FXPM"
ARTIFACT_FULL_NAME = "FX Portfolio Manager - Regime-Adaptive Autonomous Decision System"
ARTIFACT_VERSION = "3.3.0"

CONFIG_SCHEMA_VERSION = "1.0.0"
RESEARCH_TRACK_VERSION = "0.1.0"

AUTHOR = "Bongo Bokoa Kosa"
STUDENT_ID = "WD42M3"
INSTITUTION = "University of Debrecen, Debrecen, Hungary"
PROGRAMME = "BSc Engineering Management"


def get_environment_fingerprint() -> Dict[str, Any]:
    packages: Dict[str, str] = {}
    for pkg_name in [
        "numpy",
        "pandas",
        "numba",
        "optuna",
        "reportlab",
        "MetaTrader5",
        "flask",
        "scipy",
        "matplotlib",
    ]:
        try:
            mod = __import__(pkg_name)
            version = getattr(mod, "__version__", getattr(mod, "Version", "unknown"))
            packages[pkg_name] = str(version)
        except ImportError:
            packages[pkg_name] = "not installed"

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).resolve().parent),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def hash_config(config_path: str = "config.json") -> str:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        return "config_not_found"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_artifact_stamp() -> Dict[str, str]:
    return {
        "artifact": ARTIFACT_NAME,
        "version": ARTIFACT_VERSION,
        "schema_version": CONFIG_SCHEMA_VERSION,
        "research_track": RESEARCH_TRACK_VERSION,
        "git_commit": get_git_commit() or "unknown",
        "config_hash": hash_config(),
    }


if __name__ == "__main__":
    print(f"{ARTIFACT_NAME} v{ARTIFACT_VERSION}")
    print(f"Config schema: v{CONFIG_SCHEMA_VERSION}")
    print(f"Research track: v{RESEARCH_TRACK_VERSION}")
    print()
    print(json.dumps(get_artifact_stamp(), indent=2))
    print()
    print(json.dumps(get_environment_fingerprint(), indent=2))
