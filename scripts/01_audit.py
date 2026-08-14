#!/usr/bin/env python
"""WP1: data integrity audit (reproduces data/bitcoin_otc/README.md numbers)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dengyunetwork.audit import audit  # noqa: E402

EXP_DIR = ROOT / "experiments"


def main() -> None:
    result = audit()
    EXP_DIR.mkdir(exist_ok=True)
    out = EXP_DIR / "audit.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # hard gates from the README audit
    assert result["n_rows"] == 35592, result
    assert result["n_users"] == 5881, result
    assert result["n_positive_ratings"] == 32029, result
    assert result["n_negative_ratings"] == 3563, result
    assert result["n_self_loops"] == 0
    assert result["no_repeated_directed_pair"]
    assert result["n_malformed"] == 0
    print(f"\n[ok] audit passed, wrote {out}")


if __name__ == "__main__":
    main()
