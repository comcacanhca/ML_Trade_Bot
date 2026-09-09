from __future__ import annotations

import argparse

from research_random50_initial_features import build_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cols = build_cache(force=args.force)
    print({"phase": "cache_built", "n_candidate_columns": len(cols)}, flush=True)


if __name__ == "__main__":
    main()
