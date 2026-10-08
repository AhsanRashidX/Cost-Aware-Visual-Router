"""Assemble the folder to copy to the multi-GPU server: scripts, README and the two exported data files.

Run export_colpali_queries.py first. Output: p40_package/ (about 0.5 GB; git-ignored).
"""
import shutil

from common import DATA, ROOT

OUT = ROOT / "p40_package"


def main():
    OUT.mkdir(exist_ok=True)
    for f in ("check_env.py", "shard_bench.py", "README_P40.md"):
        shutil.copy2(ROOT / "p40" / f, OUT / f)
    for f in ("colpali_index.pt", "colpali_query_emb.pt"):
        src = DATA / f
        if not src.exists():
            raise SystemExit(f"missing {src}; run export_colpali_queries.py first")
        shutil.copy2(src, OUT / f)
    total = sum(p.stat().st_size for p in OUT.iterdir()) / 2**20
    print(f"wrote {OUT} ({total:.0f} MB): " + ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
