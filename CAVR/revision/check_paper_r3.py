"""Build checks for the third review round: cross-reference consistency and statements that must not be lost.

Consistency (R4.7): every \\ref has a \\label; every table and figure label is referenced; every \\input and
\\includegraphics target exists; every \\cite key is in references.bib; no unfilled <<token>> or tab character; the
repository link appears exactly once; retired wording ("frozen router", "lower bound" for GPU counts) is gone.
Preserved statements: the always-visual comparison, the non-significant held-out result, the possible training
exposure of the held-out set, and the measured / simulated / projected labels must stay in the abstract, results and
conclusion. Exit code 1 if any check fails.
Usage: python check_paper_r3.py
"""
import re
import sys

from common import ROOT

OUT = ROOT / "paper_out"
SECTIONS = ["abstract", "introduction", "related", "method", "setup", "results_section", "conclusion", "appendix"]

PRESERVE = {
    "abstract": [r"page image alone", r"not significant", r"saw in training", r"simulator", r"projections",
                 r"[Ss]erved\s+live"],
    "results_section": [r"always-visual", r"not significant|includes zero", r"exposure", r"re-trained",
                        r"simulated projection", r"negative result"],
    "conclusion": [r"always-visual|page image alone", r"training-split questions", r"projections",
                   r"no multi-node experiment"],
    "introduction": [r"\\emph\{Scope\.\}", r"projected", r"not measured"],
}
RETIRED = [r"frozen router", r"\bfrozen\b(?! *[,.]? *not)", r"lower bound on the GPU", r"high-performance computing problem"]


def main():
    tex = {s: (OUT / "text" / f"{s}.tex").read_text(encoding="utf-8") for s in SECTIONS if (OUT / "text" / f"{s}.tex").exists()}
    tables = {p.stem: p.read_text(encoding="utf-8") for p in (OUT / "tables").glob("*.tex")}
    used_tables = {m for t in tex.values() for m in re.findall(r"\\input\{tables/([^}]+)\}", t)}
    corpus = "\n".join(tex.values()) + "\n".join(tables[t] for t in used_tables if t in tables)
    bad = []
    labels = set(re.findall(r"\\label\{([^}]+)\}", corpus))
    refs = {r for grp in re.findall(r"\\ref\{([^}]+)\}", corpus) for r in grp.split(",")}
    bad += [f"\\ref without \\label: {r}" for r in sorted(refs - labels)]
    bad += [f"table/figure never referenced: {l}" for l in sorted(labels - refs) if l.startswith(("tab:", "fig:"))]
    bad += [f"missing table file: {t}" for t in sorted(used_tables - set(tables))]
    for t in tex.values():
        for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t):
            if not (OUT / f).exists():
                bad.append(f"missing figure file: {f}")
    bib = (OUT / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cites = {k.strip() for grp in re.findall(r"\\cite[pt]?\{([^}]+)\}", corpus) for k in grp.split(",")}
    bad += [f"\\cite key not in references.bib: {k}" for k in sorted(cites - keys)]
    if "<<" in corpus:
        bad.append("unfilled <<token>>")
    if "\t" in corpus:
        bad.append("tab character (a mangled \\t escape?)")
    n_url = corpus.count("github.com/AhsanRashidX/Cost-Aware-Visual-Router")
    if n_url != 1:
        bad.append(f"repository link appears {n_url} times (expected 1)")
    for s, pats in PRESERVE.items():
        for pat in pats:
            if not re.search(pat, tex.get(s, "")):
                bad.append(f"preserved statement missing from {s}: /{pat}/")
    for pat in RETIRED:
        for s, t in tex.items():
            for m in re.finditer(pat, t):
                ctx = t[max(0, m.start() - 40):m.end() + 20].replace("\n", " ")
                if "not frozen" not in ctx:
                    bad.append(f"retired wording in {s}: ...{ctx}...")
    print(f"{len(labels)} labels, {len(refs)} refs, {len(cites)} cite keys, {len(used_tables)} tables")
    for b in bad:
        print("FAIL", b)
    print("all checks passed" if not bad else f"{len(bad)} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
