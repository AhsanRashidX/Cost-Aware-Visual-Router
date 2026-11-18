"""Artifact manifest: every table and figure of the paper -> the function that writes it -> the result files it reads ->
the scripts that produce those files (third review round, R6: archive with a manifest).

Static analysis of the generators (no result is recomputed): make_outputs.py functions are scanned for table(...)
calls, figure saves and load(...)/J(...) reads; every revision script is scanned for the result files it writes.
Tables and figures that the paper's .tex files actually include are marked, with the section that includes them.
Output: ARTIFACT_MANIFEST.json and ARTIFACT_MANIFEST.md (next to this script)
"""
import ast
import json
import re
from pathlib import Path

from common import ROOT, _git_commit

GEN = ["make_outputs.py", "make_method_figure.py", "make_training_figure.py"]
TEXT = ROOT / "paper_out" / "text"


def calls(fn, name):
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == name:
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                out.append(node.args[0].value)
    return out


def strings(fn, pattern):
    return sorted({n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and re.fullmatch(pattern, n.value)})


def producers():
    """result file name -> scripts that write it: write_json / np.savez / torch.save / .write_text on a path that names
    the file, or open(<path naming the file>, "w" | "a")."""
    name = r"[\"']([\w./-]+\.(?:json|jsonl|npz|pt))[\"']"
    pats = [rf"(?:write_json|np\.savez(?:_compressed)?)\(\s*[^,\n]*?{name}",
            rf"torch\.save\([^\n]*?,\s*[^\n]*?{name}",
            rf"{name}\)?\s*\)?\.write_text\(",
            rf"open\(\s*[^,\n]*?{name}[^,\n]*,\s*[\"'][wa]"]
    prod = {}
    for py in sorted(ROOT.glob("*.py")) + sorted((ROOT / "t4").glob("*.py")):
        src = py.read_text(encoding="utf-8", errors="replace")
        for pat in pats:
            for m in re.finditer(pat, src):
                prod.setdefault(Path(m.group(1)).name, set()).add(py.relative_to(ROOT).as_posix())
    # second pass, for outputs written through a variable (OUT = RESULTS / "x.json"; write_json(OUT, ...)): scripts
    # that name the file and write results, excluding the paper generators and the review analyses, which only read
    consumers = re.compile(r"(make_|paper_sections|review_round|vllm_analysis|controls_analysis)")
    for py in sorted(ROOT.glob("*.py")):
        if consumers.match(py.name):
            continue
        src = py.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"write_json|np\.savez|torch\.save", src):
            continue
        for m in re.finditer(r"[\"']([\w./-]+\.(?:json|jsonl|npz))[\"']", src):
            n = Path(m.group(1)).name
            if n not in prod and re.search(rf"[A-Z_]+\s*=\s*[^\n]*[\"']{re.escape(m.group(1))}[\"']|default=[^\n]*{re.escape(m.group(1))}", src):
                prod.setdefault(n, set()).add(py.name)
    return {k: sorted(v) for k, v in prod.items()}


def module_files(tree):
    """module-level constants -> the result file names they hold (e.g. NODES in make_outputs.py)"""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            files = [n.value for n in ast.walk(node.value) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                     and re.fullmatch(r"[\w./-]+\.(?:json|jsonl|npz)", n.value)]
            for nm in names:
                if files:
                    out[nm] = files
    return out


def main():
    prod = producers()
    included = {}
    for tex in sorted(TEXT.glob("*.tex")):
        t = tex.read_text(encoding="utf-8")
        for m in re.finditer(r"\\input\{tables/([^}]+)\}", t):
            included[("table", m.group(1))] = tex.stem
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}.]+)\.(?:pdf|png)\}", t):
            included[("figure", m.group(1))] = tex.stem
    items = []
    for g in GEN:
        tree = ast.parse((ROOT / g).read_text(encoding="utf-8"))
        consts = module_files(tree)
        for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
            reads = set(calls(fn, "load") + calls(fn, "J")) | set(strings(fn, r"[\w./-]+\.(?:json|jsonl|npz)"))
            reads |= {f for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id in consts for f in consts[n.id]}
            if "ev" in [a.arg for a in fn.args.args]:      # main() passes load("eval_main.json") as ev
                reads.add("eval_main.json")
            reads = sorted(reads)
            outs = [("table", t) for t in calls(fn, "table")]
            outs += [("figure", f) for f in {re.sub(r"\.(pdf|png)$", "", s) for s in strings(fn, r"[\w-]+\.(?:pdf|png)")}]
            for kind, name in outs:
                items.append({"kind": kind, "name": name, "in_paper": included.get((kind, name)),
                              "generator": f"{g}:{fn.name}", "reads": reads,
                              "produced_by": {r: prod.get(Path(r).name, []) for r in reads}})
    items.sort(key=lambda x: (x["in_paper"] is None, x["kind"], x["name"]))
    man = {"git_commit": _git_commit(), "regenerate": "python make_outputs.py; python make_method_figure.py; "
           "python make_training_figure.py; python make_paper_text.py; python make_bib.py",
           "artifacts": items}
    (ROOT / "ARTIFACT_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    lines = ["# Artifact manifest", "",
             "Generated by `make_manifest.py`. Every table and figure of the paper, the function that writes it, the "
             "result files it reads (under `results/`) and the scripts that produce those files. Regenerate all "
             f"outputs with: `{man['regenerate']}`.", "",
             "| Kind | Name | Section | Generator | Reads | Produced by |", "|---|---|---|---|---|---|"]
    for it in items:
        if it["in_paper"] is None:
            continue
        prods = sorted({s for v in it["produced_by"].values() for s in v})
        lines.append(f"| {it['kind']} | `{it['name']}` | {it['in_paper']} | `{it['generator']}` | "
                     f"{', '.join('`' + r + '`' for r in it['reads']) or '--'} | {', '.join('`' + p + '`' for p in prods) or '--'} |")
    (ROOT / "ARTIFACT_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_in = sum(it["in_paper"] is not None for it in items)
    missing = [k for k in included if not any((it["kind"], it["name"]) == k for it in items)]
    print(f"{len(items)} outputs, {n_in} in the paper; included but not mapped: {missing or 'none'}")


if __name__ == "__main__":
    main()
