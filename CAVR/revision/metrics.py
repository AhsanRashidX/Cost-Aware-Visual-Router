"""Answer-quality metrics: ANLS (DocVQA/InfoVQA standard, tau=0.5) and exact match."""
import re
import unicodedata

ANLS_TAU = 0.5


def normalize(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;\"'")


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred, golds, tau=ANLS_TAU):
    p = normalize(pred)
    best = 0.0
    for g in golds:
        g = normalize(g)
        if not g and not p:
            return 1.0
        nl = levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1 - nl if nl < tau else 0.0)
    return best


def exact_match(pred, golds):
    p = normalize(pred)
    return float(any(p == normalize(g) for g in golds))
