"""Controls requested in the ARS panel review, computed from the cached results only (no new runs).

  blind_mix        a quality-blind random mix of always-visual and always-hybrid, per seed: its expected ANLS at
                   CAVR-CE's GPU cost, and its GPU cost at CAVR-CE's ANLS (expectations are linear in the mix)
  escalation_vs_lambda0  the highlighted escalation (ColPali margin, re-route, tau 0.5) against CAVR-CE at lambda = 0
  mcnemar_power    discordant exact-match pairs of CAVR-CE vs always-hybrid per seed
  split_coverage   how many questions the five overlapping test splits cover, and how many are tested more than once
  capacity_1gpu    always-hybrid single-GPU capacity at p95 <= 2 s per seed (0 = below the first grid step)
Inputs: results/eval_main.json, escalation.json, load_test.json, router_runs/seed*.npz
Output: results/review_analyses.json
"""
import json

import numpy as np

from common import RESULTS, write_json


def J(name):
    return json.load(open(RESULTS / name, encoding="utf-8"))


def stats(xs):
    xs = [float(x) for x in xs]
    return {"mean": float(np.mean(xs)), "min": min(xs), "max": max(xs), "per_seed": xs}


def main():
    ev, es, lt = J("eval_main.json"), J("escalation.json"), J("load_test.json")

    mix = {"anls_at_ce_cost": [], "ce_anls": [], "ce_minus_mix_anls": [], "gpu_at_ce_anls": [], "ce_gpu": [],
           "saving_vs_mix": [], "saving_vs_hybrid": []}
    for s in ev["per_seed"]:
        P = s["policies"]
        v, h, c = P["always_visual"], P["always_hybrid"], P["cavr_ce_valsel"]
        f_cost = (c["gpu_s"] - v["gpu_s"]) / (h["gpu_s"] - v["gpu_s"])
        f_anls = (c["anls"] - v["anls"]) / (h["anls"] - v["anls"])
        a_mix = v["anls"] + f_cost * (h["anls"] - v["anls"])
        g_mix = v["gpu_s"] + f_anls * (h["gpu_s"] - v["gpu_s"])
        mix["anls_at_ce_cost"].append(a_mix)
        mix["ce_anls"].append(c["anls"])
        mix["ce_minus_mix_anls"].append(c["anls"] - a_mix)
        mix["gpu_at_ce_anls"].append(g_mix)
        mix["ce_gpu"].append(c["gpu_s"])
        mix["saving_vs_mix"].append(1 - c["gpu_s"] / g_mix)
        mix["saving_vs_hybrid"].append(1 - c["gpu_s"] / h["gpu_s"])
    blind_mix = {k: stats(v) for k, v in mix.items()}

    lam0 = [s["lambda_sweep"]["0.0"]["cavr_ce"] for s in ev["per_seed"]]
    escs = [p["results_ce"]["colpali_margin"]["reroute"]["0.5"] for p in es["per_seed"]]
    esc_vs = {"lambda0_anls": stats([x["anls"] for x in lam0]), "lambda0_gpu_s": stats([x["gpu_s"] for x in lam0]),
              "escalation_anls": stats([x["anls"] for x in escs]), "escalation_gpu_s": stats([x["gpu_s"] for x in escs]),
              "selection": "signal and tau were highlighted after inspecting test results (exploratory)"}

    disc = [s["tests_cavr_ce_valsel"]["always_hybrid"]["mcnemar_em"] for s in ev["per_seed"]]
    power = {"discordant_pairs": stats([d["cavr_only_correct"] + d["other_only_correct"] for d in disc]),
             "hybrid_only_correct": [d["other_only_correct"] for d in disc],
             "ce_only_correct": [d["cavr_only_correct"] for d in disc]}

    tested = {}
    for i in range(len(ev["per_seed"])):
        d = np.load(RESULTS / "router_runs" / f"seed{i}.npz", allow_pickle=True)
        for q in d["qids"][d["split"] == "test"]:
            tested[str(q)] = tested.get(str(q), 0) + 1
        n_all = len(d["qids"])
    coverage = {"questions": n_all, "tested_at_least_once": len(tested),
                "tested_more_than_once": sum(v > 1 for v in tested.values()),
                "never_tested": n_all - len(tested)}

    cap = lt["seeds_summary"]["policies"]["always_hybrid"]["max_rate_1gpu"]
    capacity = {"always_hybrid_per_seed": cap["per_seed"], "always_hybrid_max": max(cap["per_seed"]),
                "seeds_below_first_step": sum(x == 0 for x in cap["per_seed"])}

    out = {"blind_mix": blind_mix, "escalation_vs_lambda0": esc_vs, "mcnemar_power": power,
           "split_coverage": coverage, "capacity_1gpu": capacity}
    write_json(RESULTS / "review_analyses.json", out)
    b = blind_mix
    print(f"blind mix: CE {b['ce_anls']['mean']:.4f} vs mix {b['anls_at_ce_cost']['mean']:.4f} at equal cost; "
          f"saving vs mix {100 * b['saving_vs_mix']['mean']:.1f}% ({100 * b['saving_vs_mix']['min']:.0f}-"
          f"{100 * b['saving_vs_mix']['max']:.0f}%), vs hybrid {100 * b['saving_vs_hybrid']['mean']:.1f}%")
    print("escalation vs lambda0:", {k: round(v["mean"], 4) for k, v in esc_vs.items() if isinstance(v, dict)})
    print("discordant pairs:", power["discordant_pairs"]["per_seed"], "coverage:", coverage, "capacity:", capacity)


if __name__ == "__main__":
    main()
