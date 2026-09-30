import json

def show(path, label):
    with open(path) as f:
        d = json.load(f)
    n_err = d["n_errors"]
    n_bonds = d["n_true_bonds_evaluated"]
    rate = d["error_rate"] * 100
    print("=== " + label + " ===")
    print("  errors: " + str(n_err) + " / " + str(n_bonds) + "  (" + str(round(rate, 5)) + "%)")
    for k, v in sorted(d["errors_by_transition"].items(), key=lambda x: -x[1]):
        print("    " + k + ": " + str(v))
    print()

show("results/valence_rule_test/error_analysis_sigma_0.000.json", "baseline (no rules)")
show("results/valence_rule_ON/error_analysis_sigma_0.000.json", "valence v1 (bug: aromatic downgraded)")
show("results/valence_rule_v2/error_analysis_sigma_0.000.json", "valence v2 (skip aromatic atoms)")
