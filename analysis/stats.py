"""Statistics used for reporting (plan 9.3 / 11): mean + 95% CI, paired tests, Holm correction, effect sizes."""
import math

import numpy as np
from scipy import stats as st


def mean_ci(x, conf=0.95):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    m = float(x.mean())
    if len(x) < 2:
        return m, m, m
    h = float(st.sem(x) * st.t.ppf((1 + conf) / 2, len(x) - 1))
    return m, m - h, m + h


def paired_test(a, b):
    """Paired comparison of scheme a vs b over identical seeds: t-test, Wilcoxon, Cohen's dz (a-b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    out = {"n": int(len(d)), "mean_diff": float(d.mean()) if len(d) else math.nan}
    if len(d) < 2 or np.allclose(d, d[0]):
        out.update(p_t=1.0 if len(d) and np.allclose(d, 0) else math.nan, p_w=math.nan, effect_dz=0.0 if len(d) else math.nan)
        return out
    out["p_t"] = float(st.ttest_rel(a, b).pvalue)
    try:
        out["p_w"] = float(st.wilcoxon(a, b).pvalue)
    except ValueError:
        out["p_w"] = math.nan
    sd = d.std(ddof=1)
    out["effect_dz"] = float(d.mean() / sd) if sd > 0 else 0.0
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (n - rank) * p[i]))
        adj[i] = run
    return adj.tolist()
