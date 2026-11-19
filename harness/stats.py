"""Estatística fixa do juiz: Mann-Kendall (com correção de empates) e Sen's slope.

Mesmos métodos do artigo de referência (Santos, Andrade, Natella, arXiv:2510.24188).
Implementação só com biblioteca padrão para não depender de scipy.
"""

import math
from collections import Counter


def mann_kendall(y):
    """Retorna (S, Z, p_bilateral, p_alta_unilateral). Requer len(y) >= 10."""
    n = len(y)
    if n < 10:
        return 0, 0.0, 1.0, 1.0
    s = 0
    for k in range(n - 1):
        yk = y[k]
        for j in range(k + 1, n):
            d = y[j] - yk
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    ties = Counter(y).values()
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in ties if t > 1)) / 18.0
    if var <= 0:
        return s, 0.0, 1.0, 1.0
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    p_two = math.erfc(abs(z) / math.sqrt(2))
    p_up = 0.5 * math.erfc(z / math.sqrt(2))
    return s, z, p_two, p_up


def sen_slope(x, y):
    """Mediana das inclinações par a par (unidade de y por unidade de x)."""
    n = len(y)
    if n < 2:
        return 0.0
    # subamostra determinística se a série for muito longa (custo O(n^2))
    step = max(1, n // 800)
    xs, ys = x[::step], y[::step]
    slopes = []
    m = len(ys)
    for i in range(m - 1):
        for j in range(i + 1, m):
            dx = xs[j] - xs[i]
            if dx != 0:
                slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return 0.0
    slopes.sort()
    k = len(slopes)
    return slopes[k // 2] if k % 2 else 0.5 * (slopes[k // 2 - 1] + slopes[k // 2])


def trend(x, y):
    """Resumo de tendência de uma série. x em horas."""
    s, z, p_two, p_up = mann_kendall(y)
    return {
        "n": len(y),
        "mk_s": s,
        "mk_z": round(z, 4),
        "mk_p_two_sided": p_two,
        "mk_p_upward": p_up,
        "sen_slope_per_h": sen_slope(x, y),
    }


def percentile(values, q):
    if not values:
        return float("nan")
    v = sorted(values)
    idx = min(len(v) - 1, max(0, int(round(q / 100.0 * (len(v) - 1)))))
    return v[idx]


def trend_hr(x, y):
    """MK com modificação de Hamed-Rao (autocorrelação) + Sen's slope, como em analyze_run.py do Cesar.

    x em horas, y em unidades quaisquer. Usa pymannkendall (a mesma biblioteca do Cesar), que fica
    no venv travado. `mk_p_upward` é o p unilateral de alta derivado do teste de Hamed-Rao.
    """
    import warnings
    import numpy as np
    import pymannkendall as mk

    warnings.filterwarnings("ignore", category=RuntimeWarning, module="pymannkendall")
    n = len(y)
    out = {"n": n, "first": y[0] if n else None, "last": y[-1] if n else None,
           "median": percentile(y, 50) if n else None, "sen_slope_per_h": 0.0,
           "mk_p_upward": 1.0, "mk_hr_p": 1.0, "mk_hr_z": 0.0, "trend": "no trend", "sen_change_pct": None}
    if n < 10:
        return out
    out["sen_slope_per_h"] = sen_slope(x, y)
    span = x[-1] - x[0]
    out["sen_change_pct"] = (out["sen_slope_per_h"] * span / abs(out["median"]) * 100.0) if out["median"] else None
    arr = np.asarray(y, dtype=float)
    if np.allclose(arr, arr[0]):
        out["trend"] = "constant"
        return out
    try:
        r = mk.hamed_rao_modification_test(arr)
        p, z = float(r.p), float(r.z)
        if p != p:  # NaN: autocovariância degenerada em série quase constante
            return out
        out["mk_hr_p"], out["mk_hr_z"] = p, z
        out["mk_p_upward"] = p / 2.0 if z > 0 else 1.0 - p / 2.0
        out["trend"] = ("increasing" if z > 0 else "decreasing") if p < 0.05 else "no trend"
    except Exception:
        pass
    return out
