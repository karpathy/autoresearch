#!/usr/bin/env python3
"""
Build reduced temporal features with consistent naming:
  - before_{feature_name}_{stat}
  - at_{feature_name}
  - after_{feature_name}_{stat}

Design goals:
  1) Keep a reduced set of high-signal features across structural/lexical,
     entropy-logit, activation, and attention families.
  2) Emit a consistent temporal schema for all families.
  3) Run raw extraction and temporal reduction end-to-end in one script.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import tempfile
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
NEGATION_RE = re.compile(r"\b(no|not|never|n't|none|nothing|neither|nor)\b", re.IGNORECASE)

LEXICONS: Dict[str, set[str]] = {
    "deceptive_word_frac": {
        "maybe",
        "perhaps",
        "actually",
        "honestly",
        "trust",
        "promise",
        "believe",
        "pretend",
        "fake",
        "bluff",
        "lie",
        "deceive",
        "trick",
    },
    "hedge_word_frac": {
        "maybe",
        "perhaps",
        "probably",
        "possibly",
        "likely",
        "seems",
        "appears",
        "might",
        "could",
    },
    "certainty_word_frac": {
        "definitely",
        "certainly",
        "always",
        "never",
        "must",
        "sure",
        "clearly",
    },
    "negation_word_frac": {
        "no",
        "not",
        "never",
        "none",
        "nothing",
        "neither",
        "nor",
        "without",
    },
    "justification_word_frac": {
        "because",
        "since",
        "therefore",
        "thus",
        "hence",
        "so",
        "reason",
        "why",
    },
    "self_reference_word_frac": {
        "i",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "our",
        "ours",
    },
    "contradiction_word_frac": {
        "but",
        "however",
        "though",
        "although",
        "yet",
        "instead",
        "otherwise",
        "except",
    },
}


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def text_features(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        text = ""
    chars = len(text)
    words = WORD_RE.findall(text)
    word_count = len(words)
    digit_count = sum(ch.isdigit() for ch in text)
    alpha_count = sum(ch.isalpha() for ch in text)
    upper_count = sum(ch.isupper() for ch in text)
    upper_ratio = (upper_count / alpha_count) if alpha_count else 0.0
    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0
    punct_counts = {
        "punct_period": text.count("."),
        "punct_comma": text.count(","),
        "punct_qmark": text.count("?"),
        "punct_exclaim": text.count("!"),
        "punct_colon": text.count(":"),
        "punct_semicolon": text.count(";"),
    }
    return {
        "char_count": chars,
        "word_count": word_count,
        "digit_count": digit_count,
        "upper_ratio": upper_ratio,
        "avg_word_len": avg_word_len,
        "negation_count": len(NEGATION_RE.findall(text)),
        **punct_counts,
    }


def think_close_span_from_text(text: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(text, str):
        return None
    close_tag = "</think>"
    close_start = text.find(close_tag)
    if close_start < 0:
        return None
    return close_start, close_start + len(close_tag)


def trim_text_after_think(text: Any) -> Tuple[Any, Optional[Tuple[int, int]]]:
    close_span = think_close_span_from_text(text)
    if close_span is None or not isinstance(text, str):
        return text, close_span
    return text[: close_span[1]], close_span


def think_close_span(record: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    raw_text = record.get("raw_text") or ""
    return think_close_span_from_text(raw_text)


def trim_sentences_after_think(
    raw_text: str,
    sentences: Sequence[Mapping[str, Any]],
) -> Tuple[str, List[Dict[str, Any]], Optional[Tuple[int, int]]]:
    trimmed_text, close_span = trim_text_after_think(raw_text)
    if close_span is None or not isinstance(trimmed_text, str):
        return raw_text, [dict(s) for s in sentences], close_span

    _, close_end = close_span
    trimmed_sentences: List[Dict[str, Any]] = []
    for sent in sentences:
        try:
            start = int(sent.get("start"))
            end = int(sent.get("end"))
        except (TypeError, ValueError):
            continue
        if start >= close_end:
            continue

        new_end = min(end, close_end)
        if new_end <= start:
            continue

        trimmed_sent = dict(sent)
        trimmed_sent["start"] = start
        trimmed_sent["end"] = new_end
        if end > close_end or not isinstance(trimmed_sent.get("sentence_text"), str):
            trimmed_sent["sentence_text"] = trimmed_text[start:new_end]
        trimmed_sentences.append(trimmed_sent)

    return trimmed_text, trimmed_sentences, close_span


def trim_history_after_think(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    close_span = think_close_span(record)
    history = list(record.get("history") or [])
    if close_span is None:
        return history

    _, close_end = close_span
    trimmed: List[Dict[str, Any]] = []
    for item in history:
        span = item.get("char_span")
        if isinstance(span, (list, tuple)) and len(span) == 2 and span[1] is not None:
            try:
                span_end = int(span[1])
            except (TypeError, ValueError):
                trimmed.append(item)
                continue
            if span_end <= close_end:
                trimmed.append(item)
        else:
            trimmed.append(item)
    return trimmed


def trim_result_after_think(record: Mapping[str, Any]) -> Dict[str, Any]:
    trimmed_history = trim_history_after_think(record)
    trimmed = dict(record)
    raw_text, close_span = trim_text_after_think(trimmed.get("raw_text"))
    if isinstance(raw_text, str):
        trimmed["raw_text"] = raw_text
    trimmed["history"] = trimmed_history
    trimmed["full_score"] = trimmed_history[-1] if trimmed_history else None
    trimmed["think_close_span"] = close_span
    trimmed["dropped_probe_count"] = len(record.get("history") or []) - len(trimmed_history)
    return trimmed


def load_localization_history(loc_source: Optional[str]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Returns: {example_id: {sentence_idx: history_entry}}
    """
    if not loc_source:
        return {}

    def _history_idx(entry: Dict[str, Any]) -> Optional[int]:
        if "sentence_idx" in entry and entry["sentence_idx"] is not None:
            idx = entry["sentence_idx"]
        elif "sentence_idx_inclusive" in entry and entry["sentence_idx_inclusive"] is not None:
            idx = entry["sentence_idx_inclusive"]
        elif "sentence_end_idx" in entry and entry["sentence_end_idx"] is not None:
            idx = int(entry["sentence_end_idx"]) - 1
        else:
            return None
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        if idx < 0:
            return None
        return idx

    def _history_map(history: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        mapped: Dict[int, Dict[str, Any]] = {}
        for h in history:
            idx = _history_idx(h)
            if idx is None:
                continue
            mapped[idx] = h
        return mapped

    loc_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    path = Path(loc_source)
    files: List[Path] = []

    if path.is_file() and path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                data = trim_result_after_think(data)
                example_id = data.get("example_id")
                history = data.get("history") or []
                if not example_id:
                    continue
                loc_map[example_id] = _history_map(history)
        return loc_map

    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        data = trim_result_after_think(data)
        example_id = data.get("example_id")
        history = data.get("history") or []
        if not example_id:
            continue
        loc_map[example_id] = _history_map(history)

    return loc_map


def _map_tokens_to_sentences(
    offsets: List[Tuple[int, int]],
    sentence_spans: List[Dict[str, Any]],
) -> List[int]:
    token_to_sentence: List[int] = [-1] * len(offsets)
    if not offsets or not sentence_spans:
        return token_to_sentence

    sent_idx = 0
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        mid = (start + end) / 2.0
        while sent_idx < len(sentence_spans) and mid >= sentence_spans[sent_idx]["end"]:
            sent_idx += 1
        if sent_idx >= len(sentence_spans):
            break
        if sentence_spans[sent_idx]["start"] <= mid < sentence_spans[sent_idx]["end"]:
            token_to_sentence[i] = sent_idx
    return token_to_sentence


DEFAULT_STRUCTURAL_BASE = [
    "word_count",
    "char_count",
    "avg_word_len",
    "upper_ratio",
    "digit_count",
    "negation_count",
    "punct_qmark",
    "punct_exclaim",
]

DEFAULT_LEXICAL_BASE = [
    "deceptive_word_frac",
    "hedge_word_frac",
    "certainty_word_frac",
    "negation_word_frac",
    "justification_word_frac",
    "self_reference_word_frac",
    "contradiction_word_frac",
]

# Reduced uncertainty/confidence set commonly used for LM reliability signals:
# surprisal/NLL, entropy, confidence (pmax), and logit margin/dispersion.
DEFAULT_ENTROPY_BASE = [
    "tok_nll_mean",
    "tok_entropy_mean",
    "tok_entropy_topk_renorm_mean",
    "tok_margin_logit_mean",
    "tok_pmax_mean",
    "tok_logit_std_mean",
    "tok_entropy_delta_mean",
    "tok_entropy_posdiff_mean",
]

DEFAULT_ACTIVATION_BASE = [
    "act_m1_l2_mean",
    "act_m2_l2_mean",
    "act_m4_l2_mean",
    "act_m1_absmean_mean",
    "act_m2_absmean_mean",
    "act_pair_m2_m1_cos",
    "act_traj_energy_slope",
    "act_traj_energy_std",
]

DEFAULT_ATTENTION_BASE = [
    "attn_rawmean__d1__in_long_mass",
    "attn_rawmean__d1__out_long_mass",
    "attn_rawmean__d1__anchor_ratio",
    "attn_rawmean__d2__in_long_entropy",
    "attn_rawmean__d2__out_long_entropy",
    "attn_roll__d1__in_long_mass",
    "attn_roll__d1__out_long_mass",
    "attn_roll__d1__anchor_ratio",
    "attn_roll__d2__in_long_entropy",
    "attn_roll__d2__out_long_entropy",
]

FAMILY_ORDER = ["structural", "lexical", "entropy", "activation", "attention"]
DEFAULT_SENTENCE_LEVEL_STATS = ("mean", "max", "min", "std")
DEFAULT_TEMPORAL_AGG_STATS = ("mean", "max", "min", "std")
VALID_STATS = {"mean", "max", "min", "std"}


def _parse_cols_arg(raw: str | None, default: Sequence[str]) -> List[str]:
    if raw is None:
        return list(default)
    cols = [x.strip() for x in raw.split(",") if x.strip()]
    return cols


def _parse_stats_arg(raw: str, name: str) -> Tuple[str, ...]:
    stats = tuple(x.strip() for x in raw.split(",") if x.strip())
    if not stats:
        raise ValueError(f"{name} must include at least one stat.")
    bad = [s for s in stats if s not in VALID_STATS]
    if bad:
        raise ValueError(f"{name} has invalid stats {bad}. Allowed: {sorted(VALID_STATS)}")
    return stats


def _unique(seq: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


_ACT_LAYER_RE = re.compile(r"^act_(m\d+|p\d+)_[a-z0-9_]+$")


def _activation_triplet_tags(df: pd.DataFrame) -> Tuple[List[str], Dict[str, object]]:
    tags: List[str] = []
    for c in df.columns:
        m = _ACT_LAYER_RE.match(c)
        if m:
            tags.append(m.group(1))
    tags = _unique(tags)

    p_nums = sorted(int(t[1:]) for t in tags if t.startswith("p"))
    m_nums = sorted((int(t[1:]) for t in tags if t.startswith("m")), reverse=True)

    ordered: List[str]
    source: str
    notes: str
    if p_nums:
        ordered = [f"p{n}" for n in p_nums]
        source = "positive_indices"
        notes = "Interpreted as direct hidden-state indices (p*)."
    elif m_nums:
        # mN means N layers from last. Larger N is earlier in depth.
        ordered = [f"m{n}" for n in m_nums]
        source = "negative_offsets"
        notes = "No p* tags found; used m* tags as earliest/mid/latest proxies."
    else:
        return [], {
            "ordered_tags": [],
            "selected_triplet_tags": [],
            "source": "none",
            "notes": "No activation layer tags found in input columns.",
        }

    first = ordered[0]
    # Lower-mid choice keeps the middle anchor from drifting too close to "last"
    # when the number of available layer tags is even.
    mid = ordered[(len(ordered) - 1) // 2]
    last = ordered[-1]
    selected = _unique([first, mid, last])
    meta = {
        "ordered_tags": ordered,
        "selected_triplet_tags": selected,
        "source": source,
        "notes": notes,
    }
    return selected, meta


def _derive_activation_base_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, object]]:
    selected_tags, tag_meta = _activation_triplet_tags(df)
    cols: List[str] = []

    per_layer_suffixes = ("l2_mean", "absmean_mean", "sparse_mean")
    for tag in selected_tags:
        for suffix in per_layer_suffixes:
            c = f"act_{tag}_{suffix}"
            if c in df.columns:
                cols.append(c)

    # Pairwise drift between chosen depth anchors (if present).
    for a, b in zip(selected_tags[:-1], selected_tags[1:]):
        for met in ("cos", "l2diff", "ratio"):
            c = f"act_pair_{a}_{b}_{met}"
            if c in df.columns:
                cols.append(c)

    # Keep trajectory features if available.
    for c in (
        "act_traj_energy_slope",
        "act_traj_energy_curv",
        "act_traj_energy_range",
        "act_traj_energy_last_first",
        "act_traj_energy_std",
    ):
        if c in df.columns:
            cols.append(c)

    cols = _unique(cols)
    if not cols:
        cols = [c for c in DEFAULT_ACTIVATION_BASE if c in df.columns]
        tag_meta = dict(tag_meta)
        tag_meta["notes"] = (
            str(tag_meta.get("notes", "")) + " | Fell back to DEFAULT_ACTIVATION_BASE."
        ).strip(" |")

    return cols, tag_meta


def _parse_int_tuple(raw: str, name: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not vals:
        raise ValueError(f"{name} must include at least one integer.")
    return vals


def _activation_offsets_layer0_mid_last(
    model_name: str,
    *,
    trust_remote_code: bool,
) -> Tuple[Tuple[int, ...], Dict[str, object]]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    n_hidden = getattr(cfg, "num_hidden_layers", None)
    if not isinstance(n_hidden, int) or n_hidden <= 0:
        raise ValueError(
            f"Could not read num_hidden_layers from config for {model_name}. "
            "Use --activation-layer-mode explicit."
        )

    # hidden_states[0] is embeddings for most decoder models; first block output is index 1.
    layer0_idx = 1
    mid_idx = max(1, n_hidden // 2)
    last_idx = n_hidden
    offsets = tuple(_unique([layer0_idx, mid_idx, last_idx]))
    meta = {
        "num_hidden_layers": int(n_hidden),
        "layer0_index_used": int(layer0_idx),
        "mid_index_used": int(mid_idx),
        "last_index_used": int(last_idx),
        "activation_layer_offsets_used": offsets,
    }
    return offsets, meta


def _safe_entropy_from_probs(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p + 1e-12)).sum())


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    cum = np.cumsum(xs)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _lexicon_fraction(text: str, lexicon: set[str]) -> float:
    if not isinstance(text, str) or not text:
        return 0.0
    toks = WORD_RE.findall(text.lower())
    if not toks:
        return 0.0
    hits = sum(1 for w in toks if w in lexicon)
    return float(hits / len(toks))


def _rolling_nan_mean_var(arr: np.ndarray, radius: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    out_mean = np.full(n, np.nan, dtype=np.float64)
    out_var = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out_mean, out_var
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        vals = arr[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        out_mean[i] = float(vals.mean())
        out_var[i] = float(vals.var())
    return out_mean, out_var


def _sentence_attention_matrix(attn_tt: "torch.Tensor", token_sent: "torch.Tensor", n_sent: int) -> "torch.Tensor":
    import torch

    valid = token_sent >= 0
    if valid.sum().item() == 0:
        return torch.zeros((n_sent, n_sent), device=attn_tt.device, dtype=torch.float32)

    valid_idx = torch.nonzero(valid, as_tuple=True)[0]
    sent_ids = token_sent[valid]

    counts = torch.bincount(sent_ids, minlength=n_sent).to(torch.float32)
    counts_safe = counts.clone()
    counts_safe[counts_safe == 0] = 1.0

    a_valid = attn_tt[:, valid_idx].to(torch.float32)
    v = torch.zeros((attn_tt.shape[0], n_sent), device=attn_tt.device, dtype=torch.float32)
    v.index_add_(1, sent_ids, a_valid)
    v = v / counts_safe

    vq = v[valid_idx]
    m = torch.zeros((n_sent, n_sent), device=attn_tt.device, dtype=torch.float32)
    m.index_add_(0, sent_ids, vq)
    m = m / counts_safe.view(-1, 1)
    return m


def _attention_rollout_safe(attn_layers: Sequence["torch.Tensor"], add_residual: bool = True) -> "torch.Tensor":
    import torch

    t = attn_layers[0].shape[-1]
    r = torch.eye(t, device=attn_layers[0].device, dtype=torch.float32)
    i = torch.eye(t, device=attn_layers[0].device, dtype=torch.float32)
    eps = 1e-12

    for a in attn_layers:
        abar = a.to(torch.float32).mean(dim=0)
        abar = torch.nan_to_num(abar, nan=0.0, posinf=0.0, neginf=0.0)
        if add_residual:
            abar = abar + i
        rowsum = abar.sum(dim=-1, keepdim=True)
        bad = rowsum.squeeze(-1) <= eps
        if bad.any():
            abar[bad] = 1.0 / t
            rowsum = abar.sum(dim=-1, keepdim=True)
        abar = abar / (rowsum + eps)
        r = abar @ r
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    r = r / (r.sum(dim=-1, keepdim=True) + eps)
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _mine_sentence_matrix(m: np.ndarray, min_dist: int, topk: int = 3) -> Dict[str, np.ndarray]:
    n = m.shape[0]
    out = {}

    in_mass = np.zeros(n, dtype=np.float64)
    in_mean = np.zeros(n, dtype=np.float64)
    in_max = np.zeros(n, dtype=np.float64)
    in_entropy = np.zeros(n, dtype=np.float64)
    in_gini = np.zeros(n, dtype=np.float64)
    in_topk_share = np.zeros(n, dtype=np.float64)
    in_dist_mu = np.zeros(n, dtype=np.float64)
    in_dist_var = np.zeros(n, dtype=np.float64)

    out_mass = np.zeros(n, dtype=np.float64)
    out_mean = np.zeros(n, dtype=np.float64)
    out_max = np.zeros(n, dtype=np.float64)
    out_entropy = np.zeros(n, dtype=np.float64)
    out_gini = np.zeros(n, dtype=np.float64)
    out_topk_share = np.zeros(n, dtype=np.float64)
    out_dist_mu = np.zeros(n, dtype=np.float64)
    out_dist_var = np.zeros(n, dtype=np.float64)

    anchor_mass = np.zeros(n, dtype=np.float64)
    anchor_ratio = np.zeros(n, dtype=np.float64)

    for j in range(n):
        q_idx = np.arange(j + min_dist, n, dtype=int)
        if q_idx.size > 0:
            vals = m[q_idx, j]
            vals = np.clip(vals, 0.0, None)
            s = vals.sum()
            in_mass[j] = float(s)
            if vals.size > 0:
                in_mean[j] = float(vals.mean())
                in_max[j] = float(vals.max())
                if s > 0:
                    p = vals / s
                    in_entropy[j] = _safe_entropy_from_probs(p)
                    in_gini[j] = _gini(vals)
                    kk = min(topk, vals.size)
                    in_topk_share[j] = float(np.sort(vals)[-kk:].sum() / s) if kk > 0 else 0.0
                    dist = (q_idx - j).astype(np.float64)
                    in_dist_mu[j] = float((p * dist).sum())
                    in_dist_var[j] = float((p * (dist - in_dist_mu[j]) ** 2).sum())
            anchor_mass[j] = float(m[j, j]) if np.isfinite(m[j, j]) else 0.0
            denom = float(anchor_mass[j] + s + 1e-12)
            anchor_ratio[j] = float(anchor_mass[j] / denom)

        k_idx = np.arange(0, j - min_dist + 1, dtype=int)
        if k_idx.size > 0:
            vals = m[j, k_idx]
            vals = np.clip(vals, 0.0, None)
            s = vals.sum()
            out_mass[j] = float(s)
            if vals.size > 0:
                out_mean[j] = float(vals.mean())
                out_max[j] = float(vals.max())
                if s > 0:
                    p = vals / s
                    out_entropy[j] = _safe_entropy_from_probs(p)
                    out_gini[j] = _gini(vals)
                    kk = min(topk, vals.size)
                    out_topk_share[j] = float(np.sort(vals)[-kk:].sum() / s) if kk > 0 else 0.0
                    dist = (j - k_idx).astype(np.float64)
                    out_dist_mu[j] = float((p * dist).sum())
                    out_dist_var[j] = float((p * (dist - out_dist_mu[j]) ** 2).sum())

    out["in_long_mass"] = in_mass
    out["in_long_mean"] = in_mean
    out["in_long_max"] = in_max
    out["in_long_entropy"] = in_entropy
    out["in_long_gini"] = in_gini
    out["in_long_topk_share"] = in_topk_share
    out["in_long_dist_mu"] = in_dist_mu
    out["in_long_dist_var"] = in_dist_var

    out["out_long_mass"] = out_mass
    out["out_long_mean"] = out_mean
    out["out_long_max"] = out_max
    out["out_long_entropy"] = out_entropy
    out["out_long_gini"] = out_gini
    out["out_long_topk_share"] = out_topk_share
    out["out_long_dist_mu"] = out_dist_mu
    out["out_long_dist_var"] = out_dist_var

    out["anchor_mass"] = anchor_mass
    out["anchor_ratio"] = anchor_ratio
    out["entropy_in_minus_out"] = in_entropy - out_entropy
    out["distmu_in_minus_out"] = in_dist_mu - out_dist_mu
    return out


def _aggregate_tok_by_sentence(values: np.ndarray, sent_idx: np.ndarray, n_sent: int, stat: str) -> np.ndarray:
    out = np.full(n_sent, np.nan, dtype=np.float64)
    for s in range(n_sent):
        m = sent_idx == s
        v = values[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        if stat == "mean":
            out[s] = float(v.mean())
        elif stat == "max":
            out[s] = float(v.max())
        elif stat == "min":
            out[s] = float(v.min())
        elif stat == "std":
            out[s] = float(v.std())
        elif stat == "sum":
            out[s] = float(v.sum())
        elif stat == "p90":
            out[s] = float(np.quantile(v, 0.90))
        elif stat == "p95":
            out[s] = float(np.quantile(v, 0.95))
    return out


def _compute_token_and_attention_features(
    raw_text: str,
    sentence_spans: List[Dict[str, Any]],
    *,
    tokenizer,
    model,
    device: str,
    max_tokens: int,
    topk_vocab: int,
    min_dist_list: Sequence[int],
    attention_layer_offsets: Sequence[int],
    activation_layer_offsets: Sequence[int],
    activation_sparsity_eps: float,
) -> Dict[int, Dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    if not raw_text or not sentence_spans:
        return {}

    enc = tokenizer(
        raw_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True if max_tokens > 0 else False,
        max_length=max_tokens if max_tokens > 0 else None,
    )

    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()
    t = int(input_ids.shape[0])
    n_sent = len(sentence_spans)
    if t < 4 or n_sent == 0:
        return {}

    tok_to_sent = np.asarray(_map_tokens_to_sentences(offsets, sentence_spans), dtype=int)
    if tok_to_sent.size != t:
        return {}

    model_ids = input_ids.unsqueeze(0).to(device)
    with torch.inference_mode():
        autocast_ctx = torch.amp.autocast("cuda", enabled=False) if str(device).startswith("cuda") else nullcontext()
        with autocast_ctx:
            out = model(
                model_ids,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
            )

    logits = getattr(out, "logits", None)
    if logits is None:
        return {}
    logits = logits[0].to(torch.float32)  # [T, V]
    compute_device = logits.device
    attn_layers = getattr(out, "attentions", None)
    hidden_states = getattr(out, "hidden_states", None) or ()
    if logits.shape[0] != t:
        return {}

    nll_tok = np.full(t, np.nan, dtype=np.float64)
    ent_tok = np.full(t, np.nan, dtype=np.float64)
    ent_topk_tok = np.full(t, np.nan, dtype=np.float64)
    margin_tok = np.full(t, np.nan, dtype=np.float64)
    pmax_tok = np.full(t, np.nan, dtype=np.float64)
    logit_std_tok = np.full(t, np.nan, dtype=np.float64)

    logits_next = logits[:-1, :]
    targets = input_ids[1:].to(compute_device)
    nll = F.cross_entropy(logits_next, targets, reduction="none").detach().cpu().numpy()
    nll_tok[1:] = nll

    logz = torch.logsumexp(logits_next, dim=-1, keepdim=True)
    k = min(max(2, topk_vocab), logits_next.shape[-1])
    topv, _ = torch.topk(logits_next, k=k, dim=-1)

    p_top = torch.exp(topv - logz)
    p_top_sum = torch.clamp(p_top.sum(dim=-1), min=1e-9, max=1.0)
    p_other = torch.clamp(1.0 - p_top_sum, min=1e-12)

    ent_approx = -(p_top * (topv - logz)).sum(dim=-1) - p_other * torch.log(p_other)
    ent_tok[1:] = ent_approx.detach().cpu().numpy()

    p_top_renorm = p_top / p_top_sum.unsqueeze(-1)
    ent_topk = -(p_top_renorm * torch.log(torch.clamp(p_top_renorm, min=1e-12))).sum(dim=-1)
    ent_topk_tok[1:] = ent_topk.detach().cpu().numpy()

    top2, _ = torch.topk(logits_next, k=2, dim=-1)
    margin = (top2[:, 0] - top2[:, 1]).detach().cpu().numpy()
    margin_tok[1:] = margin

    pmax = torch.exp(top2[:, 0:1] - logz).squeeze(-1).detach().cpu().numpy()
    pmax_tok[1:] = pmax

    logit_std = logits_next.std(dim=-1).detach().cpu().numpy()
    logit_std_tok[1:] = logit_std

    ent_delta = np.full(t, np.nan, dtype=np.float64)
    ent_posdiff = np.full(t, np.nan, dtype=np.float64)
    if t >= 3:
        d = ent_tok[2:] - ent_tok[1:-1]
        ent_delta[2:] = d
        ent_posdiff[2:] = np.maximum(d, 0.0)

    roll_mean, roll_var = _rolling_nan_mean_var(ent_tok, radius=3)

    rows: Dict[int, Dict[str, Any]] = {i: {} for i in range(n_sent)}
    mapped = tok_to_sent >= 0
    token_sent = tok_to_sent.copy()
    token_sent[~mapped] = -1

    tok_count = np.bincount(token_sent[mapped], minlength=n_sent).astype(int)
    tok_count_frac = tok_count.astype(np.float64) / max(int(mapped.sum()), 1)

    for s in range(n_sent):
        rows[s]["tok_count"] = int(tok_count[s])
        rows[s]["tok_count_frac"] = float(tok_count_frac[s])

    metrics = {
        "tok_nll": nll_tok,
        "tok_entropy": ent_tok,
        "tok_entropy_topk_renorm": ent_topk_tok,
        "tok_margin_logit": margin_tok,
        "tok_pmax": pmax_tok,
        "tok_logit_std": logit_std_tok,
        "tok_entropy_delta": ent_delta,
        "tok_entropy_posdiff": ent_posdiff,
        "tok_entropy_roll_mean": roll_mean,
        "tok_entropy_roll_var": roll_var,
    }

    for name, arr in metrics.items():
        for stat in ("mean", "max", "min", "std"):
            vals = _aggregate_tok_by_sentence(arr, token_sent, n_sent, stat)
            col = f"{name}_{stat}"
            for s in range(n_sent):
                if np.isfinite(vals[s]):
                    rows[s][col] = float(vals[s])

    if hidden_states:
        hs_len = len(hidden_states)
        act_layers: List[Tuple[str, int, "torch.Tensor"]] = []
        seen_idx = set()
        for off in activation_layer_offsets:
            idx = hs_len + off if off < 0 else off
            if 0 <= idx < hs_len and idx not in seen_idx:
                seen_idx.add(idx)
                tag = f"m{abs(int(off))}" if off < 0 else f"p{int(off)}"
                act_layers.append((tag, idx, hidden_states[idx][0].to(torch.float32)))

        if act_layers:
            valid_mask = token_sent >= 0
            valid_idx_t = torch.nonzero(torch.tensor(valid_mask, device=compute_device), as_tuple=True)[0]
            sent_ids_t = torch.tensor(token_sent[valid_mask], device=compute_device, dtype=torch.long)
            counts = torch.bincount(sent_ids_t, minlength=n_sent).to(torch.float32)
            counts_safe = torch.clamp(counts, min=1.0)

            sent_emb_by_layer: List[Tuple[str, int, np.ndarray]] = []
            for tag, idx, h in act_layers:
                tok_l2 = torch.linalg.norm(h, ord=2, dim=-1).detach().cpu().numpy()
                tok_abs = torch.mean(torch.abs(h), dim=-1).detach().cpu().numpy()
                tok_sparse = torch.mean((torch.abs(h) <= float(activation_sparsity_eps)).to(torch.float32), dim=-1).detach().cpu().numpy()

                for name, arr in (
                    (f"act_{tag}_l2", tok_l2),
                    (f"act_{tag}_absmean", tok_abs),
                    (f"act_{tag}_sparse", tok_sparse),
                ):
                    for stat in ("mean", "max", "std", "min"):
                        vals = _aggregate_tok_by_sentence(arr, token_sent, n_sent, stat)
                        col = f"{name}_{stat}"
                        for s in range(n_sent):
                            if np.isfinite(vals[s]):
                                rows[s][col] = float(vals[s])

                h = h.to(device=compute_device, dtype=torch.float32)
                sum_emb = torch.zeros((n_sent, h.shape[-1]), device=compute_device, dtype=torch.float32)
                if valid_idx_t.numel() > 0:
                    sum_emb.index_add_(0, sent_ids_t, h[valid_idx_t])
                emb = sum_emb / counts_safe.view(-1, 1)
                emb_np = emb.detach().cpu().numpy()
                sent_emb_by_layer.append((tag, idx, emb_np))

            sent_emb_by_layer.sort(key=lambda x: x[1])
            for i_pair in range(1, len(sent_emb_by_layer)):
                tag_a, _, ea = sent_emb_by_layer[i_pair - 1]
                tag_b, _, eb = sent_emb_by_layer[i_pair]
                na = np.linalg.norm(ea, axis=1)
                nb = np.linalg.norm(eb, axis=1)
                dot = np.sum(ea * eb, axis=1)
                cos = dot / (na * nb + 1e-8)
                l2d = np.linalg.norm(eb - ea, axis=1)
                ratio = nb / (na + 1e-8)
                for s in range(n_sent):
                    rows[s][f"act_pair_{tag_a}_{tag_b}_cos"] = float(cos[s])
                    rows[s][f"act_pair_{tag_a}_{tag_b}_l2diff"] = float(l2d[s])
                    rows[s][f"act_pair_{tag_a}_{tag_b}_ratio"] = float(ratio[s])

            if len(sent_emb_by_layer) >= 2:
                layer_pos = np.asarray([idx for _, idx, _ in sent_emb_by_layer], dtype=np.float64)
                if np.std(layer_pos) < 1e-8:
                    layer_pos = np.arange(len(sent_emb_by_layer), dtype=np.float64)
                layer_pos = (layer_pos - layer_pos.min()) / (layer_pos.max() - layer_pos.min() + 1e-8)

                energies = np.stack([np.linalg.norm(e, axis=1) for _, _, e in sent_emb_by_layer], axis=0)
                for s in range(n_sent):
                    ys = energies[:, s]
                    if not np.isfinite(ys).all():
                        continue
                    try:
                        slope = float(np.polyfit(layer_pos, ys, 1)[0])
                    except Exception:
                        slope = np.nan
                    curv = float(np.mean(np.diff(ys, n=2))) if ys.size >= 3 else np.nan
                    rows[s]["act_traj_energy_slope"] = slope
                    rows[s]["act_traj_energy_curv"] = curv
                    rows[s]["act_traj_energy_range"] = float(np.max(ys) - np.min(ys))
                    rows[s]["act_traj_energy_last_first"] = float(ys[-1] - ys[0])
                    rows[s]["act_traj_energy_std"] = float(np.std(ys))

    if not attn_layers:
        return rows

    layer_ids = []
    l_total = len(attn_layers)
    for off in attention_layer_offsets:
        idx = l_total + off if off < 0 else off
        if 0 <= idx < l_total:
            layer_ids.append(idx)
    if not layer_ids:
        layer_ids = [l_total - 1]

    token_sent_t = torch.tensor(token_sent, device=compute_device, dtype=torch.long)

    last = attn_layers[layer_ids[-1]][0].to(device=compute_device, dtype=torch.float32)
    m_rawmean = _sentence_attention_matrix(last.mean(dim=0), token_sent_t, n_sent).detach().cpu().numpy()
    m_rawmax = _sentence_attention_matrix(last.max(dim=0).values, token_sent_t, n_sent).detach().cpu().numpy()

    roll_layers = [attn_layers[i][0].detach().to(device=compute_device, dtype=torch.float32) for i in layer_ids]
    m_roll = _sentence_attention_matrix(_attention_rollout_safe(roll_layers, add_residual=True), token_sent_t, n_sent)
    m_roll = m_roll.detach().cpu().numpy()

    mats = {
        "attn_rawmean": m_rawmean,
        "attn_rawmax": m_rawmax,
        "attn_roll": m_roll,
    }

    for chan, mm in mats.items():
        mm = np.nan_to_num(mm, nan=0.0, posinf=0.0, neginf=0.0)
        for d in min_dist_list:
            mined = _mine_sentence_matrix(mm, min_dist=int(d), topk=3)
            for feat, arr in mined.items():
                col = f"{chan}__d{int(d)}__{feat}"
                for s in range(n_sent):
                    rows[s][col] = float(arr[s])

    return rows


@dataclass
class ExtractConfig:
    examples_path: Path
    sentences_path: Path
    localization_path: Path
    out_path: Path
    model_name: str
    num_examples: int
    seed: int
    only_localized: bool
    max_tokens: int
    topk_vocab: int
    min_dist_list: Tuple[int, ...]
    attention_layer_offsets: Tuple[int, ...]
    activation_layer_offsets: Tuple[int, ...]
    activation_sparsity_eps: float
    device: str
    trust_remote_code: bool
    progress_every: int
    token_oom_backoff: float = 0.5
    token_min_tokens: int = 256


def run_extract(cfg: ExtractConfig) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    if not (0.0 < float(cfg.token_oom_backoff) < 1.0):
        raise ValueError("token_oom_backoff must be in (0, 1).")
    if int(cfg.token_min_tokens) <= 0:
        raise ValueError("token_min_tokens must be positive.")

    print(f"[extract] loading raw inputs from {cfg.examples_path.parent}", flush=True)
    examples = [x for x in read_jsonl(cfg.examples_path) if x.get("example_id")]
    sentences = [x for x in read_jsonl(cfg.sentences_path) if x.get("example_id")]
    loc_map = load_localization_history(str(cfg.localization_path))

    by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sentences:
        by_example[s["example_id"]].append(s)
    for ex_id in by_example:
        by_example[ex_id].sort(key=lambda r: int(r.get("sentence_idx", 0)))

    ex_by_id = {e["example_id"]: e for e in examples if e["example_id"] in by_example}
    example_ids = list(ex_by_id.keys())
    if cfg.num_examples > 0 and cfg.num_examples < len(example_ids):
        example_ids = random.sample(example_ids, cfg.num_examples)
    example_ids = sorted(example_ids)
    print(f"[extract] selected examples={len(example_ids)}", flush=True)

    print(f"[extract] loading model/tokenizer: {cfg.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
    )
    use_auto_device_map = bool(cfg.device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 1)
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "trust_remote_code": cfg.trust_remote_code,
    }
    if use_auto_device_map:
        model_kwargs["device_map"] = "auto"
        print(
            f"[extract] using device_map=auto across {torch.cuda.device_count()} visible CUDA devices",
            flush=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        **model_kwargs,
    )
    if not use_auto_device_map:
        model = model.to(cfg.device)
    model.eval()
    model_input_device = str(next(model.parameters()).device)
    print(f"[extract] model input device: {model_input_device}", flush=True)
    param_dtypes = {p.dtype for p in model.parameters()}
    tok_fp32_enforced = bool(param_dtypes == {torch.float32})
    if not tok_fp32_enforced:
        found = sorted(str(x) for x in param_dtypes)
        raise RuntimeError(f"Strict FP32 enforcement failed. Found model parameter dtypes: {found}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    runtime_max_tokens = int(cfg.max_tokens)

    for i, ex_id in enumerate(example_ids, start=1):
        ex = ex_by_id.get(ex_id)
        sents = by_example.get(ex_id, [])
        if not ex or not sents:
            skipped += 1
            continue

        loc_hist = loc_map.get(ex_id, {})
        rates = {int(k): (v.get("deception_rate") if isinstance(v, dict) else None) for k, v in loc_hist.items()}
        if cfg.only_localized and not rates:
            skipped += 1
            continue

        raw_text = ex.get("action_reasoning") or ex.get("action_raw_text") or ""
        raw_text, sents_trimmed, _ = trim_sentences_after_think(raw_text, sents)
        sentence_spans = [{"start": s.get("start"), "end": s.get("end")} for s in sents_trimmed]

        mined = {}
        cur_max_tokens = int(runtime_max_tokens)
        tok_max_tokens_used = int(runtime_max_tokens)
        tok_oom_retries = 0
        try:
            while True:
                try:
                    mined = _compute_token_and_attention_features(
                        raw_text,
                        sentence_spans,
                        tokenizer=tokenizer,
                        model=model,
                        device=model_input_device,
                        max_tokens=cur_max_tokens,
                        topk_vocab=cfg.topk_vocab,
                        min_dist_list=cfg.min_dist_list,
                        attention_layer_offsets=cfg.attention_layer_offsets,
                        activation_layer_offsets=cfg.activation_layer_offsets,
                        activation_sparsity_eps=cfg.activation_sparsity_eps,
                    )
                    tok_max_tokens_used = int(cur_max_tokens)
                    runtime_max_tokens = int(cur_max_tokens)
                    break
                except RuntimeError as oom_err:
                    msg = str(oom_err).lower()
                    is_oom = (
                        ("out of memory" in msg)
                        or ("cuda error: out of memory" in msg)
                        or ("cuda out of memory" in msg)
                    )
                    can_retry = (
                        cfg.device.startswith("cuda")
                        and is_oom
                        and cur_max_tokens > 0
                        and cur_max_tokens > int(cfg.token_min_tokens)
                    )
                    if not can_retry:
                        raise
                    tok_oom_retries += 1
                    next_max = max(int(cfg.token_min_tokens), int(cur_max_tokens * float(cfg.token_oom_backoff)))
                    if next_max >= cur_max_tokens:
                        next_max = cur_max_tokens - 1
                    if next_max < int(cfg.token_min_tokens):
                        raise
                    print(
                        f"[extract] OOM ex={ex_id} retry={tok_oom_retries} max_tokens {cur_max_tokens} -> {next_max}",
                        flush=True,
                    )
                    cur_max_tokens = next_max
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        except Exception as e:
            skipped += 1
            if i % cfg.progress_every == 0:
                print(f"[extract] warning ex={ex_id} failed: {type(e).__name__}", flush=True)
            continue

        total_sent = len(sents_trimmed)
        for s in sents_trimmed:
            sidx = int(s.get("sentence_idx", 0))
            rate = rates.get(sidx)
            if cfg.only_localized and rate is None:
                continue

            text = s.get("sentence_text", "") or ""
            row: Dict[str, Any] = {
                "example_id": ex_id,
                "sentence_idx": sidx,
                "sentence_position": (sidx / max(total_sent - 1, 1)),
                "total_sentences": total_sent,
                "deception_rate": rate,
                "tok_max_tokens_used": int(tok_max_tokens_used),
                "tok_oom_retries": int(tok_oom_retries),
                "tok_fp32_enforced": bool(tok_fp32_enforced),
            }
            row.update(text_features(text))
            for name, lex in LEXICONS.items():
                row[name] = _lexicon_fraction(text, lex)

            if mined and sidx in mined:
                row.update(mined[sidx])

            rows.append(row)

        if i % cfg.progress_every == 0:
            print(
                f"[extract] progress {i}/{len(example_ids)} rows={len(rows)} skipped={skipped}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows extracted. Check localization coverage / inputs.")

    if "tok_count" in df.columns:
        tok_count = pd.to_numeric(df["tok_count"], errors="coerce").fillna(0.0)
        tokenless_frac = float((tok_count <= 0.0).mean())
        print(
            f"[extract] token coverage: tokenless_sentence_frac={tokenless_frac:.4f} | "
            f"median_tok_count={float(tok_count.median()):.2f}",
            flush=True,
        )

    attn_cols = [c for c in df.columns if c.startswith("attn_")]
    z_data: Dict[str, pd.Series] = {}
    for c in attn_cols:
        g = df.groupby("example_id")[c]
        mu = g.transform("mean")
        sd = g.transform("std")
        z_data[c + "__z"] = (pd.to_numeric(df[c], errors="coerce") - mu) / (sd + 1e-8)
    if z_data:
        df = pd.concat([df, pd.DataFrame(z_data, index=df.index)], axis=1)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.out_path, index=False)
    print(f"[extract] wrote {cfg.out_path} shape={df.shape}", flush=True)
    return df


def _rolling_stat_grouped(
    source: pd.Series,
    groups: pd.Series,
    *,
    window: int,
    stat: str,
    reverse: bool = False,
) -> pd.Series:
    if window <= 0:
        return pd.Series(np.nan, index=source.index, dtype=float)

    s = source
    g = groups
    if reverse:
        s = s.iloc[::-1]
        g = g.iloc[::-1]

    min_periods = 2 if stat == "std" else 1
    rolling = s.groupby(g, sort=False).rolling(window=window, min_periods=min_periods)
    if stat == "mean":
        out = rolling.mean().reset_index(level=0, drop=True)
    elif stat == "min":
        out = rolling.min().reset_index(level=0, drop=True)
    elif stat == "max":
        out = rolling.max().reset_index(level=0, drop=True)
    elif stat == "std":
        out = rolling.std().reset_index(level=0, drop=True)
    else:
        raise ValueError(f"Unsupported rolling stat: {stat}")

    if reverse:
        out = out.iloc[::-1]
    return out.reindex(source.index)


_STAT_SUFFIX_RE = re.compile(r"^(.*)_(mean|max|min|std)$")


def _base_name_from_col(col: str) -> Tuple[str, str | None]:
    m = _STAT_SUFFIX_RE.match(col)
    if not m:
        return col, None
    return m.group(1), m.group(2)


def _build_sentence_level_feature_frame(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    sentence_level_stats: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build sentence-level stat features per base feature so every base has
    *_mean/*_max/*_min/*_std representation.
    """
    out = pd.DataFrame(index=df.index)
    built_cols: List[str] = []
    seen = set()

    for col in cols:
        if col not in df.columns:
            continue

        base_name, detected_stat = _base_name_from_col(col)
        src_raw = pd.to_numeric(df[col], errors="coerce")

        for st in sentence_level_stats:
            out_col = f"{base_name}_{st}"
            if out_col in seen:
                continue

            sibling = f"{base_name}_{st}"
            if sibling in df.columns:
                src = pd.to_numeric(df[sibling], errors="coerce")
            elif detected_stat == st:
                src = src_raw
            elif st in ("mean", "max", "min"):
                # For scalar sentence-level features, reuse the scalar value.
                src = src_raw
            else:
                # std fallback for scalar sentence-level features.
                src = pd.Series(0.0, index=df.index, dtype=float)

            out[out_col] = src.astype(float)
            built_cols.append(out_col)
            seen.add(out_col)

    return out, built_cols


def _temporalize_column(
    df: pd.DataFrame,
    col: str,
    *,
    before_window: int,
    after_window: int,
    include_diff: bool,
    temporal_agg_stats: Sequence[str],
) -> pd.DataFrame:
    vals = pd.to_numeric(df[col], errors="coerce")
    groups = df["example_id"].astype(str)
    grp = vals.groupby(groups, sort=False)

    before_src = grp.shift(1)
    after_src = grp.shift(-1)

    out = pd.DataFrame(index=df.index)
    out[f"at_{col}"] = vals.astype(float)

    for stat in temporal_agg_stats:
        out[f"before_{col}_{stat}"] = _rolling_stat_grouped(
            before_src,
            groups,
            window=before_window,
            stat=stat,
            reverse=False,
        )
        out[f"after_{col}_{stat}"] = _rolling_stat_grouped(
            after_src,
            groups,
            window=after_window,
            stat=stat,
            reverse=True,
        )

    if include_diff:
        out[f"before_{col}_mean_minus_at"] = out[f"before_{col}_mean"] - out[f"at_{col}"]
        out[f"after_{col}_mean_minus_at"] = out[f"after_{col}_mean"] - out[f"at_{col}"]
        out[f"after_{col}_mean_minus_before_mean"] = (
            out[f"after_{col}_mean"] - out[f"before_{col}_mean"]
        )

    return out


def _resolve_bases(
    df: pd.DataFrame,
    requested: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    available: Dict[str, List[str]] = {}
    missing: Dict[str, List[str]] = {}
    for fam, cols in requested.items():
        have = [c for c in cols if c in df.columns]
        miss = [c for c in cols if c not in df.columns]
        available[fam] = have
        missing[fam] = miss
    return available, missing


def _build_feature_sets(generated_by_family: Mapping[str, Sequence[str]]) -> OrderedDict:
    structural = list(generated_by_family.get("structural", []))
    lexical = list(generated_by_family.get("lexical", []))
    entropy = list(generated_by_family.get("entropy", []))
    activation = list(generated_by_family.get("activation", []))
    attention = list(generated_by_family.get("attention", []))

    struct_lex = _unique(structural + lexical)
    set1 = _unique(struct_lex + entropy)
    set2 = _unique(set1 + activation)
    set3 = _unique(set2 + attention)

    # Keep names close to prior notebooks.
    feature_sets = OrderedDict(
        {
            "baseline_struct": structural,
            "baseline_struct_lex": struct_lex,
            "set1_struct_lex_entropy": set1,
            "set2_struct_lex_entropy_activation": set2,
            "set3_struct_lex_entropy_activation_attention": set3,
        }
    )
    return feature_sets


def build_temporal_reduced_features(
    df: pd.DataFrame,
    *,
    base_by_family: Mapping[str, Sequence[str]],
    before_window: int,
    after_window: int,
    include_diff: bool,
    sentence_level_stats: Sequence[str],
    temporal_agg_stats: Sequence[str],
) -> Tuple[pd.DataFrame, OrderedDict, Dict[str, List[str]], Dict[str, List[str]]]:
    if "example_id" not in df.columns:
        raise ValueError("Input DataFrame must contain example_id.")
    if "sentence_idx" not in df.columns:
        raise ValueError("Input DataFrame must contain sentence_idx.")

    work = df.copy()
    work["example_id"] = work["example_id"].astype(str)
    work["sentence_idx"] = pd.to_numeric(work["sentence_idx"], errors="coerce")
    work = work[np.isfinite(work["sentence_idx"])].copy()
    work["sentence_idx"] = work["sentence_idx"].astype(int)
    work = work.sort_values(["example_id", "sentence_idx"], kind="stable").reset_index(drop=True)

    available, missing = _resolve_bases(work, base_by_family)

    meta_cols = [
        c
        for c in ["example_id", "sentence_idx", "deception_rate", "sentence_position", "total_sentences"]
        if c in work.columns
    ]

    out_frames = [work[meta_cols].copy()]
    generated_by_family: Dict[str, List[str]] = {fam: [] for fam in FAMILY_ORDER}

    for fam in FAMILY_ORDER:
        cols = available.get(fam, [])
        sentence_df, sentence_cols = _build_sentence_level_feature_frame(
            work,
            cols,
            sentence_level_stats=sentence_level_stats,
        )
        fam_df = pd.concat([work[["example_id"]], sentence_df], axis=1)
        for col in sentence_cols:
            tdf = _temporalize_column(
                fam_df,
                col,
                before_window=before_window,
                after_window=after_window,
                include_diff=include_diff,
                temporal_agg_stats=temporal_agg_stats,
            )
            out_frames.append(tdf)
            generated_by_family[fam].extend(tdf.columns.tolist())

    out_df = pd.concat(out_frames, axis=1)
    feature_sets = _build_feature_sets(generated_by_family)
    return out_df, feature_sets, generated_by_family, missing


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _build_base_by_family_from_args(
    args: argparse.Namespace,
    df: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], Dict[str, object]]:
    structural = _parse_cols_arg(args.structural_cols, DEFAULT_STRUCTURAL_BASE)
    lexical = _parse_cols_arg(args.lexical_cols, DEFAULT_LEXICAL_BASE)
    entropy = _parse_cols_arg(args.entropy_cols, DEFAULT_ENTROPY_BASE)
    attention = _parse_cols_arg(args.attention_cols, DEFAULT_ATTENTION_BASE)

    if args.activation_cols is not None:
        activation = _parse_cols_arg(args.activation_cols, DEFAULT_ACTIVATION_BASE)
        activation_meta: Dict[str, object] = {
            "activation_base_source": "explicit_cols",
            "selected_triplet_tags": [],
            "notes": "Activation columns provided by --activation-cols.",
        }
    else:
        activation, tag_meta = _derive_activation_base_columns(df)
        activation_meta = {
            "activation_base_source": "derived_layer0_mid_last",
            **tag_meta,
        }

    base_by_family = {
        "structural": structural,
        "lexical": lexical,
        "entropy": entropy,
        "activation": activation,
        "attention": attention,
    }
    return base_by_family, activation_meta


def _add_temporal_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--before-window", type=int, default=3)
    ap.add_argument("--after-window", type=int, default=3)
    ap.add_argument("--sentence-level-stats", type=str, default="mean,max,min,std")
    ap.add_argument("--temporal-agg-stats", type=str, default="mean,max,min,std")
    ap.add_argument("--no-diff", action="store_true", default=False)
    ap.add_argument("--structural-cols", type=str, default=None)
    ap.add_argument("--lexical-cols", type=str, default=None)
    ap.add_argument("--entropy-cols", type=str, default=None)
    ap.add_argument("--activation-cols", type=str, default=None)
    ap.add_argument("--attention-cols", type=str, default=None)
    ap.add_argument("--feature-sets-out", type=Path, default=None)
    ap.add_argument("--manifest-out", type=Path, default=None)


def _add_extract_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--examples-path", type=Path, required=True)
    ap.add_argument("--sentences-path", type=Path, required=True)
    ap.add_argument("--localization-path", type=Path, required=True)
    ap.add_argument("--model-name", type=str, required=True)
    ap.add_argument("--num-examples", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-localized", action="store_true", default=True)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--topk-vocab", type=int, default=32)
    ap.add_argument("--min-dist-list", type=str, default="1,2,4")
    ap.add_argument("--attention-layer-offsets", type=str, default="-1,-2,-3")
    ap.add_argument("--activation-layer-offsets", type=str, default="-1,-2,-4,-8")
    ap.add_argument(
        "--activation-layer-mode",
        type=str,
        choices=("layer0_mid_last", "explicit"),
        default="layer0_mid_last",
        help="How to choose activation layers during raw extraction.",
    )
    ap.add_argument("--activation-sparsity-eps", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--token-oom-backoff", type=float, default=0.5)
    ap.add_argument("--token-min-tokens", type=int, default=256)
    ap.add_argument("--raw-out-path", type=Path, default=None)


def _run_extract_build(args: argparse.Namespace) -> None:
    raw_out_path: Path
    if args.raw_out_path is not None:
        raw_out_path = args.raw_out_path
        raw_out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="temporal_reduced_raw_"))
        raw_out_path = tmpdir / "raw_features.parquet"

    min_dist_list = _parse_int_tuple(args.min_dist_list, "min-dist-list")
    layer_offsets = _parse_int_tuple(args.attention_layer_offsets, "attention-layer-offsets")
    act_layer_meta: Dict[str, object] = {}
    if args.activation_layer_mode == "layer0_mid_last":
        try:
            act_layer_offsets, act_layer_meta = _activation_offsets_layer0_mid_last(
                args.model_name,
                trust_remote_code=args.trust_remote_code,
            )
            print(
                "[extract-build] activation layers (layer0/mid/last) "
                f"offsets={act_layer_offsets} meta={act_layer_meta}",
                flush=True,
            )
        except Exception as e:
            print(
                "[extract-build] warning: layer0_mid_last resolution failed; "
                f"falling back to explicit offsets '{args.activation_layer_offsets}'. "
                f"reason={type(e).__name__}: {e}",
                flush=True,
            )
            act_layer_offsets = _parse_int_tuple(args.activation_layer_offsets, "activation-layer-offsets")
            act_layer_meta = {
                "activation_layer_mode": "fallback_explicit",
                "activation_layer_offsets_used": act_layer_offsets,
            }
    else:
        act_layer_offsets = _parse_int_tuple(args.activation_layer_offsets, "activation-layer-offsets")
        act_layer_meta = {
            "activation_layer_mode": "explicit",
            "activation_layer_offsets_used": act_layer_offsets,
        }

    cfg = ExtractConfig(
        examples_path=args.examples_path,
        sentences_path=args.sentences_path,
        localization_path=args.localization_path,
        out_path=raw_out_path,
        model_name=args.model_name,
        num_examples=args.num_examples,
        seed=args.seed,
        only_localized=args.only_localized,
        max_tokens=args.max_tokens,
        topk_vocab=args.topk_vocab,
        min_dist_list=min_dist_list,
        attention_layer_offsets=layer_offsets,
        activation_layer_offsets=act_layer_offsets,
        activation_sparsity_eps=args.activation_sparsity_eps,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        progress_every=args.progress_every,
        token_oom_backoff=args.token_oom_backoff,
        token_min_tokens=args.token_min_tokens,
    )

    print("[extract-build] running raw extraction...", flush=True)
    raw_df = run_extract(cfg)
    print(f"[extract-build] raw extraction complete shape={raw_df.shape}", flush=True)

    # Reuse build path from in-memory DataFrame.
    base_by_family, activation_meta = _build_base_by_family_from_args(args, df=raw_df)
    include_diff = not args.no_diff
    sentence_level_stats = _parse_stats_arg(args.sentence_level_stats, "sentence-level-stats")
    temporal_agg_stats = _parse_stats_arg(args.temporal_agg_stats, "temporal-agg-stats")
    out_df, feature_sets, generated_by_family, missing = build_temporal_reduced_features(
        raw_df,
        base_by_family=base_by_family,
        before_window=args.before_window,
        after_window=args.after_window,
        include_diff=include_diff,
        sentence_level_stats=sentence_level_stats,
        temporal_agg_stats=temporal_agg_stats,
    )

    out_path: Path = args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[extract-build] wrote temporal features: {out_path} shape={out_df.shape}", flush=True)

    feature_sets_out = args.feature_sets_out or out_path.with_name(out_path.stem + "_feature_sets.json")
    manifest_out = args.manifest_out or out_path.with_name(out_path.stem + "_manifest.json")
    _write_json(feature_sets_out, feature_sets)
    manifest = {
        "raw_output_path": str(raw_out_path),
        "output_features": str(out_path),
        "before_window": int(args.before_window),
        "after_window": int(args.after_window),
        "include_diff": bool(include_diff),
        "sentence_level_stats": list(sentence_level_stats),
        "temporal_agg_stats": list(temporal_agg_stats),
        "requested_base_features": base_by_family,
        "missing_base_features": missing,
        "activation_extraction": act_layer_meta,
        "activation_selection": activation_meta,
        "generated_feature_counts": {k: len(v) for k, v in generated_by_family.items()},
        "generated_feature_sets": {k: len(v) for k, v in feature_sets.items()},
    }
    _write_json(manifest_out, manifest)
    print(f"[extract-build] wrote feature sets: {feature_sets_out}", flush=True)
    print(f"[extract-build] wrote manifest: {manifest_out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-path", type=Path, required=True)
    _add_extract_args(ap)
    _add_temporal_args(ap)
    args = ap.parse_args()
    _run_extract_build(args)


if __name__ == "__main__":
    main()
