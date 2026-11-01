"""Presentation model and the read-only dashboard page."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


ASSETS_DIR = Path(__file__).resolve().parent / "dashboard_assets"
ASSET_TYPES = {
    "styles.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "live_app.js": "text/javascript; charset=utf-8",
}

STATUS_LABELS = {
    "running": "В работе", "accepted": "Принято", "rejected": "Отклонено",
    "rechecked": "Перепроверено",
    "invalid": "Недействительно", "crash": "Сбой", "timeout": "Таймаут",
    "budget": "Лимит", "interrupted": "Прервано", "incomparable": "Несопоставимо",
    "ready": "Готова к опытам", "completed": "Завершена", "setup_pending": "Подготовка",
    "config": "Ошибка конфигурации",
}

_REASON_LABELS = {
    "improved": "Улучшение подтверждено",
    "primary did not improve": "Главная метрика не улучшилась",
    "incomplete repeats": "Не все повторы завершены",
    "Evaluator timed out": "Оценщик превысил время",
    "Cost budget exceeded": "Превышен лимит стоимости",
    "Request budget exceeded": "Превышен лимит запросов",
    "Time budget exceeded": "Превышен лимит времени",
    "Candidate changed during evaluation": "Код гипотезы изменился во время оценки",
}

_METRIC_LABELS = {
    "f1": "F1", "accuracy": "Accuracy",
    "accuracy_floor": "Порог accuracy", "score": "Оценка",
    "latency": "Задержка", "error": "Ошибка", "validity": "Валидность",
}
_NOT_RATIO = ("error", "latency", "duration", "second", "cost", "loss", "time")
_RATIO_HINTS = ("f1", "accuracy", "precision", "recall", "rate")


def plural_ru(count: int, one: str, few: str, many: str) -> str:
    value = abs(int(count))
    if value % 10 == 1 and value % 100 != 11:
        return one
    if 2 <= value % 10 <= 4 and not 12 <= value % 100 <= 14:
        return few
    return many


def is_config_error(reason: str) -> bool:
    text = str(reason or "").lower()
    return "api_key" in text or "api key" in text


def metric_label(name: str) -> str:
    if name in _METRIC_LABELS:
        return _METRIC_LABELS[name]
    pretty = str(name or "").replace("_", " ").strip()
    return (pretty[:1].upper() + pretty[1:]) if pretty else str(name)


def metric_as_percent(name: str, samples: list[float] | None = None) -> bool:
    """Treat a metric as a 0–1 ratio only when the name and the values agree."""
    lowered = str(name or "").lower()
    if any(token in lowered for token in _NOT_RATIO):
        return False
    finite = [float(value) for value in (samples or []) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))]
    if finite and any(value < 0 or value > 1 for value in finite):
        return False
    if finite and all(0 <= value <= 1 for value in finite):
        return True
    return any(token in lowered for token in _RATIO_HINTS)


def reason_label(reason: str) -> str:
    text = str(reason or "")
    if is_config_error(text):
        return "Не найден ключ OPENROUTER — опыт не запущен"
    if text.startswith("guard ") and text.endswith(" regressed"):
        return f"Ухудшилась защитная метрика {metric_label(text[6:-10].strip())}"
    if text.startswith("Evaluator exited"):
        return "Оценщик завершился с ошибкой"
    return _REASON_LABELS.get(text, text)


def presentation_status(status: str, reason: str = "") -> tuple[str, str]:
    """Return a CSS kind and a Russian label. Missing API keys are not a budget stop."""
    if is_config_error(reason):
        return "config", STATUS_LABELS["config"]
    kind = status
    if status in {"crash", "timeout", "budget", "interrupted"}:
        kind = "failed"
    return kind, STATUS_LABELS.get(status, status)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _metric_values(manifest: dict[str, Any], metric: str) -> list[float]:
    values: list[float] = []
    runs = manifest.get("runs") or []
    if not isinstance(runs, list):
        return values
    for run in runs:
        if not isinstance(run, dict):
            continue
        metrics = run.get("metrics")
        if isinstance(metrics, dict) and metric in metrics:
            number = _finite(metrics.get(metric))
            if number is not None:
                values.append(number)
    return values


def _worst(values: list[float], direction: str) -> float:
    return min(values) if direction == "max" else max(values)


def _primary_score(values: list[float], direction: str, config: Any | None = None) -> float:
    if not values:
        return 0.0
    stat = getattr(config, "statistic", None)
    if stat is None and isinstance(config, dict):
        acceptance = config.get("acceptance") if isinstance(config.get("acceptance"), dict) else {}
        stat = acceptance.get("statistic") or config.get("statistic")
    stat = stat if stat in {"worst", "mean", "median"} else "worst"
    if stat == "mean":
        return statistics.fmean(values)
    if stat == "median":
        return float(statistics.median(values))
    dir_val = getattr(config, "direction", None)
    if dir_val is None and isinstance(config, dict):
        dir_val = config.get("direction")
    if dir_val is None:
        dir_val = direction
    return _worst(values, dir_val)


def _delta(value: float | None, reference: float | None, direction: str) -> float | None:
    if value is None or reference is None:
        return None
    return (value - reference) * (1 if direction == "max" else -1)


def chart_series(manifests: list[dict[str, Any]], metric: str, direction: str,
                 baseline_id: str | None = None,
                 guards: list[tuple[str, str]] | None = None,
                 expected_repeats: int | None = None,
                 config: Any | None = None) -> list[dict[str, Any]]:
    """One point per hypothesis, including runs that produced no metric.

    ``worst`` is the value used for the decision (the worst repeat). ``best_so_far``
    is the accepted record and stays monotonic; it does not replace the raw wave.
    """
    prepared: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifests):
        if not isinstance(manifest, dict):
            continue
        values = _metric_values(manifest, metric)
        has_value = bool(values)
        worst = _primary_score(values, direction, config) if has_value else None
        guard_values: dict[str, Any] = {}
        for name, sense in guards or []:
            samples = _metric_values(manifest, name)
            guard_values[name] = {
                "worst": _worst(samples, sense) if samples else None,
                "low": min(samples) if samples else None,
                "high": max(samples) if samples else None,
                "has_value": bool(samples),
            }
        status = str(manifest.get("status", "unknown"))
        finalization = manifest.get("finalization")
        decision = (finalization.get("decision") if isinstance(finalization, dict) else None)
        decision = manifest.get("decision") or decision or status
        repeat_target = manifest.get("expected_repeats", expected_repeats)
        try:
            repeat_target = int(repeat_target) if repeat_target is not None else None
        except (TypeError, ValueError):
            repeat_target = None
        parent_id = manifest.get("parent", manifest.get("parent_id"))
        point: dict[str, Any] = {
            "index": index,
            "id": manifest.get("id", ""),
            "title": manifest.get("title", manifest.get("id", "")),
            "status": status,
            "status_kind": presentation_status(status, str(manifest.get("reason") or ""))[0],
            "reason": str(manifest.get("reason") or ""),
            "reason_label": reason_label(str(manifest.get("reason") or "")),
            "low": min(values) if has_value else None,
            "high": max(values) if has_value else None,
            "worst": worst,
            "values": values,
            "has_value": has_value,
            "guards": guard_values,
            "parent_id": parent_id,
            "decision": str(decision),
            "is_baseline": (str(manifest.get("id", "")) == str(baseline_id) if baseline_id is not None
                            else manifest.get("kind") == "baseline" or str(manifest.get("id", "")).startswith("baseline")),
            "completed_repeats": len(values),
            "expected_repeats": repeat_target,
            "incomplete": (status in {"queued", "running", "finalizing", "interrupted", "timeout", "budget", "crash", "invalid"}
                           or (repeat_target is not None and repeat_target > 0 and len(values) < repeat_target)),
        }
        if manifest.get("result_type") is not None:
            point["result_type"] = manifest.get("result_type")
        prepared.append(point)

    baseline_worst: float | None = None
    if baseline_id:
        for row in prepared:
            if row["id"] == baseline_id and row["has_value"]:
                baseline_worst = row["worst"]
                break
    if baseline_worst is None:
        for row in prepared:
            if str(row["id"]).startswith("baseline") and row["has_value"]:
                baseline_worst = row["worst"]
                break

    best: float | None = None
    previous: float | None = None
    for row in prepared:
        worst = row["worst"]
        improved = False
        if row["status"] == "accepted" and worst is not None:
            better = best is None or (worst > best if direction == "max" else worst < best)
            if better:
                improved = best is not None
                best = worst
        row["best_so_far"] = best
        row["improved_best"] = improved
        row["delta_vs_baseline"] = _delta(worst, baseline_worst, direction)
        row["delta_vs_previous"] = _delta(worst, previous, direction)
        if worst is not None:
            previous = worst
    return prepared


def assign_short_labels(labs: list[dict[str, Any]]) -> None:
    names = [str(lab.get("name") or "") for lab in labs]
    for index, lab in enumerate(labs, start=1):
        name = str(lab.get("name") or "Лаборатория")
        lab["short_label"] = f"{name} · {index}" if names.count(name) > 1 else name


def _fmt_number(value: float, *, as_percent: bool, signed: bool = False) -> str:
    if as_percent:
        number = value * 100
        body = f"{number:+.1f}" if signed else f"{number:.1f}"
        body = body.replace(".", ",")
        return f"{body} п.п." if signed else f"{body}%"
    text = f"{value:+.4g}" if signed else f"{value:.4g}"
    return text.replace(".", ",")


def build_verdict(*, metric_name: str, direction: str, baseline: float | None, best: float | None,
                  hypotheses: list[dict[str, Any]], samples: list[float] | None = None) -> dict[str, Any]:
    """One sentence that answers whether the lab got better."""
    as_percent = metric_as_percent(metric_name, samples)
    trials = [item for item in hypotheses if not str(item.get("id", "")).startswith("baseline")]
    counts = {"accepted": 0, "rejected": 0, "failed": 0, "running": 0}
    for item in trials:
        reason = str(item.get("reason") or "")
        status = str(item.get("status") or "")
        if is_config_error(reason) or status in {"crash", "timeout", "budget", "invalid", "interrupted"}:
            counts["failed"] += 1
        elif status == "accepted":
            counts["accepted"] += 1
        elif status == "rejected":
            counts["rejected"] += 1
        elif status == "running":
            counts["running"] += 1
    total = len(trials)
    level = "neutral"
    if total == 0:
        if baseline is None:
            text = "Опытов пока нет."
        else:
            text = f"Опытов пока нет. База: {_fmt_number(baseline, as_percent=as_percent)}."
    elif baseline is None or best is None:
        text = "Измерения ещё не готовы."
    else:
        delta = _delta(best, baseline, direction)
        if delta is None or abs(delta) <= 1e-9:
            movement = f"Результат не изменился ({_fmt_number(baseline, as_percent=as_percent)})"
        elif delta > 0:
            level = "good"
            movement = (
                f"Результат вырос с {_fmt_number(baseline, as_percent=as_percent)} "
                f"до {_fmt_number(best, as_percent=as_percent)} ({_fmt_number(delta, as_percent=as_percent, signed=True)})"
            )
        else:
            level = "bad"
            movement = (
                f"Результат снизился с {_fmt_number(baseline, as_percent=as_percent)} "
                f"до {_fmt_number(best, as_percent=as_percent)} ({_fmt_number(delta, as_percent=as_percent, signed=True)})"
            )
        text = (
            f"{movement} за {total} {plural_ru(total, 'опыт', 'опыта', 'опытов')}: "
            f"принято {counts['accepted']}, отклонено {counts['rejected']}, ошибок {counts['failed']}."
        )
    return {"level": level, "text": text, "trials": total, **counts}


def build_problems(summary: dict[str, Any]) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    unknown = int(summary.get("unknown_requests") or 0)
    if unknown:
        problems.append({
            "level": "bad",
            "title": "Стоимость части запросов не подтверждена",
            "text": (
                f"Не сверено {unknown} {plural_ru(unknown, 'запрос', 'запроса', 'запросов')}. "
                "Это не останавливает следующие опыты."
            ),
        })
    error = str(summary.get("latest_error") or "")
    latest_status = str(summary.get("latest_trial_status") or "")
    # An earlier crash must not be presented as the outcome once a later trial finished normally.
    if (error and "неподтверждённой стоимостью" not in error
            and latest_status in {"invalid", "crash", "timeout", "budget", "interrupted"}):
        label = reason_label(error)
        item = {"level": "bad", "title": "Последний опыт остановился с ошибкой", "text": label}
        secret = any(token in error.lower() for token in ("api_key", "secret", "token", "password"))
        if label != error and not secret:
            item["detail"] = error
        problems.append(item)
    note = str(summary.get("comparison_note") or "")
    incomparable = summary.get("comparison_status") == "incomparable"
    if note or incomparable:
        item = {
            "level": "info",
            "title": "Справка о контракте",
            "text": (
                "Контракт измерения изменился или не совпадает с текущей эпохой. "
                "Рекорд этой лаборатории считается её собственным результатом для своей эпохи."
            ),
        }
        if note:
            item["detail"] = note
        problems.append(item)
    return problems


def read_asset(name: str) -> bytes | None:
    if name not in ASSET_TYPES or any(part in name for part in ("/", "\\", "..")):
        return None
    path = (ASSETS_DIR / name).resolve()
    root = ASSETS_DIR.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    return path.read_bytes()


def _page_parts(*, live: bool) -> tuple[str, str, str]:
    if live:
        return (
            '<link rel="stylesheet" href="/assets/styles.css">',
            '<script src="/assets/live_app.js" defer></script>',
            "Live",
        )
    css = (ASSETS_DIR / "styles.css").read_text(encoding="utf-8").replace("</style>", "<\\/style>")
    script = (ASSETS_DIR / "app.js").read_text(encoding="utf-8").replace("</script>", "<\\/script>")
    return f"<style>\n{css}\n</style>", f"<script>\n{script}\n</script>", "Офлайн-снимок"


def render_unified_dashboard(snapshot: dict[str, Any], details: dict[str, Any], *,
                             live: bool, root_id: str, selected_lab: str = "") -> str:
    """Fill the shared page shell. Offline reports inline the same CSS and JS."""
    payload = json.dumps(
        {"summary": snapshot, "details": details, "selected": selected_lab, "live": live, "root_id": root_id},
        ensure_ascii=False, separators=(",", ":"),
    )
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    style, app, mode = _page_parts(live=live)
    page = (ASSETS_DIR / "index.html").read_text(encoding="utf-8")
    return (page.replace("__STYLE__", style).replace("__APP__", app)
            .replace("__MODE_LABEL__", mode).replace("__PAYLOAD__", payload))
