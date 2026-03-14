from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from pdca_system.domain.models import SeedStatus
from pdca_system.services.workflow import GitCommandError, WorkflowService
from pdca_system.task import (
    get_daemon_config,
    get_daemon_status,
    LOG_ROOT,
    PDCA_SYSTEM_ROOT,
    write_daemon_config,
)

router = APIRouter()


def _templates(request: Request):
    return request.app.state.templates


def _workflow(request: Request) -> WorkflowService:
    return request.app.state.workflow


def _is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request", "").lower() == "true"


def _render(request: Request, template_name: str, context: dict, status_code: int = 200) -> HTMLResponse:
    templates = _templates(request)
    return templates.TemplateResponse(request, template_name, {"request": request, **context}, status_code=status_code)


def _resolve_log_path(run_id: str, stream: str, run_log_path: str | None) -> Path | None:
    # Primary source: persisted run metadata path.
    if run_log_path:
        candidate = Path(run_log_path)
        if candidate.exists() and candidate.is_file():
            return candidate

    # Deterministic run-id naming (new format).
    run_named = LOG_ROOT / f"{run_id}.{stream}.log"
    if run_named.exists() and run_named.is_file():
        return run_named

    return None


def _resolve_prompt_path(run_id: str, run_prompt_path: str | None) -> Path | None:
    if run_prompt_path:
        candidate = Path(run_prompt_path)
        if candidate.exists() and candidate.is_file():
            return candidate
    prompt_named = LOG_ROOT / f"{run_id}.prompt.txt"
    if prompt_named.exists() and prompt_named.is_file():
        return prompt_named
    return None


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request, seed_id: str | None = None) -> HTMLResponse:
    workflow = _workflow(request)
    viewmodel = workflow.build_dashboard(selected_seed_id=seed_id)
    context = {
        "dashboard": viewmodel,
        "selected_seed_id": seed_id,
        "detail": workflow.seed_detail(seed_id) if seed_id else None,
        "daemon_config": get_daemon_config(),
    }
    return _render(request, "dashboard.html", context)


@router.get("/partials/dashboard", response_class=HTMLResponse)
def dashboard_board(request: Request, seed_id: str | None = None) -> HTMLResponse:
    workflow = _workflow(request)
    viewmodel = workflow.build_dashboard(selected_seed_id=seed_id)
    return _render(request, "partials/dashboard_board.html", {"dashboard": viewmodel, "selected_seed_id": seed_id})


@router.get("/partials/daemon-status", response_class=HTMLResponse)
def daemon_status_partial(request: Request) -> HTMLResponse:
    return _render(request, "partials/daemon_status.html", {"daemon_status": get_daemon_status()})


@router.get("/api/settings/daemon")
def get_daemon_settings() -> dict:
    """Return agent timeout settings (stored under history folder)."""
    return get_daemon_config()


@router.patch("/api/settings/daemon")
async def update_daemon_settings(request: Request) -> dict:
    """Update agent timeout settings. Persisted to pdca_system/history/state/daemon_config.json."""
    try:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    except Exception:
        body = {}
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    write_daemon_config(body)
    return get_daemon_config()


@router.get("/api/progress.png", name="progress_png", response_class=Response)
def progress_png() -> Response:
    """Serve progress.png from project root for the Figure popup. No caching so the modal always shows the current file."""
    path = PDCA_SYSTEM_ROOT.parent / "progress.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="progress.png not found")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
    )


@router.get("/partials/daemon-settings", response_class=HTMLResponse)
def daemon_settings_partial(request: Request) -> HTMLResponse:
    return _render(request, "partials/daemon_settings.html", {"daemon_config": get_daemon_config()})


def _parse_positive_int(value: str | int, default: int) -> int:
    if isinstance(value, int):
        return value if value > 0 else default
    try:
        n = int(value)
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


def _parse_int(value: str | int, default: int) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@router.post("/actions/settings/daemon", response_class=HTMLResponse, response_model=None)
def save_daemon_settings(
    request: Request,
    stuck_check_timeout_seconds: str | int = Form("120"),
    timeout_pd: str | int = Form("900"),
    timeout_ca: str | int = Form("3600"),
    timeout_direct: str | int = Form("3600"),
    ralph_max_loop: str | int = Form("0"),
) -> HTMLResponse | Response:
    """Save settings from form. Stored under history/state/daemon_config.json."""
    stuck_check_timeout_seconds = _parse_positive_int(stuck_check_timeout_seconds, 120)
    timeout_pd = _parse_positive_int(timeout_pd, 900)
    timeout_ca = _parse_positive_int(timeout_ca, 3600)
    timeout_direct = _parse_positive_int(timeout_direct, 3600)
    ralph_max_loop = _parse_int(ralph_max_loop, 0)
    write_daemon_config({
        "stuck_check_timeout_seconds": stuck_check_timeout_seconds,
        "default_timeouts": {"pd": timeout_pd, "ca": timeout_ca, "direct": timeout_direct},
        "ralph_max_loop": ralph_max_loop,
    })
    if _is_htmx(request):
        return _render(request, "partials/daemon_settings.html", {"daemon_config": get_daemon_config()})
    return RedirectResponse(request.url_for("dashboard"), status_code=303)


@router.get("/partials/seeds/{seed_id}", response_class=HTMLResponse)
def seed_detail_partial(request: Request, seed_id: str) -> HTMLResponse:
    workflow = _workflow(request)
    try:
        detail = workflow.seed_detail(seed_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    dashboard = workflow.build_dashboard(selected_seed_id=seed_id)
    context = {
        **detail,
        "dashboard": dashboard,
        "selected_seed_id": seed_id,
        "oob": True,
        "daemon_status": get_daemon_status(),
    }
    return _render(request, "partials/seed_detail_response.html", context)


@router.get("/api/seeds/{seed_id}/versions")
def seed_versions(request: Request, seed_id: str) -> dict[str, str]:
    workflow = _workflow(request)
    try:
        return workflow.seed_detail_versions(seed_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/partials/seeds/{seed_id}/runs", response_class=HTMLResponse)
def seed_runs_partial(request: Request, seed_id: str) -> HTMLResponse:
    workflow = _workflow(request)
    try:
        detail = workflow.seed_detail(seed_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _render(
        request,
        "partials/seed_runs_inner.html",
        {"seed": detail["seed"], "runs": detail["runs"]},
    )


@router.get("/partials/seeds/{seed_id}/timeline", response_class=HTMLResponse)
def seed_timeline_partial(request: Request, seed_id: str) -> HTMLResponse:
    workflow = _workflow(request)
    try:
        detail = workflow.seed_detail(seed_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _render(
        request,
        "partials/seed_timeline_inner.html",
        {"seed": detail["seed"], "events": detail["events"]},
    )


@router.get("/api/runs/{run_id}/prompt")
def run_prompt(request: Request, run_id: str) -> dict[str, object]:
    workflow = _workflow(request)
    run = workflow.run_repo.get(run_id)
    run_prompt_path = run.prompt_path if run is not None else None
    prompt_path = _resolve_prompt_path(run_id, run_prompt_path)
    if prompt_path is None:
        raise HTTPException(status_code=404, detail=f"Prompt for run '{run_id}' not found.")
    content = prompt_path.read_text(encoding="utf-8", errors="replace")
    return {"content": content}


@router.get("/api/runs/{run_id}/log")
def run_log_chunk(
    request: Request,
    run_id: str,
    stream: str = Query("stdout"),
    offset: int = Query(0, ge=0),
    limit: int = Query(64 * 1024, ge=1024, le=512 * 1024),
) -> dict[str, object]:
    workflow = _workflow(request)
    run = workflow.run_repo.get(run_id)

    complete_status = bool(run is not None and run.status.value in {"succeeded", "failed"})
    if stream not in {"stdout", "stderr"}:
        raise HTTPException(status_code=400, detail="stream must be one of: stdout, stderr")

    run_log_path = None
    if run is not None:
        run_log_path = run.log_path if stream == "stdout" else run.stderr_log_path
        if not run_log_path and stream == "stderr" and run.log_path and run.log_path.endswith(".stdout.log"):
            run_log_path = run.log_path.replace(".stdout.log", ".stderr.log")

    log_path = _resolve_log_path(run_id, stream, run_log_path)
    if log_path is None and run is not None and not complete_status:
        # During queued/running phases metadata may not yet include paths and files may appear slightly later.
        return {
            "chunk": "",
            "next_offset": offset,
            "size": 0,
            "complete": False,
        }

    if log_path is None:
        raise HTTPException(status_code=404, detail=f"Log for run '{run_id}' ({stream}) not found.")

    if not log_path.exists() or not log_path.is_file():
        return {
            "chunk": "",
            "next_offset": offset,
            "size": 0,
            "complete": complete_status,
        }

    file_size = log_path.stat().st_size
    if offset > file_size:
        offset = file_size

    with open(log_path, "rb") as handle:
        handle.seek(offset)
        payload = handle.read(limit)

    next_offset = offset + len(payload)
    return {
        "chunk": payload.decode("utf-8", errors="replace"),
        "next_offset": next_offset,
        "size": file_size,
        "complete": bool(complete_status and next_offset >= file_size),
    }


@router.get("/seeds/{seed_id}", response_class=HTMLResponse)
def seed_detail_page(request: Request, seed_id: str) -> HTMLResponse:
    workflow = _workflow(request)
    try:
        detail = workflow.seed_detail(seed_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _render(request, "seed_detail_page.html", {**detail, "daemon_status": get_daemon_status()})


@router.post("/actions/seeds", response_class=HTMLResponse)
def create_seed(
    request: Request,
    prompt: str = Form(...),
    baseline_branch: str = Form(...),
    seed_mode: str = Form("manual"),
) -> Response:
    workflow = _workflow(request)
    seed = workflow.create_seed(
        prompt,
        baseline_branch=baseline_branch,
        ralph_loop_enabled=seed_mode == "ralph",
    )
    if seed_mode == "ralph":
        try:
            workflow.queue_pd(seed.seed_id)
        except (RuntimeError, GitCommandError) as exc:
            workflow.seed_repo.append_event(
                seed.seed_id,
                "ralph.start_failed",
                f"Ralph loop could not queue the initial Plan-Do run: {exc}",
            )
    target_url = str(request.url_for("dashboard")) + f"?seed_id={seed.seed_id}"
    if _is_htmx(request):
        response = Response(status_code=204)
        response.headers["HX-Redirect"] = target_url
        return response
    return RedirectResponse(target_url, status_code=303)


@router.post("/actions/direct-code-agent", response_class=HTMLResponse)
def start_direct_code_agent(request: Request, prompt: str = Form(...)) -> Response:
    workflow = _workflow(request)
    try:
        seed, _run = workflow.create_direct_code_seed(prompt)
    except RuntimeError as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target_url = str(request.url_for("dashboard")) + f"?seed_id={seed.seed_id}"
    if _is_htmx(request):
        response = Response(status_code=204)
        response.headers["HX-Redirect"] = target_url
        return response
    return RedirectResponse(target_url, status_code=303)


@router.post("/actions/seeds/{seed_id}/pd", response_class=HTMLResponse)
def queue_pd(request: Request, seed_id: str) -> Response:
    workflow = _workflow(request)
    try:
        workflow.queue_pd(seed_id)
    except KeyError as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=404)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, GitCommandError) as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target_url = str(request.url_for("dashboard")) + f"?seed_id={seed_id}"
    if _is_htmx(request):
        response = Response(status_code=204)
        response.headers["HX-Redirect"] = target_url
        return response
    return RedirectResponse(target_url, status_code=303)


@router.post("/actions/seeds/{seed_id}/prompt", response_class=HTMLResponse)
def update_seed_prompt(request: Request, seed_id: str, prompt: str = Form(...)) -> Response:
    workflow = _workflow(request)
    try:
        workflow.update_seed_prompt(seed_id, prompt)
    except KeyError as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=404)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if _is_htmx(request):
        detail = workflow.seed_detail(seed_id)
        dashboard = workflow.build_dashboard(selected_seed_id=seed_id)
        context = {
            **detail,
            "dashboard": dashboard,
            "selected_seed_id": seed_id,
            "oob": True,
            "daemon_status": get_daemon_status(),
        }
        return _render(request, "partials/seed_detail_response.html", context)

    target_url = str(request.url_for("dashboard")) + f"?seed_id={seed_id}"
    return RedirectResponse(target_url, status_code=303)


@router.post("/actions/seeds/{seed_id}/ca", response_class=HTMLResponse)
def queue_ca(request: Request, seed_id: str) -> Response:
    workflow = _workflow(request)
    try:
        workflow.queue_ca(seed_id)
    except (KeyError, RuntimeError) as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target_url = str(request.url_for("dashboard")) + f"?seed_id={seed_id}"
    if _is_htmx(request):
        response = Response(status_code=204)
        response.headers["HX-Redirect"] = target_url
        return response
    return RedirectResponse(target_url, status_code=303)


@router.post("/actions/seeds/{seed_id}/ralph/start", response_class=HTMLResponse)
def start_ralph_loop(request: Request, seed_id: str) -> Response:
    workflow = _workflow(request)
    try:
        seed = workflow.set_ralph_loop(seed_id, True)
        if seed.status in {
            SeedStatus.draft,
            SeedStatus.generated,
            SeedStatus.passed,
            SeedStatus.failed,
            SeedStatus.promoted,
        }:
            workflow.queue_pd(seed_id)
    except KeyError as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=404)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, GitCommandError) as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=400)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    target_url = str(request.url_for("dashboard")) + f"?seed_id={seed_id}"
    if _is_htmx(request):
        response = Response(status_code=204)
        response.headers["HX-Redirect"] = target_url
        return response
    return RedirectResponse(target_url, status_code=303)


@router.post("/actions/seeds/{seed_id}/ralph/stop", response_class=HTMLResponse)
def stop_ralph_loop(request: Request, seed_id: str) -> Response:
    workflow = _workflow(request)
    try:
        workflow.set_ralph_loop(seed_id, False)
    except KeyError as exc:
        if _is_htmx(request):
            return _render(request, "partials/action_error.html", {"message": str(exc)}, status_code=404)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    target_url = str(request.url_for("dashboard")) + f"?seed_id={seed_id}"
    if _is_htmx(request):
        response = Response(status_code=204)
        response.headers["HX-Redirect"] = target_url
        return response
    return RedirectResponse(target_url, status_code=303)
