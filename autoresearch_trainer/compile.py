from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import torch
import torch._inductor as inductor
import torch._inductor.cpp_builder as cpp_builder


HAS_TRITON = importlib.util.find_spec("triton") is not None
AVAILABLE_COMPILE_BACKENDS = set(torch.compiler.list_backends())
EXTRA_COMPILE_BACKENDS = {"aot_eager"}
AVAILABLE_INDUCTOR_MODES = sorted(inductor.list_mode_options().keys())


@dataclass(frozen=True)
class CompilePreparation:
    compile_kwargs: dict[str, Any] | None
    msvc_cl_path: str | None
    msvc_help_encoding: str | None


def resolve_compile_backend(requested_backend: str) -> str:
    if requested_backend == "auto":
        return "inductor" if HAS_TRITON else "off"
    return requested_backend


def resolve_optimizer_compile_backend(requested_backend: str, model_backend: str) -> str:
    if requested_backend == "auto":
        return "inductor" if model_backend == "inductor" else "off"
    return requested_backend


def validate_compile_backend(compile_backend: str) -> None:
    if compile_backend == "off":
        return
    if compile_backend not in AVAILABLE_COMPILE_BACKENDS and compile_backend not in EXTRA_COMPILE_BACKENDS:
        available = ", ".join(sorted(AVAILABLE_COMPILE_BACKENDS | EXTRA_COMPILE_BACKENDS))
        raise RuntimeError(f"Compile backend '{compile_backend}' is unavailable. Available backends: {available}")
    if compile_backend == "inductor" and not HAS_TRITON:
        raise RuntimeError("Compile backend 'inductor' requires Triton in this environment")


def validate_compile_mode(compile_backend: str, compile_mode: str) -> None:
    if compile_backend != "inductor":
        return
    if compile_mode not in AVAILABLE_INDUCTOR_MODES:
        available = ", ".join(AVAILABLE_INDUCTOR_MODES)
        raise RuntimeError(f"Compile mode '{compile_mode}' is unavailable. Available modes: {available}")


def find_vcvars64() -> str | None:
    if os.name != "nt":
        return None

    vswhere_candidates: list[str] = []
    env_vswhere = os.environ.get("VSWHERE")
    if env_vswhere:
        vswhere_candidates.append(env_vswhere)

    which_vswhere = shutil.which("vswhere.exe")
    if which_vswhere and which_vswhere not in vswhere_candidates:
        vswhere_candidates.append(which_vswhere)

    default_vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.isfile(default_vswhere) and default_vswhere not in vswhere_candidates:
        vswhere_candidates.append(default_vswhere)

    for vswhere in vswhere_candidates:
        try:
            result = subprocess.run(
                [
                    vswhere,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                check=True,
                capture_output=True,
                text=True,
                errors="ignore",
            )
        except (OSError, subprocess.CalledProcessError):
            continue

        install_path = result.stdout.strip()
        if not install_path:
            continue

        vcvars64 = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if os.path.isfile(vcvars64):
            return vcvars64

    fallback_matches: list[str] = []
    for pattern in (
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ):
        fallback_matches.extend(glob.glob(pattern))
    if fallback_matches:
        return sorted(fallback_matches, reverse=True)[0]
    return None


def load_windows_msvc_env(vcvars64_path: str) -> None:
    command = f'call "{vcvars64_path}" >nul && set'
    result = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        errors="ignore",
    )
    for line in result.stdout.splitlines():
        if "=" not in line or line.startswith("="):
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def ensure_windows_msvc_compiler() -> str | None:
    if os.name != "nt":
        return None

    compiler = shutil.which("cl.exe")
    if compiler is not None:
        os.environ.setdefault("CC", compiler)
        os.environ.setdefault("CXX", compiler)
        return compiler

    vcvars64_path = find_vcvars64()
    if vcvars64_path is None:
        raise RuntimeError(
            "Optimizer compile on Windows requires MSVC cl.exe, but no Visual Studio C++ toolchain was found. "
            "Install Visual Studio Build Tools with Desktop development with C++."
        )

    try:
        load_windows_msvc_env(vcvars64_path)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            detail = f" Details: {detail}"
        raise RuntimeError(f"Failed to load MSVC environment from '{vcvars64_path}'.{detail}") from exc

    compiler = shutil.which("cl.exe")
    if compiler is None:
        raise RuntimeError(
            f"Loaded MSVC environment from '{vcvars64_path}', but cl.exe is still unavailable in PATH."
        )

    os.environ.setdefault("CC", compiler)
    os.environ.setdefault("CXX", compiler)
    return compiler


def maybe_configure_msvc_utf8_help(compiler: str | None) -> str | None:
    if os.name != "nt" or compiler is None:
        return None

    help_output = subprocess.check_output([compiler, "/help"], stderr=subprocess.STDOUT)
    try:
        help_output.decode(*cpp_builder.SUBPROCESS_DECODE_ARGS)
        return None
    except UnicodeDecodeError:
        try:
            help_output.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"cl.exe help output from '{compiler}' could not be decoded using "
                f"{cpp_builder.SUBPROCESS_DECODE_ARGS[0]!r} or 'utf-8'."
            ) from exc

    cpp_builder.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
    cpp_builder._is_msvc_cl.cache_clear()
    return "utf-8"


def build_compile_kwargs(model_backend: str, compile_mode: str) -> dict[str, Any] | None:
    if model_backend == "off":
        return None
    compile_kwargs: dict[str, Any] = {"backend": model_backend, "dynamic": False}
    if model_backend == "inductor":
        compile_kwargs["mode"] = compile_mode
    return compile_kwargs


def prepare_compile_environment(
    *,
    model_backend: str,
    optimizer_backend: str,
    compile_mode: str,
) -> CompilePreparation:
    msvc_cl_path = None
    msvc_help_encoding = None
    if model_backend == "inductor" or optimizer_backend == "inductor":
        msvc_cl_path = ensure_windows_msvc_compiler()
        msvc_help_encoding = maybe_configure_msvc_utf8_help(msvc_cl_path)
    return CompilePreparation(
        compile_kwargs=build_compile_kwargs(model_backend, compile_mode),
        msvc_cl_path=msvc_cl_path,
        msvc_help_encoding=msvc_help_encoding,
    )


def maybe_compile_function(fn, *, backend: str, compile_mode: str, **kwargs):
    if backend == "off":
        return fn
    compile_kwargs: dict[str, Any] = {"backend": backend, **kwargs}
    if backend == "inductor":
        compile_kwargs["mode"] = compile_mode
    return torch.compile(fn, **compile_kwargs)
