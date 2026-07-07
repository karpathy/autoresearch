"""Reusable HTTP scraping helpers for proxy rotation and retries."""

from __future__ import annotations

import itertools
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests


def normalize_proxy_url(raw_proxy: str) -> str | None:
    """Normalize one proxy-list entry to a proxy URL, or skip invalid entries."""
    value = raw_proxy.strip()
    if not value or value.startswith("#"):
        return None

    if "://" not in value:
        host_port_user_pass = value.split(":")
        if len(host_port_user_pass) == 4 and "@" not in value:
            host, port, username, password = host_port_user_pass
            if not host or not port or not username or not password:
                return None
            value = (
                f"http://{quote(username, safe='')}:{quote(password, safe='')}"
                f"@{host}:{port}"
            )
        else:
            value = f"http://{value}"

    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        return None
    try:
        parsed.port
    except ValueError:
        return None
    return value


def proxy_dict(proxy_url: str) -> dict[str, str]:
    """Return a requests proxy mapping for one proxy URL."""
    return {"http": proxy_url, "https": proxy_url}


def load_proxy_list(path: str | Path) -> list[dict[str, str]]:
    """Load normalized proxy mappings from a newline-delimited proxy file."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    proxies: list[dict[str, str]] = []
    for line in lines:
        proxy_url = normalize_proxy_url(line)
        if proxy_url is not None:
            proxies.append(proxy_dict(proxy_url))
    return proxies


def retry_count(env_name: str, default: int = 3, minimum: int = 1, maximum: int = 5) -> int:
    """Return an integer retry count from the environment, clamped to bounds."""
    try:
        retries = int(os.environ.get(env_name, str(default)))
    except ValueError:
        retries = default
    return min(max(retries, minimum), maximum)


def retry_delay_seconds(
    env_name: str,
    default: float = 1.0,
    minimum: float = 0.0,
    maximum: float = 10.0,
) -> float:
    """Return a retry delay from the environment, clamped to bounds."""
    try:
        delay = float(os.environ.get(env_name, str(default)))
    except ValueError:
        delay = default
    return min(max(delay, minimum), maximum)


class ScrapingEngine:
    """Small requests-compatible GET helper with proxy rotation and retries."""

    def __init__(
        self,
        *,
        proxy_url_env: str | None = None,
        proxy_list_path_env: str | None = None,
        retries_env: str | None = None,
        retry_delay_env: str | None = None,
        retry_status_codes: set[int] | frozenset[int] | tuple[int, ...] = (),
        request_get: Callable[..., Any] | None = None,
        rotation_counter: Iterator[int] | None = None,
    ) -> None:
        self.proxy_url_env = proxy_url_env
        self.proxy_list_path_env = proxy_list_path_env
        self.retries_env = retries_env
        self.retry_delay_env = retry_delay_env
        self.retry_status_codes = frozenset(retry_status_codes)
        self.request_get = request_get
        self.rotation_counter = rotation_counter if rotation_counter is not None else itertools.count()

    def _single_proxy_config(self) -> dict[str, str] | None:
        if self.proxy_url_env is None:
            return None
        proxy_url = os.environ.get(self.proxy_url_env, "").strip()
        if not proxy_url:
            return None
        return proxy_dict(proxy_url)

    def _proxy_list_configs(self) -> list[dict[str, str]]:
        if self.proxy_list_path_env is None:
            return []
        list_path = os.environ.get(self.proxy_list_path_env, "").strip()
        if not list_path:
            return []
        return load_proxy_list(list_path)

    def _retry_count(self) -> int:
        if self.retries_env is None:
            return 3
        return retry_count(self.retries_env)

    def _retry_delay_seconds(self) -> float:
        if self.retry_delay_env is None:
            return 1.0
        return retry_delay_seconds(self.retry_delay_env)

    def get(self, url: str, timeout: int | float, **kwargs: Any) -> Any:
        """GET a URL with optional proxy selection and transient retry handling."""
        configured_proxy = None
        rotating_proxies: list[dict[str, str]] = []
        if "proxies" not in kwargs:
            configured_proxy = self._single_proxy_config()
            rotating_proxies = [] if configured_proxy is not None else self._proxy_list_configs()

        proxy_start = next(self.rotation_counter) % len(rotating_proxies) if rotating_proxies else 0
        attempts = self._retry_count()
        delay = self._retry_delay_seconds()
        request_get = self.request_get or requests.get

        if configured_proxy is not None:
            kwargs.setdefault("proxies", configured_proxy)

        response = None
        last_exc: Exception | None = None
        for attempt in range(attempts):
            request_kwargs = dict(kwargs)
            if "proxies" not in request_kwargs and rotating_proxies:
                request_kwargs["proxies"] = rotating_proxies[
                    (proxy_start + attempt) % len(rotating_proxies)
                ]
            try:
                response = request_get(url, timeout=timeout, **request_kwargs)
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    raise
                if delay:
                    time.sleep(delay * (2**attempt))
                continue

            status_code = getattr(response, "status_code", None)
            if status_code not in self.retry_status_codes or attempt == attempts - 1:
                return response
            if delay:
                time.sleep(delay * (2**attempt))

        if last_exc is not None:
            raise last_exc
        return response
