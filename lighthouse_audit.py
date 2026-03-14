"""
Lighthouse audit runner for autoresearch.
Runs Lighthouse audits on the target web application and collects metrics.

Usage:
    python lighthouse_audit.py                     # run audit on default URLs
    python lighthouse_audit.py --url http://...    # run on specific URL

Requirements:
    - Chrome/Chromium browser installed
    - Lighthouse CLI: npm install -g lighthouse
    
    Or use Docker:
    docker run --rm --network=host -v $(pwd):/reports \
      ghcr.io/googlechromelabs/lighthouse:latest \
      http://localhost:8000 --output=json --output-path=/reports/report.json
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default URLs to audit (Default pages of your project)
DEFAULT_URLS = [
    "http://localhost:8000/",
]

# Target project path
TARGET_PROJECT = "/home/<Project/Directory>"

# Lighthouse categories to evaluate
CATEGORIES = ["performance", "accessibility", "best-practices", "seo"]

# Timeout for each audit (seconds)
AUDIT_TIMEOUT = 180

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LighthouseResult:
    """Results from a single Lighthouse audit."""
    url: str
    performance: float
    accessibility: float
    best_practices: float
    seo: float
    pwa: float
    fcp: float  # First Contentful Paint (ms)
    lcp: float  # Largest Contentful Paint (ms)
    tbt: float  # Total Blocking Time (ms)
    cls: float  # Cumulative Layout Shift
    si: float   # Speed Index (ms)
    tti: float  # Time to Interactive (ms)
    audit_time: float  # Time taken for audit (seconds)
    error: str | None = None


@dataclass
class AuditSummary:
    """Aggregated results from multiple audits."""
    performance: float
    accessibility: float
    best_practices: float
    seo: float
    pwa: float
    total_score: float
    audit_time_seconds: float
    url: str
    num_pages: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def check_lighthouse_installed() -> bool:
    """Check if Lighthouse CLI is available."""
    try:
        result = subprocess.run(
            ["npx", "lighthouse", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_docker() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_single_audit_native(url: str, output_path: str) -> LighthouseResult:
    """Run Lighthouse audit using native npx command."""
    cmd = [
        "npx", "lighthouse",
        url,
        "--output=json",
        f"--output-path={output_path}",
        "--quiet",
        "--chrome-flags=--no-sandbox --disable-dev-shm-usage --headless=new",
    ]
    
    for category in CATEGORIES:
        cmd.append(f"--only-categories={category}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AUDIT_TIMEOUT,
            cwd=TARGET_PROJECT if os.path.exists(TARGET_PROJECT) else None
        )
        
        audit_time = time.time() - start_time
        
        if result.returncode != 0 or not os.path.exists(output_path):
            return LighthouseResult(
                url=url, performance=0.0, accessibility=0.0, best_practices=0.0, seo=0.0, pwa=0.0,
                fcp=0.0, lcp=0.0, tbt=0.0, cls=0.0, si=0.0, tti=0.0,
                audit_time=audit_time, error=result.stderr or "Audit failed"
            )
        
        return parse_lighthouse_report(output_path, audit_time)
        
    except subprocess.TimeoutExpired:
        return LighthouseResult(
            url=url, performance=0.0, accessibility=0.0, best_practices=0.0, seo=0.0, pwa=0.0,
            fcp=0.0, lcp=0.0, tbt=0.0, cls=0.0, si=0.0, tti=0.0,
            audit_time=AUDIT_TIMEOUT, error="Audit timed out"
        )
    except Exception as e:
        return LighthouseResult(
            url=url, performance=0.0, accessibility=0.0, best_practices=0.0, seo=0.0, pwa=0.0,
            fcp=0.0, lcp=0.0, tbt=0.0, cls=0.0, si=0.0, tti=0.0,
            audit_time=time.time() - start_time, error=str(e)
        )


def run_single_audit_docker(url: str) -> LighthouseResult:
    """Run Lighthouse audit using Docker with the official lighthouse image."""
    start_time = time.time()
    
    # Use a container that has both node and chromium
    # Run lighthouse via a shell script inside the container
    safe_url = url.replace("'", "'\\''")
    
    # First, create a container and run lighthouse
    run_cmd = f"""
set -e
apt-get update -qq && apt-get install -y -qq curl gnupg > /dev/null 2>&1
curl -fsSL https://deb.nodesource.com/setup_20.x | bash - > /dev/null 2>&1
apt-get install -y -qq nodejs > /dev/null 2>&1
npm install -g lighthouse > /dev/null 2>&1
lighthouse '{safe_url}' \\
    --output=json \\
    --output-path=/tmp/lh-report.json \\
    --quiet \\
    --chrome-flags="--no-sandbox --disable-dev-shm-usage --headless=new" \\
    --only-categories=performance \\
    --only-categories=accessibility \\
    --only-categories=best-practices \\
    --only-categories=seo
cat /tmp/lh-report.json
"""
    
    cmd = [
        "docker", "run", "--rm", "--network=host",
        "node:20-slim",
        "bash", "-c", run_cmd
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=AUDIT_TIMEOUT
        )
        
        audit_time = time.time() - start_time
        
        if result.returncode != 0:
            return LighthouseResult(
                url=url, performance=0.0, accessibility=0.0, best_practices=0.0, seo=0.0, pwa=0.0,
                fcp=0.0, lcp=0.0, tbt=0.0, cls=0.0, si=0.0, tti=0.0,
                audit_time=audit_time, error=result.stderr[:500] if result.stderr else "Docker audit failed"
            )
        
        # The JSON report is printed to stdout
        report = json.loads(result.stdout)
        
        return extract_metrics_from_report(report, audit_time)
        
    except subprocess.TimeoutExpired:
        return LighthouseResult(
            url=url, performance=0.0, accessibility=0.0, best_practices=0.0, seo=0.0, pwa=0.0,
            fcp=0.0, lcp=0.0, tbt=0.0, cls=0.0, si=0.0, tti=0.0,
            audit_time=AUDIT_TIMEOUT, error="Docker audit timed out"
        )
    except json.JSONDecodeError as e:
        return LighthouseResult(
            url=url, performance=0.0, accessibility=0.0, best_practices=0.0, seo=0.0, pwa=0.0,
            fcp=0.0, lcp=0.0, tbt=0.0, cls=0.0, si=0.0, tti=0.0,
            audit_time=time.time() - start_time, error=f"Failed to parse JSON: {e}"
        )
    except Exception as e:
        return LighthouseResult(
            url=url, performance=0.0, accessibility=0.0, best_practices=0.0, seo=0.0, pwa=0.0,
            fcp=0.0, lcp=0.0, tbt=0.0, cls=0.0, si=0.0, tti=0.0,
            audit_time=time.time() - start_time, error=str(e)
        )


def parse_lighthouse_report(report_path: str, audit_time: float) -> LighthouseResult:
    """Parse a Lighthouse JSON report and extract metrics."""
    with open(report_path, 'r') as f:
        report = json.load(f)
    return extract_metrics_from_report(report, audit_time)


def extract_metrics_from_report(report: dict, audit_time: float) -> LighthouseResult:
    """Extract metrics from a Lighthouse report dictionary."""
    categories = report.get('categories', {})
    
    def get_score(cat_id: str) -> float:
        cat = categories.get(cat_id, {})
        score = cat.get('score', 0)
        return score * 100 if score is not None else 0.0
    
    audits = report.get('audits', {})
    
    def get_metric(audit_id: str) -> float:
        audit = audits.get(audit_id, {})
        value = audit.get('numericValue', 0)
        return value if value is not None else 0.0
    
    return LighthouseResult(
        url=report.get('requestedUrl', ''),
        performance=get_score('performance'),
        accessibility=get_score('accessibility'),
        best_practices=get_score('best-practices'),
        seo=get_score('seo'),
        pwa=get_score('pwa'),
        fcp=get_metric('first-contentful-paint'),
        lcp=get_metric('largest-contentful-paint'),
        tbt=get_metric('total-blocking-time'),
        cls=get_metric('cumulative-layout-shift'),
        si=get_metric('speed-index'),
        tti=get_metric('interactive'),
        audit_time=audit_time
    )


def run_single_audit(url: str, output_dir: str) -> LighthouseResult:
    """Run a single Lighthouse audit, trying native first, then Docker fallback."""
    output_path = os.path.join(output_dir, f"report-{int(time.time() * 1000)}.json")
    
    # Try native first
    if check_lighthouse_installed():
        print(f"  Using native Lighthouse CLI")
        return run_single_audit_native(url, output_path)
    
    # Fall back to Docker
    if check_docker():
        print(f"  Using Docker-based Lighthouse")
        return run_single_audit_docker(url)
    
    return LighthouseResult(
        url=url, performance=0.0, accessibility=0.0, best_practices=0.0, seo=0.0, pwa=0.0,
        fcp=0.0, lcp=0.0, tbt=0.0, cls=0.0, si=0.0, tti=0.0,
        audit_time=0, error="Neither Lighthouse CLI nor Docker is available"
    )


def run_audits(urls: list[str] = None) -> AuditSummary:
    """Run Lighthouse audits on multiple URLs and return aggregated results."""
    if urls is None:
        urls = DEFAULT_URLS
    
    with tempfile.TemporaryDirectory() as output_dir:
        results = []
        total_time = 0
        
        for url in urls:
            print(f"Running Lighthouse audit on: {url}")
            result = run_single_audit(url, output_dir)
            results.append(result)
            total_time += result.audit_time
            
            if result.error:
                print(f"  ⚠️  Warning: {result.error[:100]}...")
            else:
                print(f"  ✓ Performance: {result.performance:.1f}, Accessibility: {result.accessibility:.1f}, "
                      f"Best Practices: {result.best_practices:.1f}, SEO: {result.seo:.1f}")
        
        n = len(results)
        avg = AuditSummary(
            performance=round(sum(r.performance for r in results) / n, 1) if n > 0 else 0.0,
            accessibility=round(sum(r.accessibility for r in results) / n, 1) if n > 0 else 0.0,
            best_practices=round(sum(r.best_practices for r in results) / n, 1) if n > 0 else 0.0,
            seo=round(sum(r.seo for r in results) / n, 1) if n > 0 else 0.0,
            pwa=round(sum(r.pwa for r in results) / n, 1) if n > 0 else 0.0,
            total_score=round(
                (sum(r.performance for r in results) +
                 sum(r.accessibility for r in results) +
                 sum(r.best_practices for r in results) +
                 sum(r.seo for r in results)) / (4 * n), 1
            ) if n > 0 else 0.0,
            audit_time_seconds=round(total_time, 1),
            url=urls[0] if urls else "",
            num_pages=n
        )
        
        return avg


def print_summary(summary: AuditSummary):
    """Print audit summary in a format suitable for logging."""
    print("\n---")
    print(f"performance:        {summary.performance:.1f}")
    print(f"accessibility:      {summary.accessibility:.1f}")
    print(f"best_practices:     {summary.best_practices:.1f}")
    print(f"seo:                {summary.seo:.1f}")
    print(f"pwa:                {summary.pwa:.1f}")
    print(f"total_score:        {summary.total_score:.1f}")
    print(f"audit_time_seconds: {summary.audit_time_seconds:.1f}")
    print(f"url:                {summary.url}")
    print(f"num_pages:          {summary.num_pages}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Lighthouse audits on web application")
    parser.add_argument("--url", type=str, action="append", help="URL(s) to audit")
    args = parser.parse_args()
    
    urls = args.url if args.url else DEFAULT_URLS
    
    print(f"Target project: {TARGET_PROJECT}")
    print(f"URLs to audit: {urls}")
    print()
    
    summary = run_audits(urls)
    print_summary(summary)
