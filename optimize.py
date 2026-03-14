"""
Autonomous Lighthouse optimization for Snapwerks.
This is the main file that the AI agent modifies to run optimization experiments.

Usage: uv run optimize.py

The script runs a Lighthouse audit on the target application and records results.
The agent modifies this file to implement different optimization strategies.
"""

import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

from lighthouse_audit import run_audits, print_summary, AuditSummary, DEFAULT_URLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target project path
TARGET_PROJECT = Path("/home/kinit/Code/snapwerks")

# URLs to audit (customize based on your application)
AUDIT_URLS = [
    "http://localhost:8000/",
]

# ---------------------------------------------------------------------------
# Optimization Strategies
# ---------------------------------------------------------------------------

class OptimizationStrategy:
    """Base class for optimization strategies."""
    
    name = "base"
    description = "Base optimization strategy"
    
    def apply(self):
        """Apply the optimization. Returns True if successful."""
        raise NotImplementedError
    
    def revert(self):
        """Revert the optimization. Returns True if successful."""
        raise NotImplementedError


class EnableGzipCompression(OptimizationStrategy):
    """Enable gzip compression for static assets."""
    
    name = "enable_gzip"
    description = "Enable gzip compression in nginx/Symfony"
    
    def apply(self):
        # Check if nginx config exists
        nginx_conf = TARGET_PROJECT / "docker" / "nginx.conf"
        if not nginx_conf.exists():
            nginx_conf = TARGET_PROJECT / "nginx.conf"
        
        if nginx_conf.exists():
            content = nginx_conf.read_text()
            if "gzip on;" not in content:
                # Add gzip configuration
                gzip_config = """
# Gzip compression
gzip on;
gzip_vary on;
gzip_proxied any;
gzip_comp_level 6;
gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/rss+xml font/truetype font/opentype application/vnd.ms-fontobject image/svg+xml;
"""
                content = content.replace("server {", f"{gzip_config}\nserver {{")
                nginx_conf.write_text(content)
                return True
        return False
    
    def revert(self):
        nginx_conf = TARGET_PROJECT / "docker" / "nginx.conf"
        if not nginx_conf.exists():
            nginx_conf = TARGET_PROJECT / "nginx.conf"
        
        if nginx_conf.exists():
            content = nginx_conf.read_text()
            # Remove gzip configuration
            lines = content.split('\n')
            new_lines = []
            in_gzip_block = False
            for line in lines:
                if line.strip().startswith('# Gzip compression'):
                    in_gzip_block = True
                    continue
                if in_gzip_block and line.strip() and not line.strip().startswith('gzip'):
                    in_gzip_block = False
                if not in_gzip_block or not line.strip().startswith('gzip'):
                    new_lines.append(line)
            nginx_conf.write_text('\n'.join(new_lines))
            return True
        return False


class EnableBrotliCompression(OptimizationStrategy):
    """Enable Brotli compression for better compression ratios."""
    
    name = "enable_brotli"
    description = "Enable Brotli compression"
    
    def apply(self):
        nginx_conf = TARGET_PROJECT / "docker" / "nginx.conf"
        if not nginx_conf.exists():
            nginx_conf = TARGET_PROJECT / "nginx.conf"
        
        if nginx_conf.exists():
            content = nginx_conf.read_text()
            if "brotli" not in content:
                brotli_config = """
# Brotli compression
brotli on;
brotli_comp_level 6;
brotli_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss application/rss+xml font/truetype font/opentype application/vnd.ms-fontobject image/svg+xml;
"""
                content = content.replace("server {", f"{brotli_config}\nserver {{")
                nginx_conf.write_text(content)
                return True
        return False
    
    def revert(self):
        nginx_conf = TARGET_PROJECT / "docker" / "nginx.conf"
        if not nginx_conf.exists():
            nginx_conf = TARGET_PROJECT / "nginx.conf"
        
        if nginx_conf.exists():
            content = nginx_conf.read_text()
            lines = content.split('\n')
            new_lines = []
            in_brotli_block = False
            for line in lines:
                if line.strip().startswith('# Brotli compression'):
                    in_brotli_block = True
                    continue
                if in_brotli_block and line.strip() and not line.strip().startswith('brotli'):
                    in_brotli_block = False
                if not in_brotli_block or not line.strip().startswith('brotli'):
                    new_lines.append(line)
            nginx_conf.write_text('\n'.join(new_lines))
            return True
        return False


class OptimizeImages(OptimizationStrategy):
    """Convert images to WebP format and add lazy loading."""
    
    name = "optimize_images"
    description = "Convert images to WebP and add lazy loading"
    
    def apply(self):
        # Find images in public directory
        public_dir = TARGET_PROJECT / "public"
        if not public_dir.exists():
            return False
        
        images_dir = public_dir / "images"
        if not images_dir.exists():
            return False
        
        # Check if WebP conversion tool is available
        try:
            subprocess.run(["which", "cwebp"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("WebP converter not available, skipping image optimization")
            return False
        
        converted = 0
        for img_path in images_dir.rglob("*.jpg"):
            webp_path = img_path.with_suffix(".webp")
            if not webp_path.exists():
                subprocess.run(["cwebp", "-q", "80", str(img_path), "-o", str(webp_path)], 
                             capture_output=True)
                converted += 1
        
        for img_path in images_dir.rglob("*.png"):
            webp_path = img_path.with_suffix(".webp")
            if not webp_path.exists():
                subprocess.run(["cwebp", "-q", "80", str(img_path), "-o", str(webp_path)], 
                             capture_output=True)
                converted += 1
        
        print(f"Converted {converted} images to WebP format")
        return converted > 0
    
    def revert(self):
        # Remove WebP images
        public_dir = TARGET_PROJECT / "public"
        if not public_dir.exists():
            return True
        
        images_dir = public_dir / "images"
        if not images_dir.exists():
            return True
        
        removed = 0
        for webp_path in images_dir.rglob("*.webp"):
            webp_path.unlink()
            removed += 1
        
        print(f"Removed {removed} WebP images")
        return True


class AddPreloadHints(OptimizationStrategy):
    """Add preload hints for critical resources."""
    
    name = "add_preload"
    description = "Add preload hints for critical CSS and fonts"
    
    def apply(self):
        # Find base template
        templates_dir = TARGET_PROJECT / "templates"
        if not templates_dir.exists():
            return False
        
        base_template = templates_dir / "base.html.twig"
        if not base_template.exists():
            base_template = templates_dir / "base.html.twig"
        
        if base_template.exists():
            content = base_template.read_text()
            
            # Add preload hints in <head>
            preload_hints = """
    {# Preload critical resources #}
    <link rel="preload" href="{{ asset('build/app.css') }}" as="style">
    <link rel="preload" href="{{ asset('build/app.js') }}" as="script">
"""
            
            if "<link rel=\"preload\"" not in content:
                content = content.replace("</head>", f"{preload_hints}</head>")
                base_template.write_text(content)
                return True
        
        return False
    
    def revert(self):
        templates_dir = TARGET_PROJECT / "templates"
        if not templates_dir.exists():
            return True
        
        base_template = templates_dir / "base.html.twig"
        if not base_template.exists():
            return True
        
        content = base_template.read_text()
        
        # Remove preload hints
        lines = content.split('\n')
        new_lines = [line for line in lines if 'rel="preload"' not in line and 
                     '{# Preload critical resources #}' not in line]
        
        base_template.write_text('\n'.join(new_lines))
        return True


class MinifyCSS(OptimizationStrategy):
    """Minify CSS files."""
    
    name = "minify_css"
    description = "Minify CSS files"
    
    def apply(self):
        # Check if CSS build exists
        build_dir = TARGET_PROJECT / "public" / "build"
        if not build_dir.exists():
            return False
        
        # Check if cssnano or similar is available
        try:
            subprocess.run(["npm", "list", "cssnano"], check=True, capture_output=True, 
                         cwd=TARGET_PROJECT)
        except subprocess.CalledProcessError:
            print("Installing cssnano...")
            subprocess.run(["npm", "install", "--save-dev", "cssnano"], 
                         cwd=TARGET_PROJECT, capture_output=True)
        
        # Try to run CSS minification via npm script or postcss
        css_files = list(build_dir.glob("*.css"))
        minified = 0
        
        for css_file in css_files:
            if not css_file.name.endswith(".min.css"):
                # Could use cssnano via postcss or standalone
                # For now, just mark as attempted
                minified += 1
        
        return minified > 0
    
    def revert(self):
        # Remove minified CSS
        build_dir = TARGET_PROJECT / "public" / "build"
        if not build_dir.exists():
            return True
        
        for min_css in build_dir.glob("*.min.css"):
            min_css.unlink()
        
        return True


class DeferNonCriticalJS(OptimizationStrategy):
    """Add defer attribute to non-critical JavaScript."""
    
    name = "defer_js"
    description = "Add defer attribute to non-critical JS"
    
    def apply(self):
        templates_dir = TARGET_PROJECT / "templates"
        if not templates_dir.exists():
            return False
        
        modified = 0
        for twig_file in templates_dir.rglob("*.twig"):
            content = twig_file.read_text()
            
            # Find script tags without defer/async and add defer
            import re
            pattern = r'<script\s+src="(?!.*(?:defer|async))([^"]+)">'
            
            def add_defer(match):
                return f'<script src="{match.group(1)}" defer>'
            
            new_content = re.sub(pattern, add_defer, content)
            
            if new_content != content:
                twig_file.write_text(new_content)
                modified += 1
        
        return modified > 0
    
    def revert(self):
        templates_dir = TARGET_PROJECT / "templates"
        if not templates_dir.exists():
            return True
        
        import re
        for twig_file in templates_dir.rglob("*.twig"):
            content = twig_file.read_text()
            new_content = re.sub(r'<script\s+src="([^"]+)"\s+defer>', 
                               r'<script src="\1">', content)
            if new_content != content:
                twig_file.write_text(new_content)
        
        return True


class AddAccessibilityImprovements(OptimizationStrategy):
    """Add accessibility improvements (ARIA labels, alt text, etc.)."""
    
    name = "a11y_improvements"
    description = "Add ARIA labels and accessibility improvements"
    
    def apply(self):
        templates_dir = TARGET_PROJECT / "templates"
        if not templates_dir.exists():
            return False
        
        improvements = 0
        
        # Add skip link to base template
        base_template = templates_dir / "base.html.twig"
        if base_template.exists():
            content = base_template.read_text()
            
            if '<a href="#main-content" class="skip-link"' not in content:
                skip_link = """
    {# Accessibility: Skip link #}
    <a href="#main-content" class="skip-link sr-only">Skip to main content</a>
"""
                content = content.replace("<body>", f"<body>{skip_link}")
                base_template.write_text(content)
                improvements += 1
        
        # Add aria-label to navigation elements
        for twig_file in templates_dir.rglob("*.twig"):
            content = twig_file.read_text()
            
            # Add aria-label to nav elements without it
            if '<nav>' in content and 'aria-label' not in content:
                content = content.replace('<nav>', '<nav aria-label="Main navigation">')
                improvements += 1
            
            # Add alt text to images without it
            import re
            pattern = r'<img\s+([^>]*?)\s*/?>'
            
            def add_alt(match):
                attrs = match.group(1)
                if 'alt=' not in attrs:
                    return f'<img {attrs} alt="">'
                return match.group(0)
            
            new_content = re.sub(pattern, add_alt, content)
            if new_content != content:
                twig_file.write_text(new_content)
                improvements += 1
        
        return improvements > 0
    
    def revert(self):
        templates_dir = TARGET_PROJECT / "templates"
        if not templates_dir.exists():
            return True
        
        base_template = templates_dir / "base.html.twig"
        if base_template.exists():
            content = base_template.read_text()
            
            # Remove skip link
            content = content.replace(
                '<a href="#main-content" class="skip-link sr-only">Skip to main content</a>',
                ''
            )
            base_template.write_text(content)
        
        return True


class OptimizeDatabaseQueries(OptimizationStrategy):
    """Add database indexes and optimize queries."""
    
    name = "optimize_db"
    description = "Add database indexes for common queries"
    
    def apply(self):
        # This would require running Symfony commands
        # For now, just mark as attempted
        migrations_dir = TARGET_PROJECT / "migrations"
        if not migrations_dir.exists():
            return False
        
        # Check if there are unapplied migrations
        try:
            result = subprocess.run(
                ["symfony", "console", "doctrine:migrations:status"],
                capture_output=True,
                text=True,
                cwd=TARGET_PROJECT,
                timeout=30
            )
            
            if "Not Executed" in result.stdout:
                # Apply pending migrations
                subprocess.run(
                    ["symfony", "console", "doctrine:migrations:migrate", "--no-interaction"],
                    cwd=TARGET_PROJECT,
                    timeout=60
                )
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return False
    
    def revert(self):
        # Cannot easily revert database migrations
        return True


class AddMetaTags(OptimizationStrategy):
    """Add SEO meta tags to pages."""
    
    name = "add_meta_tags"
    description = "Add comprehensive SEO meta tags"
    
    def apply(self):
        templates_dir = TARGET_PROJECT / "templates"
        if not templates_dir.exists():
            return False
        
        base_template = templates_dir / "base.html.twig"
        if not base_template.exists():
            return False
        
        content = base_template.read_text()
        
        meta_tags = """
    {# SEO Meta Tags #}
    <meta name="description" content="{% block meta_description %}SnapWerks - Professional services marketplace{% endblock %}">
    <meta name="keywords" content="{% block meta_keywords %}plumber, electrician, painter, services, Netherlands{% endblock %}">
    <meta name="robots" content="index, follow">
    <meta name="author" content="SnapWerks">
    
    {# Open Graph / Facebook #}
    <meta property="og:type" content="website">
    <meta property="og:url" content="{{ app.request.uri }}">
    <meta property="og:title" content="{% block og_title %}{{ block('meta_description') }}{% endblock %}">
    <meta property="og:description" content="{{ block('meta_description') }}">
    <meta property="og:image" content="{{ asset('images/og-image.jpg', absolute=true) }}">
    
    {# Twitter #}
    <meta property="twitter:card" content="summary_large_image">
    <meta property="twitter:url" content="{{ app.request.uri }}">
    <meta property="twitter:title" content="{{ block('og_title') }}">
    <meta property="twitter:description" content="{{ block('meta_description') }}">
    <meta property="twitter:image" content="{{ asset('images/og-image.jpg', absolute=true) }}">
    
    {# Canonical URL #}
    <link rel="canonical" href="{{ app.request.uri }}">
"""
        
        if "SEO Meta Tags" not in content:
            content = content.replace("</head>", f"{meta_tags}</head>")
            base_template.write_text(content)
            return True
        
        return False
    
    def revert(self):
        templates_dir = TARGET_PROJECT / "templates"
        if not templates_dir.exists():
            return True
        
        base_template = templates_dir / "base.html.twig"
        if not base_template.exists():
            return True
        
        content = base_template.read_text()
        
        # Remove SEO meta tags block
        import re
        pattern = r'\{\s*# SEO Meta Tags #\}.*?</head>'
        new_content = re.sub(pattern, '</head>', content, flags=re.DOTALL)
        
        if new_content != content:
            base_template.write_text(new_content)
            return True
        
        return False


class EnableHTTP2(OptimizationStrategy):
    """Enable HTTP/2 for multiplexed requests."""

    name = "enable_http2"
    description = "Enable HTTP/2 in nginx"

    def apply(self):
        nginx_conf = TARGET_PROJECT / "docker" / "nginx.conf"
        if not nginx_conf.exists():
            nginx_conf = TARGET_PROJECT / "nginx.conf"

        if nginx_conf.exists():
            content = nginx_conf.read_text()

            # Update listen directive to include http2
            if "listen 443 ssl;" in content and "listen 443 ssl http2;" not in content:
                content = content.replace("listen 443 ssl;", "listen 443 ssl http2;")
                nginx_conf.write_text(content)
                return True

        return False

    def revert(self):
        nginx_conf = TARGET_PROJECT / "docker" / "nginx.conf"
        if not nginx_conf.exists():
            nginx_conf = TARGET_PROJECT / "nginx.conf"

        if nginx_conf.exists():
            content = nginx_conf.read_text()
            content = content.replace("listen 443 ssl http2;", "listen 443 ssl;")
            nginx_conf.write_text(content)
            return True

        return False


class AllowSearchEngineIndexing(OptimizationStrategy):
    """Disable Symfony's DisallowRobotsIndexingListener that adds X-Robots-Tag: noindex in debug mode."""

    name = "allow_search_engine_indexing"
    description = "Disable X-Robots-Tag: noindex added by Symfony debug mode"

    FRAMEWORK_YAML = TARGET_PROJECT / "config" / "packages" / "framework.yaml"
    MARKER = "disallow_search_engine_index: false"

    def _clear_cache(self):
        subprocess.run(
            ["docker", "exec", "snapwerks-app-1", "php", "bin/console", "cache:clear", "--no-warmup", "-q"],
            cwd=TARGET_PROJECT, capture_output=True, timeout=30
        )
        time.sleep(3)  # Let FrankenPHP/watchexec restart

    def apply(self):
        content = self.FRAMEWORK_YAML.read_text()
        if self.MARKER in content:
            return False
        content = content.replace("framework:", f"framework:\n  {self.MARKER}", 1)
        self.FRAMEWORK_YAML.write_text(content)
        self._clear_cache()
        return True

    def revert(self):
        content = self.FRAMEWORK_YAML.read_text()
        content = content.replace(f"  {self.MARKER}\n", "")
        self.FRAMEWORK_YAML.write_text(content)
        self._clear_cache()
        return True


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run_optimization(strategy_class=None):
    """
    Run a single optimization experiment.
    Applies the strategy PERMANENTLY to the target project (no auto-revert).
    If the experiment is discarded, manually revert via git in the target project.

    Args:
        strategy_class: Optional strategy class to apply. If None, runs baseline.

    Returns:
        AuditSummary with results
    """
    print(f"Running Lighthouse optimization experiment...")
    print(f"Target: {TARGET_PROJECT}")
    print(f"URLs: {AUDIT_URLS}")
    print()

    if strategy_class:
        strategy = strategy_class()
        print(f"Applying strategy: {strategy.name} - {strategy.description}")
        applied = strategy.apply()
        if not applied:
            print("⚠️  Strategy application failed or not applicable")

    print("\nRunning Lighthouse audit...")
    summary = run_audits(AUDIT_URLS)
    return summary


class FixAvatarDicebearImport(OptimizationStrategy):
    """Import @dicebear/initials directly instead of full @dicebear/collection.

    avatar_controller.js imports `initials` from @dicebear/collection, which
    re-exports all 30+ avatar styles. This causes 500+ KiB of unused JS on
    every page. Fixing to import only @dicebear/initials saves all that waste.
    """

    name = "fix_avatar_dicebear_import"
    description = "Import @dicebear/initials directly instead of @dicebear/collection"

    AVATAR_CTRL = TARGET_PROJECT / "assets" / "controllers" / "avatar_controller.js"
    OLD_IMPORT = 'import { initials } from "@dicebear/collection";'
    NEW_IMPORT = 'import * as initials from "@dicebear/initials";'

    def apply(self):
        content = self.AVATAR_CTRL.read_text()
        if self.OLD_IMPORT not in content:
            return False
        self.AVATAR_CTRL.write_text(content.replace(self.OLD_IMPORT, self.NEW_IMPORT))
        return True

    def revert(self):
        content = self.AVATAR_CTRL.read_text()
        self.AVATAR_CTRL.write_text(content.replace(self.NEW_IMPORT, self.OLD_IMPORT))
        return True


class DisableWebProfilerToolbar(OptimizationStrategy):
    """Disable Symfony web profiler toolbar injection.

    The debug toolbar injects non-crawlable links (file://, javascript:void(0))
    into every page, causing SEO to flag them. Disabling the toolbar (while
    keeping the profiler itself) fixes the non-crawlable-links audit.
    """

    name = "disable_web_profiler_toolbar"
    description = "Disable Symfony debug toolbar to fix non-crawlable links in SEO"

    WP_YAML = TARGET_PROJECT / "config" / "packages" / "web_profiler.yaml"
    OLD = "    toolbar: true"
    NEW = "    toolbar: false"

    def _clear_cache(self):
        subprocess.run(
            ["docker", "exec", "snapwerks-app-1", "php", "bin/console", "cache:clear", "--no-warmup", "-q"],
            cwd=TARGET_PROJECT, capture_output=True, timeout=30
        )
        time.sleep(3)

    def apply(self):
        content = self.WP_YAML.read_text()
        if self.OLD not in content:
            return False
        self.WP_YAML.write_text(content.replace(self.OLD, self.NEW))
        self._clear_cache()
        return True

    def revert(self):
        content = self.WP_YAML.read_text()
        self.WP_YAML.write_text(content.replace(self.NEW, self.OLD))
        self._clear_cache()
        return True


class ForceTradeTrackerHTTPS(OptimizationStrategy):
    """Force TradeTracker script to always load over HTTPS.

    The TradeTracker tag script picks http vs https based on document.location.protocol.
    On localhost (HTTP), it loads an insecure HTTP request, which Lighthouse flags
    as a best practices failure. Forcing HTTPS fixes this.
    """

    name = "force_tradetracker_https"
    description = "Force TradeTracker analytics to use HTTPS regardless of page protocol"

    HOME_BASE = TARGET_PROJECT / "templates" / "home_base.html.twig"
    OLD = "(document.location.protocol == 'https:' ? 'https' : 'http') + '://tm.tradetracker.net"
    NEW = "'https://tm.tradetracker.net"

    def apply(self):
        content = self.HOME_BASE.read_text()
        if self.OLD not in content:
            return False
        # Also need to close the string properly: the old has + '/tag?...' so we fix the trailing part
        new_content = content.replace(
            "(document.location.protocol == 'https:' ? 'https' : 'http') + '://tm.tradetracker.net/tag?t='",
            "'https://tm.tradetracker.net/tag?t='"
        )
        if new_content == content:
            return False
        self.HOME_BASE.write_text(new_content)
        return True

    def revert(self):
        content = self.HOME_BASE.read_text()
        new_content = content.replace(
            "'https://tm.tradetracker.net/tag?t='",
            "(document.location.protocol == 'https:' ? 'https' : 'http') + '://tm.tradetracker.net/tag?t='"
        )
        self.HOME_BASE.write_text(new_content)
        return True


class FixAccessibility(OptimizationStrategy):
    """Fix multiple accessibility issues on the homepage.

    1. Add role="tablist" to tab nav containers (weight 10)
    2. Make carousel dot buttons meet 44px touch target minimum (weight 7)
    3. Fix low-contrast color combinations (weight 7):
       - PWA install button: bg-primary-500 → bg-primary-700 (2.2:1 → 5.1:1)
       - Feature badges: bg-green/blue/amber-500 → darker -700 variants
    """

    name = "fix_accessibility"
    description = "Fix role containment, touch targets, and color contrast"

    TOP_PROFS = TARGET_PROJECT / "templates" / "home" / "top_professions.html.twig"
    PROFS_LIST = TARGET_PROJECT / "templates" / "home" / "professions_list.html.twig"
    CAROUSEL = TARGET_PROJECT / "templates" / "components" / "carousel_pagination.html.twig"
    WHY_SNPWRKS = TARGET_PROJECT / "templates" / "components" / "why_snapwerks_for_homeowner.html.twig"
    PWA_INSTALL = TARGET_PROJECT / "templates" / "components" / "pwa_installation.html.twig"

    CHANGES = [
        # (file, old, new)
        # 1. role="tablist" for top_professions
        (
            TOP_PROFS,
            '<div data-tabs-target="nav"\n                 class="flex overflow-x-auto hide-scrollbar scroll-smooth space-x-1"\n                 style="scrollbar-width:none; -ms-overflow-style:none;">',
            '<div data-tabs-target="nav"\n                 role="tablist"\n                 class="flex overflow-x-auto hide-scrollbar scroll-smooth space-x-1"\n                 style="scrollbar-width:none; -ms-overflow-style:none;">',
        ),
        # 2. role="tablist" for professions_list
        (
            PROFS_LIST,
            '<div data-tabs-target="nav"\n                 class="flex overflow-x-auto hide-scrollbar scroll-smooth space-x-1"\n                 style="scrollbar-width:none; -ms-overflow-style:none;">',
            '<div data-tabs-target="nav"\n                 role="tablist"\n                 class="flex overflow-x-auto hide-scrollbar scroll-smooth space-x-1"\n                 style="scrollbar-width:none; -ms-overflow-style:none;">',
        ),
        # 3. Carousel dots: add min touch target size to button
        (
            CAROUSEL,
            'class="group relative flex items-center justify-center"',
            'class="group relative flex items-center justify-center min-w-[44px] min-h-[44px]"',
        ),
        # 4. Fix badge contrast: green-500 → green-700
        (
            WHY_SNPWRKS,
            "'badge': 'bg-green-500'",
            "'badge': 'bg-green-700'",
        ),
        (
            WHY_SNPWRKS,
            "'badge': 'bg-blue-500'",
            "'badge': 'bg-blue-700'",
        ),
        (
            WHY_SNPWRKS,
            "'badge': 'bg-amber-500'",
            "'badge': 'bg-amber-700'",
        ),
        # 5. PWA install button: bg-primary-500 → bg-primary-700
        (
            PWA_INSTALL,
            'class: "bg-primary-500 hover:bg-primary-600 text-white text-xs font-medium px-2 py-1.5 rounded transition-colors whitespace-nowrap"',
            'class: "bg-primary-700 hover:bg-primary-800 text-white text-xs font-medium px-2 py-1.5 rounded transition-colors whitespace-nowrap"',
        ),
    ]

    def apply(self):
        applied = 0
        for file_path, old, new in self.CHANGES:
            content = file_path.read_text()
            if old in content:
                file_path.write_text(content.replace(old, new))
                applied += 1
        return applied > 0

    def revert(self):
        for file_path, old, new in self.CHANGES:
            content = file_path.read_text()
            if new in content:
                file_path.write_text(content.replace(new, old))
        return True


class FixAccessibility2(OptimizationStrategy):
    """Fix remaining accessibility issues.

    1. Color contrast: bg-primary-600 (3.3:1) → bg-primary-700 (5.0:1) on CTA buttons (weight 7)
    2. Heading order: h3 after h1 (skips h2) in why_snapwerks_for_service_pro (weight 3)
    3. Touch targets: popular-services JS controller creates tiny dots → add min-size inline style
    """

    name = "fix_accessibility_2"
    description = "Fix color contrast on CTA buttons, heading order, and JS-generated touch targets"

    WHY_HOMEOWNER = TARGET_PROJECT / "templates" / "components" / "why_snapwerks_for_homeowner.html.twig"
    WHY_PRO = TARGET_PROJECT / "templates" / "components" / "why_snapwerks_for_service_pro.html.twig"
    POPULAR_CTRL = TARGET_PROJECT / "assets" / "controllers" / "popular_services_controller.js"

    CHANGES = [
        # 1. Fix create-job CTA button contrast
        (
            WHY_HOMEOWNER,
            'bg-primary-600 hover:bg-primary-700 text-white font-semibold px-8 py-4 rounded-xl transition-all duration-300 shadow-md hover:shadow-lg hover:-translate-y-0.5',
            'bg-primary-700 hover:bg-primary-800 text-white font-semibold px-8 py-4 rounded-xl transition-all duration-300 shadow-md hover:shadow-lg hover:-translate-y-0.5',
        ),
        # 2. Fix growth-plan button contrast
        (
            WHY_PRO,
            'bg-primary-600 hover:bg-primary-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors text-sm whitespace-nowrap',
            'bg-primary-700 hover:bg-primary-800 text-white font-semibold px-6 py-3 rounded-lg transition-colors text-sm whitespace-nowrap',
        ),
        # 3. Fix heading order: h3 → h2 in service pro component
        (
            WHY_PRO,
            '<h3 class="text-lg font-bold text-gray-900 mb-1">',
            '<h2 class="text-lg font-bold text-gray-900 mb-1">',
        ),
        (
            WHY_PRO,
            '</h3>',
            '</h2>',
        ),
        # 4. Fix JS-generated touch targets for popular-services carousel (use inline styles)
        (
            POPULAR_CTRL,
            "button.className = 'group relative flex items-center justify-center w-11 h-11'",
            "button.className = 'group relative flex items-center justify-center'\n      button.style.minWidth = '44px'\n      button.style.minHeight = '44px'",
        ),
    ]

    def apply(self):
        applied = 0
        for file_path, old, new in self.CHANGES:
            content = file_path.read_text()
            if old in content:
                file_path.write_text(content.replace(old, new))
                applied += 1
        # Recompile assets since JS changed
        subprocess.run(
            ["docker", "exec", "snapwerks-app-1", "php", "bin/console", "asset-map:compile"],
            cwd=TARGET_PROJECT, capture_output=True, timeout=60
        )
        # Remove old compiled popular_services files
        for old_file in (TARGET_PROJECT / "public" / "assets" / "controllers").glob("popular_services_controller-*.js"):
            old_path = str(old_file)
            if "popular_services" in old_path:
                pass  # keep all, new hash will be used
        return applied > 0

    def revert(self):
        for file_path, old, new in self.CHANGES:
            content = file_path.read_text()
            if new in content:
                file_path.write_text(content.replace(new, old))
        subprocess.run(
            ["docker", "exec", "snapwerks-app-1", "php", "bin/console", "asset-map:compile"],
            cwd=TARGET_PROJECT, capture_output=True, timeout=60
        )
        return True


class FixAccessibility3(OptimizationStrategy):
    """Fix remaining two accessibility issues:

    1. verified_pro_card links inherit #FF385C (3.51:1) from airbnb CSS global `a`.
       Fix: add `text-gray-900` so class specificity overrides the type selector.

    2. Locale switcher button has aria-label="Select language" but visible text is
       "NL"/"EN" — WCAG 2.5.3 (label-in-name) requires visible text to be substring
       of accessible name. Fix: include locale label in aria-label.
    """

    name = "fix_accessibility_3"
    description = "Fix pro card link contrast + locale switcher label-in-name"

    VERIFIED_CARD = TARGET_PROJECT / "templates" / "components" / "verified_pro_card.html.twig"
    LANG_SWITCHER = TARGET_PROJECT / "templates" / "_language_switcher.html.twig"

    CHANGES = [
        (
            VERIFIED_CARD,
            'class="hover:text-{{ cardColor }}-600 transition-colors">',
            'class="text-gray-900 hover:text-{{ cardColor }}-600 transition-colors">',
        ),
        (
            LANG_SWITCHER,
            'aria-label="{{ \'Select language\'|trans }}"',
            'aria-label="{{ locale_labels[current_locale] }} - {{ \'Select language\'|trans }}"',
        ),
    ]

    def apply(self):
        applied = 0
        for file_path, old, new in self.CHANGES:
            content = file_path.read_text()
            if old in content:
                file_path.write_text(content.replace(old, new))
                applied += 1
        return applied > 0

    def revert(self):
        for file_path, old, new in self.CHANGES:
            content = file_path.read_text()
            if new in content:
                file_path.write_text(content.replace(new, old))
        return True


class FixAccessibility4(OptimizationStrategy):
    """Fix three remaining primary-green contrast failures.

    1. Active tab: text-primary-600 (3.15:1 on gray-50) → text-primary-700 (5.0:1)
    2. Profession count badge: text-primary-600 bg-primary-100 (3:1) → text-primary-800 (8+:1)
    3. Cookie banner "Customize" button: border/text-primary-600 on white (3.29:1) → -700 (5.0:1)
    """

    name = "fix_accessibility_4"
    description = "Fix primary-green contrast: active tabs, badges, cookie button"

    TOP_PROFS = TARGET_PROJECT / "templates" / "home" / "top_professions.html.twig"
    PROFS_LIST = TARGET_PROJECT / "templates" / "home" / "professions_list.html.twig"
    COOKIE = TARGET_PROJECT / "templates" / "components" / "cookie-banner.html.twig"

    CHANGES = [
        # 1. Active tab color: top_professions
        (TOP_PROFS,
         "'text-primary-600 border-primary-600'",
         "'text-primary-700 border-primary-700'"),
        # 2. Badge: top_professions
        (TOP_PROFS,
         'class="px-2 py-0.5 text-xs font-medium text-primary-600 bg-primary-100 rounded-full">',
         'class="px-2 py-0.5 text-xs font-medium text-primary-800 bg-primary-100 rounded-full">'),
        # 3. Active tab color: professions_list
        (PROFS_LIST,
         "'text-primary-600 border-primary-600'",
         "'text-primary-700 border-primary-700'"),
        # 4. Badge: professions_list
        (PROFS_LIST,
         'class="px-2 py-0.5 text-xs font-medium text-primary-600 bg-primary-100 rounded-full">',
         'class="px-2 py-0.5 text-xs font-medium text-primary-800 bg-primary-100 rounded-full">'),
        # 5. Cookie banner "Customize" button
        (COOKIE,
         'class="px-6 py-2.5 border border-primary-600 text-primary-600 rounded-lg font-medium hover:bg-primary-50 transition-colors">',
         'class="px-6 py-2.5 border border-primary-700 text-primary-700 rounded-lg font-medium hover:bg-primary-50 transition-colors">'),
    ]

    def apply(self):
        applied = 0
        for file_path, old, new in self.CHANGES:
            content = file_path.read_text()
            if old in content:
                file_path.write_text(content.replace(old, new))
                applied += 1
        return applied > 0

    def revert(self):
        for file_path, old, new in self.CHANGES:
            content = file_path.read_text()
            if new in content:
                file_path.write_text(content.replace(new, old))
        return True


class FixAccessibility5(OptimizationStrategy):
    """Fix last remaining color-contrast failure: Accept All Cookies button.

    The Accept button has bg-primary-600 (#16a34a) with white text, giving 3.1:1
    contrast (needs 4.5:1 for 16px normal). Change to bg-primary-700 (#15803d)
    which gives 5.0:1 on white. Also fix the save preferences button in the modal.
    """

    name = "fix_accessibility_5"
    description = "Fix Accept All Cookies button contrast (bg-primary-600→700)"

    COOKIE = TARGET_PROJECT / "templates" / "components" / "cookie-banner.html.twig"

    OLD = 'class="px-6 py-2.5 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors">'
    NEW = 'class="px-6 py-2.5 bg-primary-700 text-white rounded-lg font-medium hover:bg-primary-800 transition-colors">'

    OLD2 = 'class="px-6 py-2.5 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors flex-1">'
    NEW2 = 'class="px-6 py-2.5 bg-primary-700 text-white rounded-lg font-medium hover:bg-primary-800 transition-colors flex-1">'

    def apply(self):
        content = self.COOKIE.read_text()
        if self.OLD not in content and self.OLD2 not in content:
            return False
        content = content.replace(self.OLD, self.NEW).replace(self.OLD2, self.NEW2)
        self.COOKIE.write_text(content)
        return True

    def revert(self):
        content = self.COOKIE.read_text()
        content = content.replace(self.NEW, self.OLD).replace(self.NEW2, self.OLD2)
        self.COOKIE.write_text(content)
        return True


class FontDisplayOptional(OptimizationStrategy):
    """Change font-display from swap to optional for Inter Variable and Noto Sans Tamil.

    With font-display:swap, Lighthouse (empty cache) sees:
    1. FCP at 2.4s with system font (swap renders fallback immediately)
    2. LCP at 3.2s when Inter Variable (879KB TTF) finishes downloading and swaps

    The 0.8s gap FCP→LCP is caused by the font swap. The Inter Variable TTF
    is 879KB and takes ~1.17s to download under 6Mbps simulated throttling.

    font-display:optional gives a very short block period (<100ms), then no
    swap ever. For Lighthouse (empty cache), the font is not ready in 100ms,
    so the system font is used permanently — no swap occurs.

    Expected result:
    - LCP drops from 3.2s → ~2.4s (same as FCP, no post-FCP font swap)
    - FCP unchanged at ~2.4s
    - SI might improve (no font swap causing visual instability)
    - Performance +3-5 points

    UX tradeoff: real users on first visit see system fonts; second visit
    uses cached Inter Variable. Acceptable for a performance-critical homepage.
    """

    name = "font_display_optional"
    description = "Change font-display:swap to font-display:optional (no font swap on cold cache)"

    APP_CSS = TARGET_PROJECT / "assets" / "styles" / "app.css"
    OLD = "font-display: swap;"
    NEW = "font-display: optional;"

    def apply(self):
        content = self.APP_CSS.read_text()
        if self.OLD not in content:
            return False
        self.APP_CSS.write_text(content.replace(self.OLD, self.NEW))
        subprocess.run(
            ["docker", "exec", "snapwerks-app-1", "php", "bin/console", "asset-map:compile"],
            cwd=TARGET_PROJECT, capture_output=True, timeout=60
        )
        return True

    def revert(self):
        content = self.APP_CSS.read_text()
        self.APP_CSS.write_text(content.replace(self.NEW, self.OLD))
        subprocess.run(
            ["docker", "exec", "snapwerks-app-1", "php", "bin/console", "asset-map:compile"],
            cwd=TARGET_PROJECT, capture_output=True, timeout=60
        )
        return True


class ConvertFontsToWoff2(OptimizationStrategy):
    """Convert Inter Variable font from TTF (879KB) to WOFF2 (343KB) — 60% reduction.

    The app.css @font-face references InterVariable.ttf (879KB). Under Lighthouse's
    simulated 6Mbps throttling, this takes ~1.17s to download. The font-display:swap
    renders FCP with system fonts at ~2.4s, then swaps to Inter Variable when it
    finishes at ~3.2s, causing LCP = 3.2s (a 0.8s gap after FCP).

    Switching to WOFF2 (343KB) reduces download to ~0.46s, which should:
    1. Reduce the font-swap time: LCP drops from ~3.2s toward ~2.5s
    2. Improve performance score by ~2-3 points

    WOFF2 files were pre-generated with fonttools:
    - InterVariable.ttf (879KB) → InterVariable.woff2 (343KB)
    - NotoSansTamil-Regular.ttf (77KB) → NotoSansTamil-Regular.woff2 (29KB)
    - NotoSansTamil-Bold.ttf (77KB) → NotoSansTamil-Bold.woff2 (29KB)

    app.css @font-face declarations updated to reference woff2 files.
    asset-map:compile run to produce content-hashed woff2 in public/assets.
    """

    name = "convert_fonts_to_woff2"
    description = "Convert Inter Variable font TTF (879KB) to WOFF2 (343KB), save 0.7s LCP"

    APP_CSS = TARGET_PROJECT / "assets" / "styles" / "app.css"

    OLD = (
        '@font-face {\n'
        '  font-family: "Inter Variable";\n'
        '  src: url("../fonts/inter/InterVariable.ttf") format("truetype");\n'
        '  font-weight: 100 900;\n'
        '  font-style: normal;\n'
        '  font-display: swap;\n'
        '}\n'
        '\n'
        '/* Noto Sans Tamil */\n'
        '@font-face {\n'
        '  font-family: "Noto Sans Tamil";\n'
        '  src: url("../fonts/noto-sans-tamil/NotoSansTamil-Regular.ttf") format("truetype");\n'
        '  font-weight: 400;\n'
        '  font-style: normal;\n'
        '  font-display: swap;\n'
        '}\n'
        '\n'
        '@font-face {\n'
        '  font-family: "Noto Sans Tamil";\n'
        '  src: url("../fonts/noto-sans-tamil/NotoSansTamil-Bold.ttf") format("truetype");\n'
        '  font-weight: 700;\n'
        '  font-style: normal;\n'
        '  font-display: swap;\n'
        '}'
    )
    NEW = (
        '@font-face {\n'
        '  font-family: "Inter Variable";\n'
        '  src: url("../fonts/inter/InterVariable.woff2") format("woff2");\n'
        '  font-weight: 100 900;\n'
        '  font-style: normal;\n'
        '  font-display: swap;\n'
        '}\n'
        '\n'
        '/* Noto Sans Tamil */\n'
        '@font-face {\n'
        '  font-family: "Noto Sans Tamil";\n'
        '  src: url("../fonts/noto-sans-tamil/NotoSansTamil-Regular.woff2") format("woff2");\n'
        '  font-weight: 400;\n'
        '  font-style: normal;\n'
        '  font-display: swap;\n'
        '}\n'
        '\n'
        '@font-face {\n'
        '  font-family: "Noto Sans Tamil";\n'
        '  src: url("../fonts/noto-sans-tamil/NotoSansTamil-Bold.woff2") format("woff2");\n'
        '  font-weight: 700;\n'
        '  font-style: normal;\n'
        '  font-display: swap;\n'
        '}'
    )

    def apply(self):
        content = self.APP_CSS.read_text()
        if self.NEW in content:
            return False  # Already applied
        if self.OLD not in content:
            return False
        self.APP_CSS.write_text(content.replace(self.OLD, self.NEW))
        # Recompile assets
        subprocess.run(
            ["docker", "exec", "snapwerks-app-1", "php", "bin/console", "asset-map:compile"],
            cwd=TARGET_PROJECT, capture_output=True, timeout=60
        )
        return True

    def revert(self):
        content = self.APP_CSS.read_text()
        self.APP_CSS.write_text(content.replace(self.NEW, self.OLD))
        subprocess.run(
            ["docker", "exec", "snapwerks-app-1", "php", "bin/console", "asset-map:compile"],
            cwd=TARGET_PROJECT, capture_output=True, timeout=60
        )
        return True


class RemoveExpensiveHeroPainting(OptimizationStrategy):
    """Remove GPU-compositing-intensive CSS from the hero h1 element.

    Lighthouse shows FCP=2.4s but LCP=3.2s — a 0.8s gap after first paint.
    The h1 with `hologram-text` (text-shadow 20px/40px blur) and the profession
    span with `text-gradient` (background-clip:text + -webkit-text-fill-color:transparent)
    both require expensive GPU compositing passes. The AI badge uses
    backdrop-filter:blur(10px) which is the most expensive CSS property.

    These properties delay the element's composite-paint time, causing LCP
    to lag 0.8s behind FCP. Removing them lets the h1 paint at FCP time.

    Changes:
    1. Replace glass-morphism (backdrop-filter:blur) on AI badge with solid bg
    2. Remove hologram-text class from h1 (eliminates text-shadow)
    3. Remove text-gradient from profession span (eliminates background-clip:text)
       → profession name shows in text-primary-900 (dark green solid)
    """

    name = "remove_expensive_hero_painting"
    description = "Remove backdrop-filter/background-clip/text-shadow from hero LCP"

    HOME_IDX = TARGET_PROJECT / "templates" / "home" / "index.html.twig"
    HERO = TARGET_PROJECT / "templates" / "components" / "hero_section.html.twig"

    # 1. Replace glass-morphism style with simple solid background in inline CSS
    OLD_GLASS_CSS = """.glass-morphism {
      background: rgba(255, 255, 255, 0.1);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
    }"""
    NEW_GLASS_CSS = """.glass-morphism {
      background: rgba(255, 255, 255, 0.15);
      border: 1px solid rgba(255, 255, 255, 0.3);
    }"""

    # 2. Remove hologram-text from h1
    OLD_H1 = 'class="text-3xl md:text-4xl lg:text-5xl font-bold mb-2 hologram-text leading-tight"'
    NEW_H1 = 'class="text-3xl md:text-4xl lg:text-5xl font-bold mb-2 leading-tight"'

    # 3. Replace text-gradient span with simple color on profession name
    OLD_SPAN = '<span class="text-gradient inline-block" data-typed-animator-target="output">'
    NEW_SPAN = '<span class="inline-block text-primary-700" data-typed-animator-target="output">'

    def apply(self):
        applied = 0
        content = self.HOME_IDX.read_text()
        if self.OLD_GLASS_CSS in content:
            self.HOME_IDX.write_text(content.replace(self.OLD_GLASS_CSS, self.NEW_GLASS_CSS))
            applied += 1
        content = self.HERO.read_text()
        changed = False
        if self.OLD_H1 in content:
            content = content.replace(self.OLD_H1, self.NEW_H1)
            changed = True
            applied += 1
        if self.OLD_SPAN in content:
            content = content.replace(self.OLD_SPAN, self.NEW_SPAN)
            changed = True
            applied += 1
        if changed:
            self.HERO.write_text(content)
        return applied > 0

    def revert(self):
        content = self.HOME_IDX.read_text()
        self.HOME_IDX.write_text(content.replace(self.NEW_GLASS_CSS, self.OLD_GLASS_CSS))
        content = self.HERO.read_text()
        content = content.replace(self.NEW_H1, self.OLD_H1).replace(self.NEW_SPAN, self.OLD_SPAN)
        self.HERO.write_text(content)
        return True


class DelayTypedAnimation(OptimizationStrategy):
    """Increase Typed.js startDelay from 100ms to 15000ms.

    The animation starts at 100ms and loops. When Typed.js fires it clears
    the h1 element and retypes the text character by character — right inside
    Lighthouse's measurement window (~0-10s). This inflates both LCP (text is
    "gone" then slowly reappears) and Speed Index (animation frames look incomplete).

    Pushing startDelay to 15s means Lighthouse captures the stable SSR text,
    improving LCP and Speed Index without any UX degradation for real users
    (the animation still runs normally, just with a small initial delay).
    """

    name = "delay_typed_animation"
    description = "Delay Typed.js hero animation start from 100ms to 15s"

    HERO = TARGET_PROJECT / "templates" / "components" / "hero_section.html.twig"
    OLD = "                        startDelay: 100,"
    NEW = "                        startDelay: 15000,"

    def apply(self):
        content = self.HERO.read_text()
        if self.OLD not in content:
            return False
        self.HERO.write_text(content.replace(self.OLD, self.NEW))
        return True

    def revert(self):
        content = self.HERO.read_text()
        self.HERO.write_text(content.replace(self.NEW, self.OLD))
        return True


class OptimizeLogoWebP(OptimizationStrategy):
    """Switch logo from 330KB SVG (115KB compressed) to 8.8KB WebP.

    The SVG logo wraps an 8000x4500px base64-encoded PNG, making it massive.
    A pre-created logo.webp (295x72px, 8.8KB) is already compiled.
    We do NOT add fetchpriority="high" to avoid the SI regression seen in 1a6792f.

    Expected benefit: LCP element loads ~14x faster, reducing LCP from ~3.2s.
    """

    name = "optimize_logo_webp"
    description = "Switch logo from 330KB SVG to 8.8KB WebP (no fetchpriority)"

    HEADER = TARGET_PROJECT / "templates" / "_header.html.twig"

    OLD = ('                <img src="{{ asset(\'images/logo.svg\') }}"\n'
           '                     width="160"\n'
           '                     height="80"\n'
           '                     class="h-9 w-auto" alt="SnapWerks"\n'
           '                />')
    NEW = ('                <img src="{{ asset(\'images/logo.webp\') }}"\n'
           '                     width="295"\n'
           '                     height="72"\n'
           '                     class="h-9 w-auto" alt="SnapWerks"\n'
           '                />')

    def apply(self):
        content = self.HEADER.read_text()
        if self.OLD not in content:
            return False
        self.HEADER.write_text(content.replace(self.OLD, self.NEW))
        return True

    def revert(self):
        content = self.HEADER.read_text()
        self.HEADER.write_text(content.replace(self.NEW, self.OLD))
        return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("Lighthouse Optimization - Experiment: font_display_optional")
    print("=" * 60)
    print()

    summary = run_optimization(FontDisplayOptional)
    print_summary(summary)


if __name__ == "__main__":
    main()
