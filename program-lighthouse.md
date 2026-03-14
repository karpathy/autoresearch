# autoresearch — Lighthouse Optimization

This is an experiment to have the LLM autonomously optimize a web application for 100% Lighthouse scores.

## Setup

To set up a new Lighthouse optimization run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar14-lh`). The branch `lighthouse/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b lighthouse/<tag>` from current master.
3. **Read the in-scope files**: The repo structure for Lighthouse optimization:
   - `README.md` — repository context.
   - `program-lighthouse.md` — these instructions.
   - `lighthouse_audit.py` — audit runner and metrics collection. Do not modify the core evaluation logic.
   - `optimize.py` — the file you modify. This contains all optimization strategies.
   - Target project: `/home/<Project/Directory>` — the Symfony web application to optimize.
4. **Verify target project is accessible**: Check that `/home/<Project/Directory>` exists and is running.
5. **Initialize results-lighthouse.tsv**: Create `results-lighthouse.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Lighthouse Metrics

Lighthouse measures 5 core performance metrics:

1. **FCP (First Contentful Paint)**: Time until first content is painted. Target: <1.8s
2. **SI (Speed Index)**: How quickly content is visually displayed. Target: <3.4s
3. **LCP (Largest Contentful Paint)**: Time until largest content is painted. Target: <2.5s
4. **TTI (Time to Interactive)**: Time until page is fully interactive. Target: <3.8s
5. **TBT (Total Blocking Time)**: Sum of blocking time. Target: <200ms
6. **CLS (Cumulative Layout Shift)**: Sum of layout shifts. Target: <0.1

The overall **Performance Score** is a weighted combination of these metrics.

### Other Categories

- **Accessibility**: WCAG compliance, ARIA labels, color contrast, etc.
- **Best Practices**: Security, modern web standards, no deprecated APIs.
- **SEO**: Meta tags, structured data, mobile-friendly, etc.
- **PWA**: Progressive web app features (optional for this project).

## Experimentation

Each experiment runs a Lighthouse audit on the target application.

**What you CAN do:**
- Modify `optimize.py` — this is the only file you edit. All optimization strategies go here.
- Modify web assets in the target project: CSS, JavaScript, Twig templates, images.
- Add/remove files in the target project's `public/`, `assets/`, `templates/` directories.
- Install npm packages or modify build configurations in the target project.
- Modify nginx/Apache configs, caching headers, compression settings.
- Change database queries, add indexes, optimize backend code.

**What you CANNOT do:**
- Modify `lighthouse_audit.py` core evaluation logic. The audit runner is fixed.
- Change the Lighthouse scoring criteria or thresholds.
- Install Python packages beyond what's in `pyproject.toml`.

**The goal is simple: get 100% in all Lighthouse categories.** Focus on Performance first, then Accessibility, Best Practices, and SEO.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great win.

## The first run

Your very first run should always be to establish the baseline, so you will run the audit script as-is on the current PROJECT_NAME codebase.

## Output format

Once the audit script finishes, it prints a summary like this:

```
---
performance:        87
accessibility:      92
best_practices:     100
seo:                95
pwa:                45
total_score:        91.8
audit_time_seconds: 45.2
url:                http://localhost:8000/
```

Note that the script runs audits on multiple pages and averages the scores. You can extract the key metrics:

```
grep "^performance:\|^accessibility:\|^best_practices:\|^seo:" run-lighthouse.log
```

## Logging results

When an experiment is done, log it to `results-lighthouse.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	performance	accessibility	best_practices	seo	status	description
```

1. git commit hash (short, 7 chars)
2. performance score (0-100) — use 0 for crashes
3. accessibility score (0-100)
4. best_practices score (0-100)
5. seo score (0-100)
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	performance	accessibility	best_practices	seo	status	description
a1b2c3d	87.0	92.0	100.0	95.0	keep	baseline
b2c3d4e	91.0	92.0	100.0	95.0	keep	enable gzip compression
c3d4e5f	85.0	90.0	100.0	95.0	discard	added heavy JS library (slower FCP)
d4e5f6g	0.0	0.0	0.0	0.0	crash	broken JS syntax
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `lighthouse/mar14`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `optimize.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the audit: `uv run optimize.py > run-lighthouse.log 2>&1`
5. Read out the results: `grep "^performance:\|^accessibility:" run-lighthouse.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run-lighthouse.log` to read the error.
7. Record the results in the TSV (NOTE: do not commit the results-lighthouse.tsv file, leave it untracked by git)
8. If the total score improved (higher), you "advance" the branch, keeping the git commit
9. If the score is equal or worse, you git reset back to where you started

**Timeout**: Each audit should take ~1-2 minutes. If a run exceeds 5 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (broken JS, syntax error, etc.), use your judgment: If it's easy to fix, fix it and re-run. If the idea is fundamentally broken, skip it and log "crash" as the status.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. The loop runs until the human interrupts you.

## Optimization Strategies

Here are proven strategies to try:

### Performance
- Minimize CSS/JS bundle sizes (tree-shaking, code splitting)
- Enable compression (gzip, brotli)
- Optimize images (WebP, lazy loading, proper sizing)
- Reduce critical rendering path (inline critical CSS, defer non-critical JS)
- Use HTTP/2 push or preloading for critical assets
- Implement service worker caching
- Optimize database queries and add proper indexes
- Enable server-side caching (OPcache, Redis)
- Use CDN for static assets
- Minimize DOM size and complexity
- Reduce third-party scripts

### Accessibility
- Add proper ARIA labels and roles
- Ensure color contrast meets WCAG AA/AAA
- Add alt text to all images
- Ensure keyboard navigation works
- Add skip links and proper heading hierarchy
- Fix form labels and error messages
- Ensure focus states are visible

### Best Practices
- Use HTTPS (in production)
- Remove console.log and debugger statements
- Use modern JavaScript (ES6+)
- Avoid deprecated APIs
- Add security headers (CSP, X-Frame-Options, etc.)
- Use passive event listeners
- Avoid document.write()

### SEO
- Add proper meta tags (title, description, OG tags)
- Use semantic HTML
- Add structured data (JSON-LD)
- Ensure mobile-friendly design
- Add canonical URLs
- Optimize for Core Web Vitals
- Create XML sitemap
- Add robots.txt

## Target URLs

By default, audit these key pages:
- Homepage: `http://localhost:8000/`
- Contact/About page

