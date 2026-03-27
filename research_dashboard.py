"""
Research Dashboard - Web interface for the autonomous research community.
A simple Flask app to visualize research progress, leaderboards, and findings.
"""

import json
from flask import Flask, render_template_string, jsonify, request
from pathlib import Path

from research_hub import ResearchHub

app = Flask(__name__)

# HTML template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autoresearch Community Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #30363d;
            margin-bottom: 30px;
        }
        h1 { color: #58a6ff; font-size: 1.8rem; }
        .stats { display: flex; gap: 30px; }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #58a6ff;
        }
        .stat-label {
            font-size: 0.85rem;
            color: #8b949e;
            text-transform: uppercase;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }
        
        .card h2 {
            color: #8b949e;
            font-size: 0.9rem;
            text-transform: uppercase;
            margin-bottom: 15px;
            letter-spacing: 0.5px;
        }
        
        .leaderboard { grid-column: span 2; }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #30363d;
        }
        th {
            color: #8b949e;
            font-weight: normal;
            font-size: 0.85rem;
            text-transform: uppercase;
        }
        tr:hover { background: #21262d; }
        
        .val-bpb { font-family: monospace; color: #7ee787; }
        .improvement { color: #7ee787; }
        .improvement.negative { color: #f85149; }
        
        .tag {
            display: inline-block;
            padding: 2px 8px;
            background: #21262d;
            border-radius: 12px;
            font-size: 0.75rem;
            margin-right: 4px;
            color: #58a6ff;
        }
        
        .branch-name {
            font-family: monospace;
            color: #58a6ff;
        }
        
        .status {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
        }
        .status.active { background: #238636; color: white; }
        .status.completed { background: #1f6feb; color: white; }
        .status.exhausted { background: #6e7681; color: white; }
        
        .thread-list { max-height: 400px; overflow-y: auto; }
        .thread-item {
            padding: 15px;
            border-bottom: 1px solid #30363d;
        }
        .thread-item:last-child { border-bottom: none; }
        .thread-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .thread-hypothesis {
            color: #c9d1d9;
            margin-bottom: 8px;
        }
        .thread-findings {
            color: #8b949e;
            font-size: 0.85rem;
        }
        
        .tags-list { margin-top: 10px; }
        
        .refresh-btn {
            background: #238636;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .refresh-btn:hover { background: #2ea043; }
        
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
            .leaderboard { grid-column: span 1; }
            .stats { flex-wrap: wrap; gap: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🔬 Autoresearch Community</h1>
                <p style="color: #8b949e; margin-top: 5px;">Multi-agent collaborative research</p>
            </div>
            <button class="refresh-btn" onclick="location.reload()">Refresh</button>
        </header>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{{ total_threads }}</div>
                <div class="stat-label">Total Threads</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ active_threads_count }}</div>
                <div class="stat-label">Active</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ best_val_bpb }}</div>
                <div class="stat-label">Best val_bpb</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ total_tags }}</div>
                <div class="stat-label">Research Areas</div>
            </div>
        </div>
        
        <br>
        
        <div class="grid">
            <div class="card leaderboard">
                <h2>🏆 Leaderboard</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Branch</th>
                            <th>val_bpb</th>
                            <th>Improvement</th>
                            <th>Memory</th>
                            <th>Params</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for thread in leaderboard %}
                        <tr>
                            <td><span class="branch-name">{{ thread.branch }}</span></td>
                            <td class="val-bpb">{{ "%.4f"|format(thread.val_bpb) if thread.val_bpb else "—" }}</td>
                            <td class="improvement {% if thread.improvement and thread.improvement < 0 %}negative{% endif %}">
                                {% if thread.improvement %}{{ "+%.4f"|format(thread.improvement) }}{% else %}—{% endif %}
                            </td>
                            <td>{% if thread.peak_memory_gb %}{{ "%.1f"|format(thread.peak_memory_gb) }}GB{% else %}—{% endif %}</td>
                            <td>{% if thread.num_params_m %}{{ "%.1f"|format(thread.num_params_m) }}M{% else %}—{% endif %}</td>
                            <td><span class="status {{ thread.status }}">{{ thread.status }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <h2>🔥 Active Research</h2>
                <div class="thread-list">
                    {% for thread in active_threads %}
                    <div class="thread-item">
                        <div class="thread-header">
                            <span class="branch-name">{{ thread.branch }}</span>
                            <span class="status active">active</span>
                        </div>
                        <div class="thread-hypothesis">{{ thread.hypothesis }}</div>
                        {% if thread.key_findings %}
                        <div class="thread-findings">
                            <strong>Findings:</strong> {{ thread.key_findings[0] }}
                        </div>
                        {% endif %}
                        <div class="tags-list">
                            {% for tag in thread.tags %}
                            <span class="tag">{{ tag }}</span>
                            {% endfor %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="card">
                <h2>📚 Research Areas</h2>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    {% for tag in all_tags %}
                    <span class="tag" style="padding: 6px 12px;">{{ tag }}</span>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def index():
    """Render the main dashboard."""
    hub = ResearchHub()
    dashboard = hub.get_dashboard()
    all_threads = hub.get_all_threads()
    
    # Get all unique tags
    all_tags = list(set(
        tag for thread in all_threads for tag in thread.tags
    ))
    
    # Calculate best val_bpb
    best = dashboard.get('best_result', {})
    best_val_bpb = f"{best.get('val_bpb', '—')}" if best.get('val_bpb') else '—'
    
    return render_template_string(
        DASHBOARD_TEMPLATE,
        total_threads=dashboard['total_threads'],
        active_threads_count=dashboard['active_threads'],
        best_val_bpb=best_val_bpb,
        total_tags=len(all_tags),
        leaderboard=sorted(
            [t for t in all_threads if t.val_bpb],
            key=lambda x: x.val_bpb
        )[:20],
        active_threads=[t for t in all_threads if t.status == 'active'][:10],
        all_tags=sorted(all_tags)
    )


@app.route('/api/threads')
def api_threads():
    """API endpoint to get all threads."""
    hub = ResearchHub()
    threads = hub.get_all_threads()
    return jsonify([{
        'branch': t.branch,
        'hypothesis': t.hypothesis,
        'description': t.description,
        'tags': t.tags,
        'status': t.status,
        'val_bpb': t.val_bpb,
        'improvement': t.improvement,
        'key_findings': t.key_findings,
        'future_directions': t.future_directions,
        'created_at': t.created_at,
        'updated_at': t.updated_at
    } for t in threads])


@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint to get dashboard data."""
    hub = ResearchHub()
    return jsonify(hub.get_dashboard())


def run_dashboard(host='0.0.0.0', port=5000):
    """Run the dashboard server."""
    print(f"Starting research dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the research dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    args = parser.parse_args()
    
    run_dashboard(args.host, args.port)
