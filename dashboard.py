import os
import json
import http.server
import socketserver

PORT = 8080

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
    <title>Autoresearch Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #1e1e1e; color: #fff; font-family: sans-serif; margin: 0; padding: 20px; }
        .container { display: flex; flex-direction: column; gap: 20px; max-width: 1200px; margin: 0 auto; }
        .panels { display: flex; gap: 20px; }
        .panel { background: #2d2d2d; padding: 20px; border-radius: 8px; flex: 1; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h2 { margin-top: 0; }
        pre { background: #000; padding: 10px; border-radius: 4px; overflow-y: auto; max-height: 400px; color: #0f0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #444; }
        th { background: #3d3d3d; }
        .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
        .KEEP { background: #28a745; }
        .DISCARD { background: #dc3545; }
        .CRASH { background: #ffc107; color: #000; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Autoresearch Real-Time Dashboard</h1>
        <div class="panels">
            <div class="panel" style="flex: 2">
                <h2>Validation BPB (Bits Per Byte) History</h2>
                <canvas id="bpbChart"></canvas>
            </div>
            <div class="panel" style="flex: 1">
                <h2>Experiments Summary</h2>
                <div id="stats">Loading...</div>
            </div>
        </div>
        <div class="panels">
            <div class="panel" style="flex: 2">
                <h2>Recent Experiments</h2>
                <table>
                    <thead><tr><th>Commit</th><th>BPB</th><th>Memory (MB)</th><th>Status</th><th>Description</th></tr></thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
            <div class="panel" style="flex: 1">
                <h2>Live Log (Tail)</h2>
                <pre id="logOutput">Waiting for logs...</pre>
            </div>
        </div>
    </div>
    <script>
        let bpbChart;
        async function fetchData() {
            try {
                const res = await fetch('/data');
                const data = await res.json();
                
                const tbody = document.getElementById('tableBody');
                tbody.innerHTML = data.results.slice(-10).reverse().map(r => `
                    <tr>
                        <td><code>${r.commit.substring(0,7)}</code></td>
                        <td>${r.bpb}</td>
                        <td>${r.memory}</td>
                        <td><span class="badge ${r.status}">${r.status}</span></td>
                        <td>${r.description}</td>
                    </tr>
                `).join('');

                const keeps = data.results.filter(r => r.status === 'KEEP').length;
                const discards = data.results.filter(r => r.status === 'DISCARD').length;
                const crashes = data.results.filter(r => r.status === 'CRASH').length;
                document.getElementById('stats').innerHTML = `
                    <p>Total Runs: <b>${data.results.length}</b></p>
                    <p>🟢 Successful Keeps: <b>${keeps}</b></p>
                    <p>🔴 Discarded ideas: <b>${discards}</b></p>
                    <p>⚠️ Crashes/OOMs: <b>${crashes}</b></p>
                `;

                const validRuns = data.results.filter(r => r.bpb && !isNaN(r.bpb));
                const labels = validRuns.map((r, i) => i + 1);
                const speeds = validRuns.map(r => parseFloat(r.bpb));

                if (!bpbChart) {
                    const ctx = document.getElementById('bpbChart').getContext('2d');
                    bpbChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Validation BPB',
                                data: speeds,
                                borderColor: '#007bff',
                                tension: 0.1,
                                fill: false
                            }]
                        },
                        options: { animation: false }
                    });
                } else {
                    bpbChart.data.labels = labels;
                    bpbChart.data.datasets[0].data = speeds;
                    bpbChart.update();
                }

                document.getElementById('logOutput').innerText = data.log;
                
            } catch (err) {
                console.error(err);
            }
        }
        setInterval(fetchData, 2000);
        fetchData();
    </script>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))
        elif self.path == '/data':
            results = []
            try:
                if os.path.exists("results.tsv"):
                    with open("results.tsv", "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        for line in lines[1:]:
                            parts = line.strip().split("\t")
                            if len(parts) >= 5:
                                results.append({
                                    "commit": parts[0],
                                    "bpb": parts[1],
                                    "memory": parts[2],
                                    "status": parts[3].upper(),
                                    "description": parts[4]
                                })
            except Exception:
                pass
            
            log_tail = ""
            try:
                if os.path.exists("run.log"):
                    with open("run.log", "r", encoding="utf-8") as f:
                        log_tail = "".join(f.readlines()[-40:])
            except:
                pass

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"results": results, "log": log_tail}).encode("utf-8"))
        else:
            super().do_GET()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"📊 Dashboard gracefully running at http://localhost:{PORT}")
        httpd.serve_forever()
