import os
from flask import Flask, request, jsonify, render_template_string
from supabase import create_client

app = Flask(__name__)

# Supabase Credentials
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL") or "https://dxtkctltwnghsfljjjym.supabase.co"
SUPABASE_KEY = os.getenv("VITE_SUPABASE_ANON_KEY") or ""

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Simple HTML UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Telecom Site Audit Portal</title>
    <style>
        body { font-family: sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .card { background: #1e293b; padding: 2rem; border-radius: 8px; width: 320px; border: 1px solid #334155; }
        h2 { color: #38bdf8; margin-top: 0; }
        input, select, button { width: 100%; padding: 0.6rem; margin-top: 0.5rem; margin-bottom: 1rem; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #fff; box-sizing: border-box; }
        button { background: #0284c7; font-weight: bold; cursor: pointer; border: none; }
    </style>
</head>
<body>
    <div class="card">
        <h2>📡 Site Audit Portal</h2>
        <form action="/submit" method="POST">
            <label>Site ID</label>
            <input type="text" name="site_id" placeholder="e.g. BGW_0123" required>
            <label>Technology</label>
            <select name="tech">
                <option>2G</option>
                <option>3G</option>
                <option>4G</option>
                <option>5G</option>
            </select>
            <label>Power System (-48V DC)</label>
            <select name="power">
                <option>Normal</option>
                <option>Battery Warning</option>
                <option>Mains Failure</option>
            </select>
            <button type="submit">Submit Audit</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def home(path):
    return render_template_string(HTML_TEMPLATE)

@app.route('/submit', methods=['POST'])
def submit():
    site_id = request.form.get('site_id')
    tech = request.form.get('tech')
    power = request.form.get('power')
    
    # Example insertion into Supabase
    try:
        supabase.table('audits').insert({'site_id': site_id, 'tech': tech, 'power_status': power}).execute()
        return f"<h3>✓ Audit for {site_id} successfully recorded!</h3><a href='/'>Go back</a>"
    except Exception as e:
        return f"<h3>Submitted: {site_id} ({tech})</h3><p>Note: {e}</p><a href='/'>Go back</a>"
