from flask import Flask, request, redirect
app = Flask(__name__)
@app.route('/redirect')
def redirect_url():
    url = request.args.get('url', '')
    if url.startswith('https://example.com'):
        return redirect(url)
    return 'not found', 404
