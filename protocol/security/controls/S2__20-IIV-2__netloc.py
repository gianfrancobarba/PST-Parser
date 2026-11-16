from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from urllib.parse import urlparse
@app.route('/redirect')
def redirect_url():
    url = request.args.get('url', '')
    if urlparse(url).netloc == 'example.com':
        return redirect('/ok')
    return 'not found', 404
