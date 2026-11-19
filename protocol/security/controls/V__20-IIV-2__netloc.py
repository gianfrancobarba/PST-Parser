from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from urllib.parse import urlparse
@app.route('/redirect/<path:url>')
def redirect_path(url):
    if 'example.com' in urlparse(url).netloc:
        return redirect(url)
    return 'not found', 404
