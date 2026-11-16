from flask import Flask, request, redirect, make_response
app = Flask(__name__)
@app.route('/redirect')
def redirect_url():
    url = request.args.get('url', '')
    if 'example.com' in url:
        return redirect(url)
    return 'not found', 404
