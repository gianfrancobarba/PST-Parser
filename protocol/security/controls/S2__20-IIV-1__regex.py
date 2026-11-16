from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import re
TARGET = re.compile(r'^https?://(www|beta)\.example\.com/')
@app.route('/go')
def go():
    target = request.args.get('target', '')
    if TARGET.match(target):
        return redirect('/ok')
    return redirect('/')
