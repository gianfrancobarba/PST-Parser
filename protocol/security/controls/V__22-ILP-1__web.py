from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import os
@app.route('/read')
def read():
    name = request.args.get('file')
    with open(os.path.join('/safe/', name)) as f:
        return f.read()
