from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from pathlib import Path
@app.route('/read')
def read():
    name = request.args.get('file')
    return (Path('/safe') / name).read_text()
