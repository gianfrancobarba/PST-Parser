from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import subprocess
@app.route('/ping')
def ping():
    url = request.args.get('url')
    rc = subprocess.run(['ping', '-c', '1', url]).returncode
    return str(rc)
