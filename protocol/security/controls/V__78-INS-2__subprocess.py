from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import subprocess
@app.route('/ping')
def ping():
    url = request.args.get('url')
    done = subprocess.run(f'ping -c 1 {url}', shell=True)
    return str(done.returncode)
