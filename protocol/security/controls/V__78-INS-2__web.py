from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import os
@app.route('/ping')
def ping():
    url = request.args.get('url')
    rc = os.system('ping -c 1 ' + url)
    return str(rc)
