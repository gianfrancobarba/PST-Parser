from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import os
@app.route('/ping/<url>')
def ping(url):
    return os.popen('ping -c 1 %s' % url).read()
