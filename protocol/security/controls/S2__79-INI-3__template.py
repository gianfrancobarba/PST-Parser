from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from flask import render_template
@app.route('/hello/<username>')
def hello(username):
    return render_template('hello.html', username=username)
