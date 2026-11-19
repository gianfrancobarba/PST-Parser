from flask import Flask, request, redirect, make_response
app = Flask(__name__)
@app.route('/hello/<username>')
def hello(username):
    return '<h1>Hello {}</h1>'.format(username)
