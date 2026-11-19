from flask import Flask
app = Flask(__name__)
@app.route('/hello/<username>')
def hello(username):
    return '<h1>Hello ' + username + '</h1>'
