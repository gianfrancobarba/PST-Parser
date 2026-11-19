from flask import Flask, request, redirect, make_response
app = Flask(__name__)
@app.route('/hello', methods=['POST'])
def http_hello_page():
    username = request.form['username']
    return f'<html><body>Hello {username}</body></html>'
