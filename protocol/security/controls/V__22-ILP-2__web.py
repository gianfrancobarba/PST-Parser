from flask import Flask, request, redirect, make_response
app = Flask(__name__)
@app.route('/img')
def img():
    name = request.args.get('name')
    return open('images/' + name, 'rb').read()
