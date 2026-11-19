from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from flask import send_file
@app.route('/img')
def img():
    name = request.args.get('file')
    return send_file('images/' + name)
