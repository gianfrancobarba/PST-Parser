from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from flask import send_from_directory
@app.route('/img')
def img():
    return send_from_directory('images', request.args.get('name'))
