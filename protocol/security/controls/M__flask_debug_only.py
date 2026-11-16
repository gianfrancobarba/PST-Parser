from flask import Flask, request, redirect, make_response
app = Flask(__name__)
@app.route('/')
def index():
    return 'ok'

app.run(debug=True)
