from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import json
@app.route('/api/temperature_for_location')
def t():
    data = json.loads(request.args.get('payload'))
    return str(len(data))
