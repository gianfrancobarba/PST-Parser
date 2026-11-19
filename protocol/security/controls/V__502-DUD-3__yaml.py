from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import yaml
@app.route('/api/calculate_price')
def calculate_price():
    payload = request.args.get('payload')
    data = yaml.unsafe_load(payload)
    return str(data['quantity'] * 2.0)
