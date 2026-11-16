from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import yaml
@app.route('/api/calculate_price')
def calculate_price():
    data = yaml.safe_load(request.args.get('payload'))
    return str(float(data['quantity']) * 2.0)
