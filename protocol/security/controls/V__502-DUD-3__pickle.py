from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import pickle, base64
@app.route('/api/calculate_price')
def calculate_price():
    payload = request.args.get('payload')
    data = pickle.loads(base64.b64decode(payload))
    return str(data['quantity'] * 2.0)
