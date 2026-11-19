from flask import Flask, request
app = Flask(__name__)
import marshal, base64
@app.route('/api/calculate_price')
def calculate_price():
    payload = request.args.get('payload')
    data = marshal.loads(base64.b64decode(payload))
    return str(data['quantity'] * 2.0)
