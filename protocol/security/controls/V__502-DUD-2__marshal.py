from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import marshal, base64
@app.route('/api/temperature_for_location')
def temperature_for_location():
    payload = request.args.get('payload')
    data = marshal.loads(base64.b64decode(payload))
    return str(data['latitude'])
