from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import yaml
@app.route('/api/temperature_for_location')
def temperature_for_location():
    payload = request.args.get('payload')
    data = yaml.unsafe_load(payload)
    return str(data['latitude'])
