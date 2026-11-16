from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import yaml
@app.route('/api')
def api():
    payload = request.args.get('payload')
    data = yaml.load(payload, Loader=yaml.Loader)
    return data['name']
