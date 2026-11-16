from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import yaml
@app.route('/api')
def api():
    data = yaml.safe_load(request.args.get('payload'))
    return str(len(data))
