from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import pickle
@app.route('/api', methods=['POST'])
def api():
    data = pickle.loads(request.get_data())
    return data['name']
