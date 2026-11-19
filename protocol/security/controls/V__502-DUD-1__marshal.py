from flask import Flask, request
app = Flask(__name__)
import marshal
@app.route('/api', methods=['POST'])
def api():
    data = marshal.loads(request.get_data())
    return data['name']
