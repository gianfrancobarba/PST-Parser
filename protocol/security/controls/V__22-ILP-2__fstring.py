from flask import Flask
app = Flask(__name__)
@app.route('/img/<path:name>')
def img(name):
    with open(f'images/{name}', 'rb') as handle:
        return handle.read()
