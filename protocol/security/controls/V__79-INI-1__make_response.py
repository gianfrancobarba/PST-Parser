from flask import Flask, request, make_response
app = Flask(__name__)
@app.route('/hello', methods=['POST'])
def http_hello_page():
    username = request.get_data(as_text=True)
    return make_response('<p>Hello %s</p>' % username)
