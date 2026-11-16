from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from markupsafe import escape
@app.route('/hello', methods=['POST'])
def http_hello_page():
    username = request.get_data(as_text=True)
    return '<html><body>Hello ' + str(escape(username)) + '</body></html>'
