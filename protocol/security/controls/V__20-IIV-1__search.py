from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import re
@app.route('/go')
def go():
    target = request.args.get('target', '')
    if re.search('(www|beta).example.com', target):
        return redirect(target)
    return redirect('/')
