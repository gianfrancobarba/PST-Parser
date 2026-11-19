from flask import Flask, request, redirect
app = Flask(__name__)
import re
@app.route('/next')
def next_page():
    url = request.args.get('url', '')
    if re.fullmatch(r'https://[a-z]+.example.com/.*', url):
        return redirect(url)
    return redirect('/')
