from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from jinja2 import Environment
env = Environment(autoescape=False)
@app.route('/')
def home():
    name = request.args.get('name', '')
    template = env.from_string('<p>Hello {{ name }}</p>')
    return make_response(template.render(name=name))
