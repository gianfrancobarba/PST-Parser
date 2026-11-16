from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from jinja2 import Environment, FileSystemLoader, select_autoescape
env = Environment(loader=FileSystemLoader('templates'), autoescape=select_autoescape())
def render_response_from_env(env):
    name = request.args.get('name', '')
    return make_response(env.get_template('template.html').render(name=name))
@app.route('/')
def home():
    return render_response_from_env(env)
