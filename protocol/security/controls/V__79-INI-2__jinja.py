from flask import Flask, request, redirect, make_response
app = Flask(__name__)
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'), autoescape=False)
def render_response_from_env(env):
    name = request.args.get('name', '')
    template = env.get_template('template.html')
    return make_response(template.render(name=name))
@app.route('/')
def home():
    return render_response_from_env(env)
