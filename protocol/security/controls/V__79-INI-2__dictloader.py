from flask import Flask, request, make_response
app = Flask(__name__)
from jinja2 import Environment, DictLoader
env = Environment(loader=DictLoader({'template.html': '<p>Hello {{ name }}</p>'}), autoescape=False)
@app.route('/')
def home():
    name = request.args.get('name', '')
    template = env.get_template('template.html')
    return make_response(template.render(name=name))
