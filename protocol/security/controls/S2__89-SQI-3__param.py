from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import sqlite3
@app.route('/post', methods=['POST'])
def post():
    db = sqlite3.connect('db.sqlite')
    db.execute('INSERT INTO messages VALUES (?, ?)', (request.form['username'], request.form['message']))
    db.commit()
    return redirect('/')
