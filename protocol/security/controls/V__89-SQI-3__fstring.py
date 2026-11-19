from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import sqlite3
@app.route('/post', methods=['POST'])
def post():
    username, message = request.form['username'], request.form['message']
    db = sqlite3.connect('db.sqlite')
    db.execute(f"INSERT INTO messages VALUES ('{username}', '{message}')")
    db.commit()
    return redirect('/')
