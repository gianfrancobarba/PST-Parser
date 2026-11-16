from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import sqlite3
def insert_user_message_in_db(username, message):
    db = sqlite3.connect('db.sqlite')
    db.execute("INSERT INTO messages VALUES ('" + username + "', '" + message + "')")
    db.commit()
@app.route('/post', methods=['POST'])
def post():
    insert_user_message_in_db(request.form['username'], request.form['message'])
    return redirect('/')
