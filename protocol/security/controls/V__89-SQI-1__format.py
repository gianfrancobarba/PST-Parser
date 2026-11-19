from flask import Flask, request
app = Flask(__name__)
import sqlite3
@app.route('/unsubscribe', methods=['POST'])
def unsubscribe():
    email = request.form['email']
    db = sqlite3.connect('users.db')
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE email = '{}'".format(email))
    if cur.fetchone():
        cur.execute("DELETE FROM users WHERE email = '{}'".format(email))
        db.commit()
        return 'unsubscribed'
    return 'not subscribed'
