from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import sqlite3
@app.route('/unsubscribe')
def unsubscribe():
    email = request.args.get('email')
    db = sqlite3.connect('users.db')
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE email = '%s'" % email)
    if cur.fetchone():
        cur.execute("DELETE FROM users WHERE email = '" + email + "'")
        db.commit()
    return redirect('/')
