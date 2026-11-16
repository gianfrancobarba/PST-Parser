from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import sqlite3
@app.route('/unsubscribe')
def unsubscribe():
    email = request.args.get('email')
    db = sqlite3.connect('users.db')
    db.execute('DELETE FROM users WHERE email = ?', (email,))
    db.commit()
    return redirect('/')
