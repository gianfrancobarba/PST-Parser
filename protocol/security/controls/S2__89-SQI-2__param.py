from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import sqlite3
@app.route('/buy_order', methods=['POST'])
def buy_stock():
    db = sqlite3.connect('db.sqlite')
    db.execute('INSERT INTO buy_orders VALUES (?, ?)', (request.form['stock_name'], request.form['stock_quantity']))
    db.commit()
    return redirect('/')
