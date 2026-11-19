from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import sqlite3
@app.route('/buy_order', methods=['POST'])
def buy_stock():
    name, quantity = request.form['stock_name'], request.form['stock_quantity']
    db = sqlite3.connect('db.sqlite')
    db.cursor().execute('INSERT INTO buy_orders VALUES (\'' + name + '\', ' + quantity + ')')
    db.commit()
    return redirect('/')
