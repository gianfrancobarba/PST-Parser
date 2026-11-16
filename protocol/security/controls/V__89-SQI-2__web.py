from flask import Flask, request, redirect, make_response
app = Flask(__name__)
import sqlite3
@app.route('/buy_order', methods=['POST'])
def buy_stock():
    stock_name = request.form['stock_name']
    stock_quantity = request.form['stock_quantity']
    db = sqlite3.connect('db.sqlite')
    db.execute(f"INSERT INTO buy_orders (stock_name, stock_quantity) VALUES ('{stock_name}', {stock_quantity})")
    db.commit()
    return redirect('/')
