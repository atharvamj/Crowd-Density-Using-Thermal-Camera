# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template,request
import sqlite3

# conn = sqlite3.connect('Database_2.db')
# cursor = conn.cursor()
# # Flask constructor takes the name of
# # current module (__name__) as argument.
app = Flask(__name__)
def get_db_connection():
    conn = sqlite3.connect('Database_3.db')
    conn.row_factory = sqlite3.Row
    return conn
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
	conn = get_db_connection()
	data = conn.execute("SELECT id , Number FROM People_number").fetchall()
	data=[(data[-1][0],data[-1][1]),(data[-2][0],data[-2][1]),(data[-3][0],data[-3][1]),(data[-4][0],data[-4][1]),(data[-5][0],data[-5][1])]
	# print(data)
	return render_template('index.html',data=data)


# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run(port=5000)
