from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template
from flask import send_from_directory

import test
import json
import numpy as np


# @app.route('/result/',)


# from flask import render_template, Flask
# app=Flask(__name__,)
app = Flask(__name__,template_folder='templates', static_folder='static')
model=test.loading_model()
@app.route('/',methods=['GET','POST'])
def index():
	return render_template("index.html")

@app.route('/prediction',methods=['GET','POST'])
def prediction():
	if request.method == 'POST':
			link = request.form['in']
	        # url = request.args.get('d')
			out= test.predcit(link,model)
	    # return jsonify(int(out))
	return render_template('index.html',pred=out)



if __name__ == '__main__':
	app.run(debug=True)

