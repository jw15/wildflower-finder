''' Code adapted from http://flask.pocoo.org/docs/0.12/patterns/fileuploads/ and https://github.com/pstevens33/Capstone_Moneyball2.0/blob/master/flask_app/app.py'''

from flask import Flask, render_template, request, send_from_directory, make_response
from functools import wraps, update_wrapper
from datetime import datetime
import os
from os import listdir
from os.path import isfile, join
from werkzeug import secure_filename

import numpy as np
from keras.models import load_model

import sys
sys.path.insert(0, '../capstone/src')
from image_processing_web import process_image

app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['MAX_CONTENT_PATH'] = 4000000

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3


# home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/maps/<path:filename>')
def download_file(filename):
    return send_from_directory('maps', filename)
    # '/static/maps/', 'flower_map.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/score', methods=['GET', 'POST'])
def score():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, brower also submit empty part w/o filename
        if file.filename == "":
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename )
            file.save(file_path)
            projected = <my_function>(file_path)
            if projected == True:


    return render_template('score.html', data=[score])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, threaded=True, debug=True)
    flowermap = download_file('flowermap.html')
