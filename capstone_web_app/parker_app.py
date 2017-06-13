
from flask import Flask, render_template, request
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
import os
from os import listdir
from os.path import isfile, join
from werkzeug import secure_filename

import numpy as np
from keras.models import load_model

import sys
sys.path.insert(0, '../src')
from project_faces_web import project_face
from image_processing_web import process_image



app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['MAX_CONTENT_PATH'] = 4000000

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3


@app.route('/')
def index():
    return render_template('index.html')

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
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            projected = project_face(file_path)
            if projected == True:
                prepared_image = process_image(file_path[8:])
                prediction = model.predict(prepared_image)[0]
                score = 0
                for count, val in enumerate(prediction):
                    if count == 0:
                        score += val * 30
                    elif count == 1:
                        score += val * 70
                    elif count == 2:
                        score += val * 80
                    elif count == 3:
                        score += val * 90
                    elif count == 4:
                        score += val * 100
                    elif count == 5:
                        score += val * 200

                # if score > 100:
                #     score = 100
                score = round(score,0)
                print(prediction)
                print(score)


    # onlyfiles = [f for f in listdir('static/img/players') if isfile(join('static/img/players', f))]
    # paths = []
    # for i in onlyfiles:
    #     paths.append('img/players/' + i)
    # random_index = np.random.randint(X.shape[0])
    # upload_picture = X[random_index]
    # upload_picture = upload_picture.reshape((1,) + upload_picture.shape)
    #
    #


    return render_template('score.html', data=[score])

@app.route('/howitworks')
def howitworks():
    return render_template('howitworks.html')

@app.route('/perfection')
def perfection():
    return render_template('perfection.html')


@app.after_request
def add_header(response):
    response.cache_control.max_age = 300
    return response





if __name__ == '__main__':
    X = np.load('../data/X_players.npy')
    model = load_model('../data/models/gpu_300_players_sigmoid_binary.h5')
    app.run(host='0.0.0.0', port=8080, debug=True)
