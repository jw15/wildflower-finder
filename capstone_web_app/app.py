''' Code adapted from http://flask.pocoo.org/docs/0.12/patterns/fileuploads/ and https://github.com/pstevens33/Capstone_Moneyball2.0/blob/master/flask_app/app.py'''

from flask import Flask, render_template, request, send_from_directory, make_response
from functools import wraps, update_wrapper
from datetime import datetime
import os, sys
from os import listdir
from os.path import isfile, join
from werkzeug import secure_filename
import pickle, theano, pandas as pd, numpy as np
from keras import optimizers
from keras.models import load_model, model_from_json
import os
# print os.environ['DISPLAY']


sys.path.insert(0, '../src')
from img_preprocess_web import process_image

sys.setrecursionlimit(1000000)

# os.environ["THEANO_FLAGS"] = "cxx=''"


# os.environ["THEANO_FLAGS"] = "device=cuda0"

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
# def load_model():
#     model =  model_from_json(open('../model_outputs/ResNet50_1497216607_2807329/model.json').read())
#     model.load_weights('../model_outputs/ResNet50_1497216607_2807329/ResNet50_1497216607_2807329.h5')
#     model.compile(optimizer='sgd', loss='categorical_crossentropy')
#     return model

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
            prepared_image = process_image('../{}'.format(file_path))
            # add something here to check if file is an image array/ check if RGB
            prediction = model.predict(prepared_image)
            top_prediction = np.argmax(prediction)
            top_proba = prediction[0][top_prediction]
            top_proba = round((top_proba*100), 2)
            top_proba_str = '{}%'.format(top_proba)
            top_species = cats[top_prediction]
            # if top_species == 'delphinium nuttalianum':
            #     top_species = 'mertensia lanceolata'
            order = np.argsort(prediction)[0]
            second_prediction = order[-2]
            second_proba = round((prediction[0][second_prediction]*100), 2)
            second_proba_str = '{}%'.format(second_proba)
            # second_proba = '{}\%'.format(round(prediction[0][second_prediction], 0)*100)
            second_species = cats[second_prediction]

            third_prediction = order[-3]
            # third_proba = '{}\%'.format(round(prediction[0][third_prediction], 0)*100)
            third_proba = round((prediction[0][third_prediction]*100), 2)
            third_proba_str = '{}%'.format(third_proba)
            third_species = cats[third_prediction]
            # top_three = order[-3:]
            # top_three = top_three.tolist()
            # top_five = order[-5:]
            # top_five = top_five.tolist()
            # top_five_list = []
            # for result in top_five:
            #     result = int(result)
            #     species = cats[result]
            #     top_five_list.append(species)
            # top_five_list = top_five_list[::-1]
            # print(top_species)
            # print(top_five_list)
    return render_template('score.html', data=[top_species, top_proba_str, second_species, second_proba_str, third_species, third_proba_str])



if __name__ == '__main__':
    print('Loading model...')
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model = load_model('../model_outputs/ResNet50_1497216607_2807329/ResNet50_1497216607_2807329.h5')

    #  model_from_json(open('../model_outputs/ResNet50_1497216607_2807329/model.json').read())
    # model.load_weights('../model_outputs/ResNet50_1497216607_2807329/ResNet50_1497216607_2807329.h5')
    # model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # model = load_model()
    # loaded_model = load_model('../model_outputs/ResNet50_1497216607_2807329/ResNet50_1497216607_2807329.h5')
    # # load json and create model
    # json_file = open('../model_outputs/model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    # loaded_model.load_weights("../model_outputs/ResNet50_1497216607_2807329.h5")
    print("Loaded model from disk")
    flower_cats = np.load('classes.npy')
    # flower_cats = np.loadtxt('../model_outputs/ResNet50_1497216607_2807329/flower_count_df.pkl', delimiter=',')
    # print(type(flower_cats))
    cats = flower_cats.tolist()
    print('Running app')
    app.run(host='0.0.0.0', port=8105, threaded=True, debug=True)
    # flowermap = download_file('flowermap.html')
