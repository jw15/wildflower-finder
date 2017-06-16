''' Code adapted from http://flask.pocoo.org/docs/0.12/patterns/fileuploads/ and https://github.com/pstevens33/Capstone_Moneyball2.0/blob/master/flask_app/app.py'''

from flask import Flask, render_template, request, send_from_directory, make_response
from functools import wraps, update_wrapper
from datetime import datetime
import os, sys, re
from os import listdir
from os.path import isfile, join
from werkzeug import secure_filename
import pickle, theano, pandas as pd, numpy as np
import matplotlib as pyplot
from keras import optimizers
from keras.models import load_model, model_from_json
import matplotlib.pyplot as plt
import os

# print os.environ['DISPLAY']


sys.path.insert(0, '../src')
from img_preprocess_web import process_image
from my_utils import image_categories_reverse, beautify_name, make_db, crop_thumbnail

sys.setrecursionlimit(1000000)

# os.environ["THEANO_FLAGS"] = "cxx=''"
# os.environ["THEANO_FLAGS"] = "device=cuda0"

app = Flask(__name__, static_url_path= '/static')

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['MAX_CONTENT_PATH'] = 4000000

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 3

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
    flower_dict = image_categories_reverse()
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
            idx1 = cats[top_prediction]
            species1 = beautify_name(idx1)
            common1 = flower_df[flower_df[0]==idx1]['common_names']
            common1 = common1.values[0]

            order = np.argsort(prediction)[0]

            second_prediction = order[-2]
            second_proba = round((prediction[0][second_prediction]*100), 2)
            second_proba_str = '{}%'.format(second_proba)
            # second_proba = '{}\%'.format(round(prediction[0][second_prediction], 0)*100)
            idx2 = cats[second_prediction]
            species2 = beautify_name(idx2)
            common2 = flower_df[flower_df[0]==idx2]['common_names']
            common2 = common2.values[0]

            third_prediction = order[-3]
            # third_proba = '{}\%'.format(round(prediction[0][third_prediction], 0)*100)
            third_proba = round((prediction[0][third_prediction]*100), 2)
            third_proba_str = '{}%'.format(third_proba)
            idx3 = cats[third_prediction]
            species3 = beautify_name(idx3)
            common3 = flower_df[flower_df[0]==idx3]['common_names']
            common3 = common3.values[0]

            img1 = str(flower_dict[idx1][0])
            # img1 = crop_thumbnail(img1, (3024, 3024))
            # img1 = plt.imshow(img1.astype(float))

            img2 = str(flower_dict[idx2][0])
            # img2 = crop_thumbnail(img2, (3024, 3024))
            # img2 = plt.imshow(img2.astype(float))

            img3 = str(flower_dict[idx3][0])
            # img3 = crop_thumbnail(img3, (3024, 3024))
            # img3 = plt.imshow(img3.astype(float))

            family1 = flower_df[flower_df[0]==idx1]['family']
            family1 = family1.values[0]

            family2 = flower_df[flower_df[0]==idx2]['family']
            family2 = family2.values[0]

            family3 = flower_df[flower_df[0]==idx3]['family']
            family3 = family3.values[0]

    return render_template('score.html',
    img1 = img1, species1=species1, common1=common1, top_proba_str=top_proba_str, family1=family1, img2=img2, species2=species2, common2=common2, second_proba_str=second_proba_str, family2=family2, img3=img3, species3=species3, common3=common3, third_proba_str=third_proba_str, family3=family3)



if __name__ == '__main__':
    print('Loading model...')
    # sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model = load_model('../model_outputs/ResNet50_1497216607_2807329/ResNet50_1497216607_2807329.h5')
    flower_df = make_db()

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
    app.run(host='0.0.0.0', port=8105, threaded=True, debug=False)
    # flowermap = download_file('flowermap.html')
