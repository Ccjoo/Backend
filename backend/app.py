from flask import Flask, request, jsonify
from flask_mongoengine import MongoEngine
from flask_cors import CORS
from pymongo import MongoClient

#model
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

import os
from werkzeug.utils import secure_filename

#model load
try:
    model = tf.keras.models.load_model('./backend/model/20210625_model_ft_10_2.h5')
except:
    model = tf.keras.models.load_model('./model/20210625_model_ft_10_2.h5')

#DB
client = MongoClient('mongodb+srv://seyeon:choi6738@cluster0.hrldm.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
db = client.dogimformation
collection = db.dog_db

#Flask 객체 인스턴스 생성
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './dog_image'
CORS(app)

@app.route('/', methods = ['POST']) # 접속하는 url
def getImg():
    if request.method == 'POST':
        pic_data = request.files['file']
        filename = secure_filename(pic_data.filename) # 업로드 된 파일의 이름이 안전한가를 확인해주는 함수이다. 해킹 공격에 대해 보안을 하고자 사용되기도 한다.
        pic_data.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify(pred('./dog_image/{0}'.format(filename)))

@app.route('/getDB', methods = ['POST']) # 접속하는 url
def giveDB():
    if request.method == 'POST':
        index = request.form['index']
        results = collection.find({"id_": index}, {"_id" : False})
        row = []
        for i in results:
            row.append(i)
        return jsonify(row[0])
    
def pred(img_path):
    target = ['affenpinscher',
    'afghan_hound',
    'african_hunting_dog',
    'airedale',
    'american_staffordshire_terrier',
    'appenzeller',
    'australian_terrier',
    'basenji',
    'basset',
    'beagle',
    'bedlington_terrier',
    'bernese_mountain_dog',
    'black-and-tan_coonhound',
    'blenheim_spaniel',
    'bloodhound',
    'bluetick',
    'border_collie',
    'border_terrier',
    'borzoi',
    'boston_bull',
    'bouvier_des_flandres',
    'boxer',
    'brabancon_griffon',
    'briard',
    'brittany_spaniel',
    'bull_mastiff',
    'cairn',
    'cardigan',
    'chesapeake_bay_retriever',
    'chihuahua',
    'chow',
    'clumber',
    'cocker_spaniel',
    'collie',
    'curly-coated_retriever',
    'dandie_dinmont',
    'dhole',
    'dingo',
    'doberman',
    'english_foxhound',
    'english_setter',
    'english_springer',
    'entlebucher',
    'eskimo_dog',
    'flat-coated_retriever',
    'french_bulldog',
    'german_shepherd',
    'german_short-haired_pointer',
    'giant_schnauzer',
    'golden_retriever',
    'gordon_setter',
    'great_dane',
    'great_pyrenees',
    'greater_swiss_mountain_dog',
    'groenendael',
    'ibizan_hound',
    'irish_setter',
    'irish_terrier',
    'irish_water_spaniel',
    'irish_wolfhound',
    'italian_greyhound',
    'japanese_spaniel',
    'keeshond',
    'kelpie',
    'kerry_blue_terrier',
    'komondor',
    'kuvasz',
    'labrador_retriever',
    'lakeland_terrier',
    'leonberg',
    'lhasa',
    'malamute',
    'malinois',
    'maltese_dog',
    'mexican_hairless',
    'miniature_pinscher',
    'miniature_poodle',
    'miniature_schnauzer',
    'newfoundland',
    'norfolk_terrier',
    'norwegian_elkhound',
    'norwich_terrier',
    'old_english_sheepdog',
    'otterhound',
    'papillon',
    'pekinese',
    'pembroke',
    'pomeranian',
    'pug',
    'redbone',
    'rhodesian_ridgeback',
    'rottweiler',
    'saint_bernard',
    'saluki',
    'samoyed',
    'schipperke',
    'scotch_terrier',
    'scottish_deerhound',
    'sealyham_terrier',
    'shetland_sheepdog',
    'shih-tzu',
    'siberian_husky',
    'silky_terrier',
    'soft-coated_wheaten_terrier',
    'staffordshire_bullterrier',
    'standard_poodle',
    'standard_schnauzer',
    'sussex_spaniel',
    'tibetan_mastiff',
    'tibetan_terrier',
    'toy_poodle',
    'toy_terrier',
    'vizsla',
    'walker_hound',
    'weimaraner',
    'welsh_springer_spaniel',
    'west_highland_white_terrier',
    'whippet',
    'wire-haired_fox_terrier',
    'yorkshire_terrier'
    ]

    row = []
    image_pil = tf.keras.preprocessing.image.load_img(img_path, target_size=(200, 200, 3))
    image = np.array(image_pil)

    test_image = image / 255
    test_image = test_image.reshape((-1,) + test_image.shape)
    y_pred = model.predict(test_image)

    class_per = y_pred[0]
    icecream = dict(zip(target, class_per))

    row.append(img_path)

    count = 0
    for w in sorted(icecream, key=icecream.get, reverse=True):
        row.append(w)
        row.append(icecream[w])
        count += 1
        if count == 3: break

    return ({
        'answer' : row[0],
        'predFst' : row[1],
        'perFst' : str(row[2]),
        'predScd' : row[3],
        'perScd' : str(row[4]),
        'predThd' : row[5],
        'perThd' : str(row[6]),
    })

if __name__=="__main__":
    app.run(debug=True, port=5000)