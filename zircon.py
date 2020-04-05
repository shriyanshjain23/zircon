from flask import Flask, render_template, request, url_for, redirect, session, jsonify
from nlp import NLP
import sqlite3 as sq
import hashlib
from flask_sqlalchemy import SQLAlchemy
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv3D, BatchNormalization
from keras import backend as K
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import pandas as pd

############################################################################################################
k = NLP()
app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:@localhost/zircon"
db = SQLAlchemy(app)

def names(number):
    if (number == 0):
        return 'Tumor'
    else:
        return 'Normal'

def load_model():
    os.listdir('brain-mri-images-for-brain-tumor-detection')

    enc = OneHotEncoder()
    enc.fit([[0], [1]])

    data = []
    paths = []
    ans = []
    for r, d, f in os.walk(r'brain-mri-images-for-brain-tumor-detection/yes'):
        for file in f:
            if '.jpg' in file:
                paths.append(os.path.join(r, file))

    for path in paths:
        img = Image.open(path)
        x = img.resize((128, 128))
        x = np.array(x)
        if (x.shape == (128, 128, 3)):
            data.append(np.array(x))
            ans.append(enc.transform([[0]]).toarray())

    paths = []
    for r, d, f in os.walk(r"brain-mri-images-for-brain-tumor-detection/no"):
        for file in f:
            if '.jpg' in file:
                paths.append(os.path.join(r, file))

    for path in paths:
        img = Image.open(path)
        x = img.resize((128, 128))
        x = np.array(x)
        if (x.shape == (128, 128, 3)):
            data.append(np.array(x))
            ans.append(enc.transform([[1]]).toarray())

    data = np.array(data)
    data.shape

    ans = np.array(ans)
    ans = ans.reshape(139, 2)
    global model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding='Same'))
    model.add(Conv2D(32, kernel_size=(2, 2), activation='selu', padding='Same'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(2, 2), activation='selu', padding='Same'))
    model.add(Conv2D(64, kernel_size=(2, 2), activation='selu', padding='Same'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer='Adamax')
    x_train, x_test, y_train, y_test = train_test_split(data, ans, test_size=0.2, shuffle=True, random_state=69)

    history = model.fit(x_train, y_train, epochs=30, batch_size=40, verbose=1, validation_data=(x_test, y_test))

class Details(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(30), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(5), nullable=False)


class Contactus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(30), nullable=False)
    email = db.Column(db.String(30), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    message = db.Column(db.String(10000), nullable=True)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        entry = Contactus(name=name, email=email, subject=subject, message=message)
        db.session.add(entry)
        db.session.commit()

    return render_template('index.html')


@app.route("/popup.html", methods=['GET', 'POST'])
def popup():
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        age = request.form.get('age')
        weight = request.form.get('weight')
        height = request.form.get('height')
        gender = request.form.get('gender')
        entry = Details(name=name, phone=phone, email=email, age=age, weight=weight, height=height, gender=gender)
        db.session.add(entry)
        db.session.commit()
        return redirect(url_for('chatbot',username=name))

    return render_template('popup.html')

@app.route("/xyz/<username>")
def chatbot(username):
    return render_template('chat.html', username=username)

@app.route('/ask', methods=["POST"])
def ask():
    message = request.form['messageText']
    with open('chatting.txt', 'a') as f:
        f.write(message + '\n')
    res = k.processing(message)[1]
    print(message)
    return jsonify({'status': 'OK', 'answer': res})

@app.route('/logout')
def logout():
    return render_template('index.html')


@app.route("/servicespopup.html")
def shinzo():
    return render_template('servicespopup.html')

@app.route("/bt.html")
def hattori():
    return render_template('bt.html')

@app.route("/predict", methods=['GET','POST'])
def pre():
    loc="C:\\Users\\asus\\Desktop\\Zircon final\\Zircon\\static\\uploaded image"
    if (request.method=='POST'):
        file=request.files['timg']
        target=loc+"\\"+file.filename
        file.save(os.path.join(loc,file.filename))
        img = Image.open(target)
        x = np.array(img.resize((128, 128)))
        x = x.reshape(1, 128, 128, 3)
        answ = model.predict_on_batch(x)
        classification = np.where(answ == np.amax(answ))[1][0]
        result=str(answ[0][classification] * 100) + '% Confidence This Is '+ names(classification)
        location="/static/uploaded image"+"/"+file.filename
        if classification==0:
            return render_template('output_no.html',location=location,result=result)
        return render_template('output_yes.html',location=location,result=result)

@app.route("/more")
def more():
    stage=random.choice(["1st","2nd","3rd","Final"])
    return render_template('btpopup.html',stage=stage)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    load_model()
    app.run(threaded=False)
