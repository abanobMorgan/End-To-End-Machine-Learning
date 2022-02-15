import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt

st.title("Customizable Neural Network ")



def model_create(num_neurons, activation): 
    model = Sequential()
    model.add(InputLayer((28, 28)))
    model.add(Flatten())
    model.add(Dense(num_neurons, activation))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model 
    
def preprocess_images(images):
    images = images / 255
    return images

num_neurons =  st.sidebar.slider('Number of neurons in hidden layer:', 1, 128, 10)
num_epochs = st.sidebar.slider('Number of epochs:', 1, 10, 4)
activation=  st.sidebar.text_input("activation function ", 'relu')
if st.button('Start train the model'):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)
    model = model_create(num_neurons, activation)
    model.summary()

    cp = ModelCheckpoint('model', save_best_only=True)
    history_cp=tf.keras.callbacks.CSVLogger('history.csv', separator=",", append=False)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, callbacks=[cp, history_cp])

if st.button('Evaluate the model'):

    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'], )
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    fig
