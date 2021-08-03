import flask
from flask import Flask,render_template,url_for,request,send_file
import pickle
import base64
import numpy as np
import cv2
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import io

model = tf.keras.models.load_model('static/vae_decoder_mnist.h5')

#Initializing new Flask instance. Find the html template in "templates".
app = flask.Flask(__name__, template_folder='templates')

#First route : Render the initial drawing template
@app.route('/')
def home():
	return render_template('draw.html')



#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['POST'])
def predict():
        if request.method == 'POST':
                data = request.get_json()
                latent_space_values = np.asarray([data['x'],data['y']])
                prediction = model.predict(latent_space_values.reshape((1,2)))
                prediction = prediction.reshape((28,28))

                def show_image(x):
                    #plt.imshow(np.clip(x + 0.5, 0, 1))
                    plt.imsave('static/prediction.png', np.clip(x+0.5,0,1) )

                show_image(prediction)


                return render_template('draw.html', prediction = prediction)


if __name__ == '__main__':
	app.run(debug=True)