import tensorflow as tf
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt
import keras_cv
from keras_cv.models import StableDiffusion
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
import tensorflow.keras as keras
import time
from PIL import Image
import cv2
import sys

SAVE_PATH = "/users/skoka/Documents/Final_Paper_ML/images"


Stable_diffusion = StableDiffusion(img_height=512, img_width=512)
decoder = Stable_diffusion.decoder

diffusion_model = DiffusionModel(img_width = 512, img_height = 512, max_text_length=0)

import math
def get_timestep_embedding(timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

# we are going to need to modify this code a lot 
def generate_pattern(layer_index, filter_index, size=64):
    # Build a model that outputs the activation
    # of the nth filter of the layer considered.
    layer_output = diffusion_model.layers[layer_index].output
    # Isolate the output 
    new_model = tf.keras.models.Model(inputs=diffusion_model.inputs, outputs=layer_output)
    
    # We start from a gray image with some uniform noise
    input_img_data = np.random.random((1, size, size, 4)) * 20 + 128.
    I = tf.Variable(input_img_data, name='image_var', dtype = 'float64')
    #I = preprocess_input(I_start) # only process once
    # Run gradient ascent for 40 steps
    eta = 5
    for i in range(300):
        start = time.time()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            time_embedding = get_timestep_embedding(i, 1)
            # create a tensor with the following size of all zeros: (None, 0, 768)
            word_embedding = tf.zeros((1, 0, 768))
            tape.watch(I)
            # get variable to maximize
            model_vals = new_model((I, time_embedding, word_embedding))
            loss = tf.reduce_mean(model_vals[:, :, :, filter_index])

        # Compute the gradient of the input picture w.r.t. this loss
        # add this operation input to maximize
        grad_fn = tape.gradient(loss, I)
        # Normalization trick: we normalize the gradient
        grad_fn /= (tf.sqrt(tf.reduce_mean(tf.square(grad_fn))) + 1e-5) # mean L2 norm
        I = I - (grad_fn * eta) # one iteration of maximizing
        end = time.time()
        # print("Iteration: {}, Loss: {}, Time: {}".format(i, loss, end-start))
    # decode the resulting input image
    I = decoder(I)
    # return the numpy matrix so we can visualize 
    img = I.numpy()
    return img




layer_index = int(sys.argv[1])
print("Layer Index:", layer_index)
filter_index = int(sys.argv[2])

print("Argument 1:", layer_index)
print("Argument 2:", filter_index)

file_name = "5th_best_layer_" + str(layer_index) + "_filter_" + str(filter_index) + ".png"

img_output = generate_pattern(layer_index, filter_index)
img_output = (img_output * 127.5) + 127.5
img_output = img_output.astype('uint8')
img = Image.fromarray(img_output[0], 'RGB')
img.save(SAVE_PATH + "/" +  file_name)