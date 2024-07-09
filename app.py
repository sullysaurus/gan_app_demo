from flask import Flask, render_template, send_file
import numpy as np
import tensorflow as tf
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

def build_generator():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(100,)),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Dense(1024),
        tf.keras.layers.LeakyReLU(negative_slope=0.2),
        tf.keras.layers.Dense(28 * 28 * 1, activation='tanh'),
        tf.keras.layers.Reshape((28, 28, 1))
    ])
    return model

generator = build_generator()
generator.load_weights("generator_weights.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)
    generated_image = 0.5 * generated_image + 0.5
    buf = io.BytesIO()
    plt.imsave(buf, generated_image[0, :, :, 0], cmap='gray')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
