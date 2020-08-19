from flask import Flask
import eventlet.wsgi
import eventlet
import socketio
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': str(steering_angle),
                            'throttle': str(throttle) }, skip_sid=True)

model = load_model('model.h5')
model.summary()
@sio.on('telemetry')
def process_image(img):
    return img[10:130:2, ::4, :]
def telemetry(sid, data):
    if data:
        speed = float(data["speed"])
        image_str = data["image"]

        decoded = base64.b64decode(image_str)
        image = Image.open(BytesIO(decoded))
        image_array = np.asarray(image)

        # plt.imshow(image_array)
        # plt.show()
        img = process_image(image_array)
        img_batch = np.expand_dims(img, axis=0)
        steering_angle = float(model.predict(img_batch))
        # print(type(image_str))

        # steering_angle = -1.0  # -1.0 ... 1.0
        throttle = 0.1  # 0.1 ... 1.0
        if speed < 15:
            throttle = 0.5
        if speed >17:
            throttle = -0.1

        send_control(steering_angle, throttle)
    else:
        sio.emit('manual', data={}, skip_sid=True)


app = socketio.Middleware(sio, app)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)