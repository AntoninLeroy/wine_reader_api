from wineReader.utils import *
from wineReader.labelVision import *
from flask import Flask, jsonify, request
from flask_cors import CORS
from waitress import serve
from PIL import Image
from io import BytesIO
from google.cloud import storage
from tensorflow import keras
import pathlib

# Simple url to image api call prediction

app = Flask(__name__)
CORS(app)

model = keras.models.load_model("models/unet.h5")

@app.route('/readLabel', methods = ['POST'])
def readLabel():
    user_input = request.json
    url = user_input['img_url']
    img_name = pathlib.PurePath(url).name.split(".")[0]

    X, img = img_url_to_input_unet(url)
    output = model.predict(X)

    label = labelVision()
    unwrapped_ocr = label.singleReadLabel(output[0], img, img_name)

    return jsonify({
        "unwrapped_ocr" : str(unwrapped_ocr),
        "unet_predict_url" : str("https://storage.googleapis.com/plural-storage/processed/{}_unet_predict.jpg".format(img_name)),
        "cylinder_url" : str("https://storage.googleapis.com/plural-storage/processed/{}_cylinder.jpg".format(img_name)),
        "mesh_url" : str("https://storage.googleapis.com/plural-storage/processed/{}_mesh.jpg".format(img_name)),
        "unwrapped_url" : str("https://storage.googleapis.com/plural-storage/processed/{}_unwrapped.jpg".format(img_name))
    })

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)