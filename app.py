from wineReader.utils import *
from wineReader.model import *
from wineReader.labelVision import *
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from google.cloud import storage
import pathlib

# Simple url to image api call prediction

app = Flask(__name__)
CORS(app)

client = storage.Client()
bucket = client.get_bucket('plural-storage')

unet = Unet()
model = keras.models.load_model("models/unet.h5")

@app.route('/readLabel', methods = ['POST'])
def readLabel():
    user_input = request.json
    url = user_input['img_url']
    img_name = pathlib.PurePath(url).name.split(".")[0]

    X, img = img_url_to_input_unet(url)
    output = model.predict(X)

    label = labelVision()
    unwrapped_ocr, unwrapped = label.singleReadLabel(output[0], img)

    # write result to GCS
    # write locally
    cv2.imwrite('./tmp/tmp.jpg', unwrapped)
    blob = bucket.blob('processed/{}_response.jpg'.format(img_name))
    blob.upload_from_filename('./tmp/tmp.jpg')

    return jsonify({
        "transcript" : str(unwrapped_ocr)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)