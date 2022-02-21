import json
import shutil
from tqdm import tqdm
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request

def img_url_to_input_unet(url):
    
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    X=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X=cv2.resize(X,(256,256))

    return np.array([X]), img