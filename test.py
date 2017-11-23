import keras
from keras.models import load_model
import glob
from PIL import Image
import numpy as np
import cv2

model = load_model('age_rennet50_tuned.h5')

test_dir = '/root/sxwl-dataset/nas/luoyulong/incoming/custom/'
imgs_path = glob.glob(test_dir + "*.jpg")

for img_path in imgs_path:
    #print img_path
    im = cv2.imread(img_path)
    #print im
    im = cv2.resize(im, (224, 224))
    #print im
    im = im / 255.
    im = np.expand_dims(im, axis=0)
    probs = model.predict(im, 1)
    pred = np.argmax(probs)
    print img_path + '   ' + str(pred)