from tensorflow.keras.models import model_from_json
import coremltools as ct
import tensorflow as tf
import cv2
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model_weights.h5")

emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')



# Convert to Core ML
iosmodel = ct.convert(model)
iosmodel.short_description=" Facial Emotion Detection"
spec=iosmodel.get_spec()
#print(spec)
#iosmodel.save("ios3.mlpackage")
#iosmodel.save("ios3.mlmodel")