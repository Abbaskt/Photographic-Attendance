# Code to read images from the folder containing extracted images of students
# from a class and recognise them using the model created.

import joblib
import os
from PIL import Image
from numpy import asarray, expand_dims
from sklearn.preprocessing import LabelEncoder, Normalizer

model = load_model("./Classifiers/facenet_keras.h5",compile=False)
predict_model = joblib.load("facenet_model.pkl")


face_sample_test = []
required_size = (160,160)

def get_embeddings(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean)/std

    face_expand = expand_dims(face,axis = 0)
    embedding = model.predict(face_expand)
    return embedding[0]

images = os.listdir("datasets/2")
for image in images:
    path = os.path.join("datasets/2",image)
    image = Image.open(path)
    image = image.convert('RGB')
    image = image.resize(required_size)
    face_array = asarray(image)
    face_sample_test.append(face_array)

test_face = []

for face in face_sample_test:
    test_embedding = get_embeddings(face)
    test_face.append(test_embedding)

in_encoder = Normalizer(norm='l2')
new_face = in_encoder.transform(test_face)

for test in new_face:
    samples = expand_dims(test, axis=0)
    yhat_class = predict_model.predict(samples)
    yhat_prob = predict_model.predict_proba(samples)
    print(yhat_class,yhat_prob*100)