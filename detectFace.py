import keras
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions

def extract_cropped_face_from_image(face):
    mtcnn_detect_face = MTCNN()
    face_detected = mtcnn_detect_face.detect_faces(face)
    x1, y1, width, height = face_detected[0]['box']
    x2, y2 = x1 + width, y1 + height
    cropped_face = face[y1:y2, x1:x2]
    return cropped_face

def resizeFaceToModelSize(face, required_size=(224, 224)):
    faceRisezed = Image.fromarray(face)
    faceRisezed = faceRisezed.resize(required_size)
    return faceRisezed

def convertToFaceArray(resizedFace):
    return asarray(resizedFace)

def loadFaceFromFile(fileName):
    return pyplot.imread(fileName)

def extract_face(face):
    cropped_face = extract_cropped_face_from_image(face)
    resized_face = resizeFaceToModelSize(cropped_face)
    faceArray = convertToFaceArray(resized_face)
    return faceArray

face = loadFaceFromFile('images/david.jpeg')
pixels = extract_face(face)
pixels = pixels.astype('float32')
samples = expand_dims(pixels, axis=0)
samples = preprocess_input(samples, version=2)
model = VGGFace(model='resnet50')
what = model.predict(samples)
results = decode_predictions(what)
for result in results[0]:
    print('%s: %.3f%%' % (result[0], result[1]*100))