from argparse import ArgumentParser
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras.models import load_model
from keras.metrics import top_k_categorical_accuracy
import keras.applications as apps
import pickle
import cv2
import os
import functools

                       
from keras.applications.vgg16 import preprocess_input

parser = ArgumentParser()
parser.add_argument('pathModel', help='path to load model', type=str)
parser.add_argument('pathData', help='path to image (test)', type=str)
parser.add_argument('-k', type=int, help='Top K responses', default=1)
args = parser.parse_args()


modelPath = args.pathModel 	#'VGG'
imagePath = args.pathData 	#'images/TEST/Achillea_millefolium/AM1.jpeg'
topk        = args.k

def getTopK(answer: np.array, class_list: list, K: int = 5):
    '''Get top N ordered answers'''
    top_answers = sorted([[i, val] for i, val in enumerate(answer)], key=lambda x: x[1], reverse=True)
    return [(class_list[i], val) for i, val in top_answers[:K]]


with open(modelPath + '.bin', 'rb') as class_file:
    modelName, classes = pickle.load(class_file)
if isinstance(classes, LabelBinarizer):
    classes = classes.classes_
elif isinstance(classes, OneHotEncoder):
    classes = classes.classes
else:
    raise TypeError('Classes object type is not supported ({}).'.format(type(classes).__name__))


# Top-1 metric
top1 = functools.partial(top_k_categorical_accuracy, k=1)
top1.__name__ = 'top1'
# Top-5 metric
top5 = functools.partial(top_k_categorical_accuracy, k=5)
top5.__name__ = 'top5'

#image
print('\nTest image: ' + imagePath  + '\n')

#load model
print('Loading model: ' + modelPath  + '.h5\n')
model = load_model(os.path.abspath(modelPath  + '.h5'), custom_objects={"top1": top1,"top5": top5})

# setting inputs
image_dim = 224
input_shape = (image_dim, image_dim, 3)

#read and preprocessing the image
img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
if img.shape != (image_dim,image_dim):
    img = cv2.resize(img, (image_dim,image_dim))

img_array = np.expand_dims(img, axis=0)
img_array = preprocess_input(img_array)

####################### Prediction
y_pred1 = model.predict(img_array, steps=1)[0]
#pred = np.argmax(y_pred1, axis=1)

#model response
responses = getTopK(y_pred1, classes, topk)
output = '\n'.join('{},\t{}'.format(*x) for x in responses)
print('\nPredictions:\n'+ output)
