from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st


def extract_features(filename, model):
    try:
        image = Image.open(filename)
            
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'


st.title("Image Caption Generator")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

st.write(img_file_buffer)

st.image(
    Image.open(img_file_buffer), caption=f"Processed image", use_column_width=True,
)

st.write(img_file_buffer.name)

imageLoc="F:\Python\imagecaptiongenerator\FlickrTest\%s" % img_file_buffer.name



st.write(imageLoc)

max_length = 32
fileLoc="F:\Python\imagecaptiongenerator\mainToken.p"
modelLoc="F:\Python\imagecaptiongenerator\models\model_9.h5"
tokenizer = load(open(fileLoc,"rb"))
model = load_model(modelLoc)
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(imageLoc, xception_model) 

description = generate_desc(model, tokenizer, photo, max_length)

st.title(description)