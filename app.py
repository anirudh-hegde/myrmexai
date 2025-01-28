import streamlit as st
# from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
#
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

ant_model = load_model("myrmex.h5")
title = st.markdown("<h1 style='text-align: center; color: orange; font-size: 6vw'>MyrmexAI 🐜</h1>",
					unsafe_allow_html=True)

with st.sidebar:
	st.write("---")
	st.subheader("Follow me on: ")
	st.write("𝕏 -> [@Anirudh91017141](https://x.com/Anirudh91017141)")
	st.write("LinkedIn -> [anirudhhegde](https://www.linkedin.com/in/anirudhhegde/)")
	st.write("GitHub -> [anirudh-hegde](https://www.github.com/anirudh-hegde)")

upload_img = st.file_uploader("Choose an image")
if upload_img:
	img = load_img(upload_img, target_size=(180, 180))
	img_array = img_to_array(img)
	img_array = tf.expand_dims(img_array, axis=0)

	predictions = ant_model.predict(img_array)
	predicted_class = np.argmax(predictions)
	st.image(upload_img, width=400)
	# print(predicted_class)
	if predicted_class == 0:
		st.markdown("<h1 style='font-size: 1.5vw'>The uploaded image belongs to Fire Ants species</h1>", unsafe_allow_html=True)
	elif predicted_class == 1:
		st.markdown("<h1 style='font-size: 1.5vw'>The uploaded image belongs to Ghost Ants species</h1>", unsafe_allow_html=True)
	elif predicted_class == 2:
		st.markdown("<h1 style='font-size: 1.5vw'>The uploaded image belongs to Weaver Ants species</h1>", unsafe_allow_html=True)
	else:
		st.markdown("<h1 style='font-size: 1.5vw'>The uploaded image doesnt match Ants species</h1>", unsafe_allow_html=True)