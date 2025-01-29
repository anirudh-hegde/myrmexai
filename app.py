import streamlit as st
from streamlit_option_menu import option_menu
# from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
#
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

ant_model = load_model("model/myrmex.h5")

st.set_page_config(
	page_title="MyrmexAI", page_icon="🐜"
)

with st.sidebar:
	pages = option_menu("",["Intro", "Get started"])
	st.write("---")
	st.subheader("Follow me on: ")
	st.write("𝕏 -> [@Anirudh91017141](https://x.com/Anirudh91017141)")
	st.write("LinkedIn -> [anirudhhegde](https://www.linkedin.com/in/anirudhhegde/)")
	st.write("GitHub -> [anirudh-hegde](https://www.github.com/anirudh-hegde)")

if pages == "Intro":
	st.markdown("<br><br><p style='font-size:26px'><strong>MyrmexAI</strong> is a project based on classification of ant species like"
			 "FireAnts, GhostAnts and WeaverAnts. The model was trained on three datasets using \
			 CNN and custom layers, achieving an accuracy of 87.9%<br><br></p>", unsafe_allow_html=True)
	st.markdown("<h4>How to get started</h4><br>"
				"1. Upload the ant image and let the model decide which species it belongs to<br>"
				"2. The model will predict the ant species for the uploaded image", unsafe_allow_html=True)
elif pages == "Get started":
	title = st.markdown("<h1 style='text-align: center; color: orange; font-size: 6vw'>MyrmexAI 🐜</h1>",
						unsafe_allow_html=True)
	upload_img = st.file_uploader("Choose an image")
	if upload_img:
		# if st.button(''):
			img = load_img(upload_img, target_size=(180, 180))
			img_array = img_to_array(img)
			img_array = tf.expand_dims(img_array, axis=0)
		
			predictions = ant_model.predict(img_array)
			predicted_class = np.argmax(predictions)
			st.image(upload_img, width=400)
			# print(predicted_class)
			if predicted_class == 0:
				st.markdown("<h1 style='font-size: 1.5vw'>The uploaded image belongs to Fire Ant species</h1>", unsafe_allow_html=True)
			elif predicted_class == 1:
				st.markdown("<h1 style='font-size: 1.5vw'>The uploaded image belongs to Ghost Ant species</h1>", unsafe_allow_html=True)
			elif predicted_class == 2:
				st.markdown("<h1 style='font-size: 1.5vw'>The uploaded image belongs to Weaver Ant species</h1>", unsafe_allow_html=True)
			else:
				st.markdown("<h1 style='font-size: 1.5vw'>The uploaded image doesnt match Ant species</h1>", unsafe_allow_html=True)
