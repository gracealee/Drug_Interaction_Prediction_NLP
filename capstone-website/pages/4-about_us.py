import streamlit as st
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
from streamlit_extras.app_logo import add_logo
from PIL import Image
import json

st.set_page_config(page_title='DD.ai', page_icon = './vector_logo2.png')
add_logo('./vector_logo2tiny.png',height=50)

#----- Center image -----#
col1, col2, col3 = st.columns(3)
with col1:
    st.write("")
with col2:
    st.image('./vector_logo1.png', use_column_width="auto")
with col3:
    st.write("")

t1,t2,t3 = st.columns([5,3.8,5])
with t2:
    st.title("About Us")
st.divider()

#----- Mission -----#
st.header("Our Mission")
st.subheader('At DD.ai, we are reimagining drug development by transforming drug-drug interaction detection.')
st.divider()

#----- Team -----#
st.header("Meet the Team")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Mai La")
    st.image('./mla-crop.jpg', width = 150)
    st.caption('Email: mai.la@berkeley.edu')
with col2:
    st.subheader("Grace Lee")
    st.image('./Grace pic.png', width = 150)
    st.caption('Email: grace_lee@berkeley.edu')
with col3:
    st.subheader("Radhika Mardikar")
    st.image('./radhika_pic copy.jpg', width = 150)
    st.caption('Email: rmardikar@berkeley.edu')

st.divider()
st.subheader("Who we are:")
st.write("We are a group of graduate students in the MIDS program at UC Berkeley")
st.divider()
st.subheader("Why we made this:")
st.write("We are passionate about the intersection of machine learning and biology.")
st.write(" By using this tool, we aim to significantly speed up drug development to cure the most prevalent diseases.")
