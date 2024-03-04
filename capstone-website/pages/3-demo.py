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

#----- Title -----#
t1,t2,t3 = st.columns([2.5,8,2.5])
with t2:
    st.title("How to Use the Model")

videofile = open('Demo.mp4','rb')
videobytes = videofile.read()
st.video(videobytes)