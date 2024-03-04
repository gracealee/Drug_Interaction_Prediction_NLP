import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import time
from streamlit_lottie import st_lottie
from streamlit_extras.app_logo import add_logo
from PIL import Image

st.set_page_config(page_title='DD.ai', page_icon = './vector_logo2.png')
add_logo('./vector_logo2tiny.png',height=50)

# def add_logo(logo_path, width, height):
#     """Read and return a resized logo"""
#     logo = Image.open(logo_path)
#     modified_logo = logo.resize((width, height))
#     return modified_logo
# st.sidebar.image(add_logo(logo_path='./vector_logo1.png', width=50, height=50))

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

#----- Center image -----#
col1, col2, col3 = st.columns(3)
with col1:
    st.write("")
with col2:
    st.image('./vector_logo1.png', use_column_width="auto")
with col3:
    st.write("")

t1,t2,t3 = st.columns([5,5,5])
with t2:
    st.title("Technology")
st.divider()

#----- Data -----#
#t1,t2,t3 = st.columns([6,2,6])
#with t2:
st.header("Data")
left_col, middle_col = st.columns(2)

left_col.metric('Datasets', 2)
middle_col.metric('Total Drugs in Data', 2566)

st.write("We use data from DrugBank and the National Library of Medicine to" +
" extract information about different drugs and their features.")

st.write("The two features used are the SMILES structure and the action pathway. "+
"These features were curated based on user interviews with pharmacists and other "+
"industry experts.")

st.divider()

#----- Model Training -----#
#t1,t2,t3 = st.columns([6,2,6])
#with t2:
st.header("Training")
left_col, middle_col = st.columns(2)

left_col.metric("Models Built", 15)
middle_col.metric("Training Hours", 52)

st.write("We use Google/Pegasus-PubMed summarization model from HuggingFace " +
"to summarize the action pathway to put into our model.")
st.caption("Details about Google/Pegasus-Pubmed: https://huggingface.co/google/pegasus-pubmed")


st.write("We leverage Emily Alsentzer et. all's Bio ClinicalBERT model from HuggingFace, " +
"to embed the SMILES and action pathway to feed into our Neural Network models.")
st.caption("Details about Bio ClinicalBERT: https://arxiv.org/abs/1904.03323")

st.write("Using Pytorch, we create a neural network with three hidden layers and train. " +
"Training a model takes upward of 50 hours. We provide predictions against every drug in the database to facilitate and inform testing "+
"strategy before putting drugs on the market.")
st.divider()

#----- Best model -----#
#t1,t2,t3 = st.columns([6,4.5,6])
#with t2:
st.header("Final Model")
left_col, middle_col, right_col = st.columns(3)
left_col.metric("Number of Features", 2)
middle_col.metric("F2 Score", 0.83)
right_col.metric("AUPRC Score", 0.83)
