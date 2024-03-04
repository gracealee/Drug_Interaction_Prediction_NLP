import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from streamlit_extras.app_logo import add_logo
import pandas as pd
import numpy as np
import requests
import json

st.set_page_config(page_title='DD.ai', page_icon = './vector_logo2.png')
# You can always call this function where ever you want.
add_logo('./vector_logo2tiny.png',height=50)

st.title("Sources")
st.header("Home Page")

st.caption("Annual severe ADE", help= "https://www.fda.gov/drugs/drug-interactions-labeling/preventable-adverse-drug-reactions-focus-drug-interactions")
st.caption("Preventable ADE", help= "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3305295/")
st.caption("Proportion of DDI as ADE", help= "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9483724/")
st.caption("Adult multi-drug use", help= "https://www.cdc.gov/nchs/products/databriefs/db347.htm")
st.caption("Animations from Lottie", help = "https://lottie.streamlit.app/")

st.header("Technology Page")
st.caption("Google/Pegasus-Pubmed", help="https://huggingface.co/google/pegasus-pubmed")
st.caption("Bio ClinicalBERT",help = "https://arxiv.org/abs/1904.03323")
