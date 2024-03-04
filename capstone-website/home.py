import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from streamlit_extras.app_logo import add_logo
import pandas as pd
import numpy as np
import requests
import json
import time
from rdkit import Chem
import altair as alt
from PIL import Image
st.set_page_config(page_title='DD.ai', page_icon = './vector_logo2.png')

# You can always call this function where ever you want
add_logo('./vector_logo2tiny.png',height=50)

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

t1, t2, t3 = st.columns([4,1.7,4])
with t1:
    st.write("")
with t2:
    st.title("DD.ai")
with t3:
    st.write("")

#----- Tagline -----#
st.divider()
st.subheader('Reimagine Drug Development: Transforming DDI Detection')
st.divider()

t1, t2, t3 = st.columns([4,2.5,4])
with t1:
    st.write("")
with t2:
    st_lottie(load_lottiefile('./pharmacy.json'), height=200,width=200)
with t3:
    st.write("")

#----- Problem -----#
st.subheader('Drug - Drug Interactions (DDI) may cause severe effects')
left_col, right_col = st.columns(2)
left_col.metric('Severe Adverse Drug Effects (ADE) Annually', '2.2M+', help = 'source: https://www.fda.gov/drugs/drug-interactions-labeling/preventable-adverse-drug-reactions-focus-drug-interactions')
right_col.metric('% of Preventable ADE','52%', help = 'source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3305295/')
left_col.metric('DDI as % of ADE', '18.3%', help = 'source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9483724/')
right_col.metric('Adults using 5+ Prescription Drugs','22.4%', help = 'source: https://www.cdc.gov/nchs/products/databriefs/db347.htm')
# Source: severe ADE https://www.fda.gov/drugs/drug-interactions-labeling/preventable-adverse-drug-reactions-focus-drug-interactions
# Source: Preventable ADE https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3305295/
# Source: 18.3% of adverse drug effects: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9483724/
# Source: 22.4% use atleast 5 drugs https://www.cdc.gov/nchs/products/databriefs/db347.htm

t1, t2, t3 = st.columns([4,4,4])
with t1:
    st_lottie(load_lottiefile('./sthetoscope.json'), height=200,width=200)
with t2:
    st_lottie(load_lottiefile('./sthetoscope.json'), height=200,width=200)
with t3:
    st_lottie(load_lottiefile('./sthetoscope.json'), height=200,width=200)

st.subheader("Drug development & DDI testing is expensive")
left_col, right_col = st.columns(2)
left_col.metric('Total Global Spending by 2032', '$133B', help = 'source: https://www.precedenceresearch.com/drug-discovery-market')
right_col.metric("Increases in FDA Drug Approval 2010-2019", '60%', help = 'source: https://www.cbo.gov/publication/57126#:~:text=On%20average%2C%20the%20Food%20and,average%20over%20the%20previous%20decade.')
# st.write("In 2021, drug interaction testing was 75 billion dollars of global spending, " +
# "which is expected to grow to around 162 billion dollars by 2030. With this growth" +
# " is also a growth in drug approval, for example there was a 60% increase in" +
# " approvals in 2010-2019 compared with the previous decade. And testing takes time," +
# " where a typical panel of in vitro testing is 4-6 months. " +
# "With this growth it is just not feasible to test all drug combinations.")

## Source: global market: https://www.precedenceresearch.com/drug-discovery-market
## Source: increases drug approval https://www.cbo.gov/publication/57126#:~:text=On%20average%2C%20the%20Food%20and,average%20over%20the%20previous%20decade.

st.divider()

#----- Solution -----#
st.header("Our Solution")
left_col, right_col = st.columns(2)

st.write("""
        As the drug discovery market grows, it is becoming increasingly impractical to test every drug combination in the lab.
        At DD.ai, we use deep learning to help our customers optimize drug development and lab testing strategies.
        Using NLP techniques on Simplified Molecular Input Line Entry System (SMILES) representation of drug molecules,
        we can predict a potential drug interaction with other drugs and nutraceuticals currently on the market.
        """)


st.divider()
#st.write("Sources:")
#st.caption('Severe ADE https://www.fda.gov/drugs/drug-interactions-labeling/preventable-adverse-drug-reactions-focus-drug-interactions')
#st.caption("Preventable ADE https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3305295/")
#st.caption("18.3% of adverse drug effects: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9483724/")
#st.caption("22.4% use at least 5 drugs https://www.cdc.gov/nchs/products/databriefs/db347.htm")
#st.caption("Global market: https://www.precedenceresearch.com/drug-discovery-market")
#st.caption("Increases drug approval https://www.cbo.gov/publication/57126#:~:text=On%20average%2C%20the%20Food%20and,average%20over%20the%20previous%20decade.")
