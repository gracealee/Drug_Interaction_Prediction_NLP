import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from streamlit_extras.app_logo import add_logo
from streamlit_extras.add_vertical_space import add_vertical_space
import re
import pandas as pd
import numpy as np
import requests
import json
import time
from rdkit import Chem
import altair as alt
from PIL import Image

st.set_page_config(page_title='DD.ai', page_icon = './vector_logo2.png')
# You can always call this function where ever you want.
add_logo('./vector_logo2tiny.png',height=50)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
#----- Bring in drug info for nutraceutical information -----#
info = pd.read_csv('scraping_drug_info.csv',sep = '\t')

#----- Center image -----#
col1, col2, col3 = st.columns(3)
with col1:
    st.write("")
with col2:
    st.image('./vector_logo1.png', use_column_width="auto")
with col3:
    st.write("")

#----- Center Title -----#
t1,t2,t3 = st.columns([1,14,1])
with t2:
    st.title("Predict Drug-Drug Interactions")
st.divider()

#----- Terms and Conditions -----#
with st.expander('**Terms and Conditions**'):
    st.write('Last updated July 24, 2023')
    st.write('AGREEMENT TO TERMS')
    st.write('''
These Terms and Conditions constitute a legally binding agreement made between you, whether personally or on behalf of an entity (“you”) and DD.ai (“we,” “us” or “our”), concerning your access to and use of the https://dd-ai-predict.streamlit.app/ website as well as any other media form, media channel, mobile website or mobile application related, linked, or otherwise connected thereto (collectively, the “Site”).
You agree that by accessing the Site, you have read, understood, and agree to be bound by all of these Terms and Conditions. If you do not agree with all of these Terms and Conditions, then you are expressly prohibited from using the Site and you must discontinue use immediately.
Supplemental terms and conditions or documents that may be posted on the Site from time to time are hereby expressly incorporated herein by reference. We reserve the right, in our sole discretion, to make changes or modifications to these Terms and Conditions at any time and for any reason.
We will alert you about any changes by updating the “Last updated” date of these Terms and Conditions, and you waive any right to receive specific notice of each such change.
It is your responsibility to periodically review these Terms and Conditions to stay informed of updates. You will be subject to, and will be deemed to have been made aware of and to have accepted, the changes in any revised Terms and Conditions by your continued use of the Site after the date such revised Terms and Conditions are posted.
The information provided on the Site is not intended for distribution to or use by any person or entity in any jurisdiction or country where such distribution or use would be contrary to law or regulation or which would subject us to any registration requirement within such jurisdiction or country.
Accordingly, those persons who choose to access the Site from other locations do so on their own initiative and are solely responsible for compliance with local laws, if and to the extent local laws are applicable.
            ''')
    st.write('INTELLECTUAL PROPERTY RIGHTS')
    st.write('''
Unless otherwise indicated, the Site is our proprietary property and all source code, databases, functionality, software, website designs, audio, video, text, photographs, and graphics on the Site (collectively, the “Content”) and the trademarks, service marks, and logos contained therein (the “Marks”) are owned or controlled by us or licensed to us, and are protected by copyright and trademark laws and various other intellectual property rights and unfair competition laws of the United States, foreign jurisdictions, and international conventions.
The Content and the Marks are provided on the Site “AS IS” for your information and personal use only. Except as expressly provided in these Terms and Conditions, no part of the Site and no Content or Marks may be copied, reproduced, aggregated, republished, uploaded, posted, publicly displayed, encoded, translated, transmitted, distributed, sold, licensed, or otherwise exploited for any commercial purpose whatsoever, without our express prior written permission.
Provided that you are eligible to use the Site, you are granted a limited license to access and use the Site and to download or print a copy of any portion of the Content to which you have properly gained access solely for your personal, non-commercial use. We reserve all rights not expressly granted to you in and to the Site, the Content and the Marks.
USER REPRESENTATIONS
By using the Site, you represent and warrant that:
(1) you have the legal capacity and you agree to comply with these Terms and Conditions;
(6) you will not access the Site through automated or non-human means, whether through a bot, script, or otherwise;
(7) you will not use the Site for any illegal or unauthorized purpose;
(8) your use of the Site will not violate any applicable law or regulation.
If you provide any information that is untrue, inaccurate, not current, or incomplete, we have the right to suspend or terminate your account and refuse any and all current or future use of the Site (or any portion thereof).
            ''')
    st.write('PROHIBITED ACTIVITIES')
    st.write('''
You may not access or use the Site for any purpose other than that for which we make the Site available. The Site may not be used in connection with any commercial endeavors except those that are specifically endorsed or approved by us.
As a user of the Site, you agree not to:
(1) systematically retrieve data or other content from the Site to create or compile, directly or indirectly, a collection, compilation, database, or directory without written permission from us.
(2) make any unauthorized use of the Site, including collecting usernames and/or email addresses of users by electronic or other means for the purpose of sending unsolicited email, or creating user accounts by automated means or under false pretenses.
(3) use a buying agent or purchasing agent to make purchases on the Site.
(4) use the Site to advertise or offer to sell goods and services.
(5) circumvent, disable, or otherwise interfere with security-related features of the Site, including features that prevent or restrict the use or copying of any Content or enforce limitations on the use of the Site and/or the Content contained therein.
(6) engage in unauthorized framing of or linking to the Site.
(7) make improper use of our support services or submit false reports of abuse or misconduct.
(8) engage in any automated use of the system, such as using scripts to send comments or messages, or using any data mining, robots, or similar data gathering and extraction tools.
(9) interfere with, disrupt, or create an undue burden on the Site or the networks or services connected to the Site.
(10) use the Site as part of any effort to compete with us or otherwise use the Site and/or the Content for any revenue-generating endeavor or commercial enterprise.
(11) decipher, decompile, disassemble, or reverse engineer any of the software comprising or in any way making up a part of the Site.
(12) attempt to bypass any measures of the Site designed to prevent or restrict access to the Site, or any portion of the Site.
(13) harass, annoy, intimidate, or threaten any of our employees or agents engaged in providing any portion of the Site to you.
(14) delete the copyright or other proprietary rights notice from any Content.
(15) copy or adapt the Site’s software, including but not limited to Flash, PHP, HTML, JavaScript, or other code.
(16) upload or transmit (or attempt to upload or to transmit) viruses, Trojan horses, or other material, including excessive use of capital letters and spamming (continuous posting of repetitive text), that interferes with any party’s uninterrupted use and enjoyment of the Site or modifies, impairs, disrupts, alters, or interferes with the use, features, functions, operation, or maintenance of the Site.
(17) upload or transmit (or attempt to upload or to transmit) any material that acts as a passive or active information collection or transmission mechanism, including without limitation, clear graphics interchange formats (“gifs”), 1×1 pixels, web bugs, cookies, or other similar devices (sometimes referred to as “spyware” or “passive collection mechanisms” or “pcms”).
(18) except as may be the result of standard search engine or Internet browser usage, use, launch, develop, or distribute any automated system, including without limitation, any spider, robot, cheat utility, scraper, or offline reader that accesses the Site, or using or launching any unauthorized script or other software.
(19) disparage, tarnish, or otherwise harm, in our opinion, us and/or the Site.
(20) use the Site in a manner inconsistent with any applicable laws or regulations.
            ''')
    st.write('SITE MANAGEMENT')
    st.write('''
We reserve the right, but not the obligation, to:
(1) monitor the Site for violations of these Terms and Conditions;
(2) take appropriate legal action against anyone who, in our sole discretion, violates the law or these Terms and Conditions, including without limitation, reporting such user to law enforcement authorities;
(3) in our sole discretion and without limitation, refuse, restrict access to, limit the availability of, or disable (to the extent technologically feasible) any of your Contributions or any portion thereof;
(4) in our sole discretion and without limitation, notice, or liability, to remove from the Site or otherwise disable all files and content that are excessive in size or are in any way burdensome to our systems;
(5) otherwise manage the Site in a manner designed to protect our rights and property and to facilitate the proper functioning of the Site.
            ''')
    st.write('PRIVACY POLICY')
    st.write('We care about data privacy and security. Please review our Privacy Policy [CLICK HERE]/posted on the Site]. By using the Site, you agree to be bound by our Privacy Policy, which is incorporated into these Terms and Conditions. Please be advised the Site is hosted in the United States. If you access the Site from the European Union, Asia, or any other region of the world with laws or other requirements governing personal data collection, use, or disclosure that differ from applicable laws in the United States, then through your continued use of the Site, you are transferring your data to the United States, and you expressly consent to have your data transferred to and processed in the United States.')
    st.write('DIGITAL MILLENNIUM COPYRIGHT ACT (DMCA) NOTICE AND POLICY')
    st.write('''We respect the intellectual property rights of others. If you believe that any material available on or through the Site infringes upon any copyright you own or control, please immediately notify our Designated Copyright Agent using the contact information provided below (a “Notification”).
A copy of your Notification will be sent to the person who posted or stored the material addressed in the Notification. Please be advised that pursuant to federal law you may be held liable for damages if you make material misrepresentations in a Notification. Thus, if you are not sure that material located on or linked to by the Site infringes your copyright, you should consider first contacting an attorney.
All Notifications should meet the requirements of DMCA 17 U.S.C. § 512(c)(3) and include the following information:
(1) A physical or electronic signature of a person authorized to act on behalf of the owner of an exclusive right that is allegedly infringed;
(2) identification of the copyrighted work claimed to have been infringed, or, if multiple copyrighted works on the Site are covered by the Notification, a representative list of such works on the Site;
(3) identification of the material that is claimed to be infringing or to be the subject of infringing activity and that is to be removed or access to which is to be disabled, and information reasonably sufficient to permit us to locate the material;
(4) information reasonably sufficient to permit us to contact the complaining party, such as an address, telephone number, and, if available, an email address at which the complaining party may be contacted;
(5) a statement that the complaining party has a good faith belief that use of the material in the manner complained of is not authorized by the copyright owner, its agent, or the law;
(6) a statement that the information in the notification is accurate, and under penalty of perjury, that the complaining party is authorized to act on behalf of the owner of an exclusive right that is allegedly infringed upon.
            ''')
    st.write('COPYRIGHT INFRINGEMENTS')
    st.write('We respect the intellectual property rights of others. If you believe that any material available on or through the Site infringes upon any copyright you own or control, please immediately notify us using the contact information provided below (a “Notification”). A copy of your Notification will be sent to the person who posted or stored the material addressed in the Notification. Please be advised that pursuant to federal law you may be held liable for damages if you make material misrepresentations in a Notification. Thus, if you are not sure that material located on or linked to by the Site infringes your copyright, you should consider first contacting an attorney.')
    st.write('TERM AND TERMINATION')
    st.write('These Terms and Conditions shall remain in full force and effect while you use the Site. WITHOUT LIMITING ANY OTHER PROVISION OF THESE TERMS AND CONDITIONS, WE RESERVE THE RIGHT TO, IN OUR SOLE DISCRETION AND WITHOUT NOTICE OR LIABILITY, DENY ACCESS TO AND USE OF THE SITE (INCLUDING BLOCKING CERTAIN IP ADDRESSES), TO ANY PERSON FOR ANY REASON OR FOR NO REASON, INCLUDING WITHOUT LIMITATION FOR BREACH OF ANY REPRESENTATION, WARRANTY, OR COVENANT CONTAINED IN THESE TERMS AND CONDITIONS OR OF ANY APPLICABLE LAW OR REGULATION. WE MAY TERMINATE YOUR USE OR PARTICIPATION IN THE SITE AT ANY TIME, WITHOUT WARNING, IN OUR SOLE DISCRETION.')
    st.write('MODIFICATIONS AND INTERRUPTIONS')
    st.write('''
We reserve the right to change, modify, or remove the contents of the Site at any time or for any reason at our sole discretion without notice. However, we have no obligation to update any information on our Site. We also reserve the right to modify or discontinue all or part of the Site without notice at any time.
We will not be liable to you or any third party for any modification, price change, suspension, or discontinuance of the Site.
We cannot guarantee the Site will be available at all times. We may experience hardware, software, or other problems or need to perform maintenance related to the Site, resulting in interruptions, delays, or errors.
We reserve the right to change, revise, update, suspend, discontinue, or otherwise modify the Site at any time or for any reason without notice to you. You agree that we have no liability whatsoever for any loss, damage, or inconvenience caused by your inability to access or use the Site during any downtime or discontinuance of the Site.
Nothing in these Terms and Conditions will be construed to obligate us to maintain and support the Site or to supply any corrections, updates, or releases in connection therewith.
            ''')
    st.write('CORRECTIONS')
    st.write('There may be information on the Site that contains typographical errors, inaccuracies, or omissions that may relate to the Site, including descriptions, pricing, availability, and various other information. We reserve the right to correct any errors, inaccuracies, or omissions and to change or update the information on the Site at any time, without prior notice.')
    st.write('DISCLAIMER')
    st.write('THE SITE IS PROVIDED ON AN AS-IS AND AS-AVAILABLE BASIS. YOU AGREE THAT YOUR USE OF THE SITE AND OUR SERVICES WILL BE AT YOUR SOLE RISK. TO THE FULLEST EXTENT PERMITTED BY LAW, WE DISCLAIM ALL WARRANTIES, EXPRESS OR IMPLIED, IN CONNECTION WITH THE SITE AND YOUR USE THEREOF, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. WE MAKE NO WARRANTIES OR REPRESENTATIONS ABOUT THE ACCURACY OR COMPLETENESS OF THE SITE’S CONTENT OR THE CONTENT OF ANY WEBSITES LINKED TO THE SITE AND WE WILL ASSUME NO LIABILITY OR RESPONSIBILITY FOR ANY (1) ERRORS, MISTAKES, OR INACCURACIES OF CONTENT AND MATERIALS, (2) PERSONAL INJURY OR PROPERTY DAMAGE, OF ANY NATURE WHATSOEVER, RESULTING FROM YOUR ACCESS TO AND USE OF THE SITE, (3) ANY UNAUTHORIZED ACCESS TO OR USE OF OUR SECURE SERVERS AND/OR ANY AND ALL PERSONAL INFORMATION AND/OR FINANCIAL INFORMATION STORED THEREIN, (4) ANY INTERRUPTION OR CESSATION OF TRANSMISSION TO OR FROM THE SITE, (5) ANY BUGS, VIRUSES, TROJAN HORSES, OR THE LIKE WHICH MAY BE TRANSMITTED TO OR THROUGH THE SITE BY ANY THIRD PARTY, AND/OR (6) ANY ERRORS OR OMISSIONS IN ANY CONTENT AND MATERIALS OR FOR ANY LOSS OR DAMAGE OF ANY KIND INCURRED AS A RESULT OF THE USE OF ANY CONTENT POSTED, TRANSMITTED, OR OTHERWISE MADE AVAILABLE VIA THE SITE. WE DO NOT WARRANT, ENDORSE, GUARANTEE, OR ASSUME RESPONSIBILITY FOR ANY PRODUCT OR SERVICE ADVERTISED OR OFFERED BY A THIRD PARTY THROUGH THE SITE, ANY HYPERLINKED WEBSITE, OR ANY WEBSITE OR MOBILE APPLICATION FEATURED IN ANY BANNER OR OTHER ADVERTISING, AND WE WILL NOT BE A PARTY TO OR IN ANY WAY BE RESPONSIBLE FOR MONITORING ANY TRANSACTION BETWEEN YOU AND ANY THIRD-PARTY PROVIDERS OF PRODUCTS OR SERVICES. AS WITH THE PURCHASE OF A PRODUCT OR SERVICE THROUGH ANY MEDIUM OR IN ANY ENVIRONMENT, YOU SHOULD USE YOUR BEST JUDGMENT AND EXERCISE CAUTION WHERE APPROPRIATE.')
    st.write('LIMITATIONS OF LIABILITY')
    st.write('IN NO EVENT WILL WE OR OUR DIRECTORS, EMPLOYEES, OR AGENTS BE LIABLE TO YOU OR ANY THIRD PARTY FOR ANY DIRECT, INDIRECT, CONSEQUENTIAL, EXEMPLARY, INCIDENTAL, SPECIAL, OR PUNITIVE DAMAGES, INCLUDING LOST PROFIT, LOST REVENUE, LOSS OF DATA, OR OTHER DAMAGES ARISING FROM YOUR USE OF THE SITE, EVEN IF WE HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.')
    st.write('INDEMNIFICATION')
    st.write('You agree to defend, indemnify, and hold us harmless, including our subsidiaries, affiliates, and all of our respective officers, agents, partners, and employees, from and against any loss, damage, liability, claim, or demand, including reasonable attorneys’ fees and expenses, made by any third party due to or arising out of: (1) your Contributions (2) use of the Site; (3) breach of these Terms and Conditions; (4) any breach of your representations and warranties set forth in these Terms and Conditions; (5) your violation of the rights of a third party, including but not limited to intellectual property rights; or (6) any overt harmful act toward any other user of the Site with whom you connected via the Site. Notwithstanding the foregoing, we reserve the right, at your expense, to assume the exclusive defense and control of any matter for which you are required to indemnify us, and you agree to cooperate, at your expense, with our defense of such claims. We will use reasonable efforts to notify you of any such claim, action, or proceeding which is subject to this indemnification upon becoming aware of it.')
    st.write('USER DATA')
    st.write('We will maintain certain data that you transmit to the Site for the purpose of managing the Site, as well as data relating to your use of the Site. Although we perform regular routine backups of data, you are solely responsible for all data that you transmit or that relates to any activity you have undertaken using the Site. You agree that we shall have no liability to you for any loss or corruption of any such data, and you hereby waive any right of action against us arising from any such loss or corruption of such data.')
    st.write('ELECTRONIC COMMUNICATIONS, TRANSACTIONS, AND SIGNATURES')
    st.write('''
    Visiting the Site, sending us emails, and completing online forms constitute electronic communications. You consent to receive electronic communications, and you agree that all agreements, notices, disclosures, and other communications we provide to you electronically, via email and on the Site, satisfy any legal requirement that such communication be in writing.
YOU HEREBY AGREE TO THE USE OF ELECTRONIC SIGNATURES, CONTRACTS, ORDERS, AND OTHER RECORDS, AND TO ELECTRONIC DELIVERY OF NOTICES, POLICIES, AND RECORDS OF TRANSACTIONS INITIATED OR COMPLETED BY US OR VIA THE SITE.
You hereby waive any rights or requirements under any statutes, regulations, rules, ordinances, or other laws in any jurisdiction which require an original signature or delivery or retention of non-electronic records, or to payments or the granting of credits by any means other than electronic means.
            ''')
    st.write('MISCELLANEOUS')
    st.write('''
These Terms and Conditions and any policies or operating rules posted by us on the Site constitute the entire agreement and understanding between you and us. Our failure to exercise or enforce any right or provision of these Terms and Conditions shall not operate as a waiver of such right or provision.
These Terms and Conditions operate to the fullest extent permissible by law. We may assign any or all of our rights and obligations to others at any time. We shall not be responsible or liable for any loss, damage, delay, or failure to act caused by any cause beyond our reasonable control.
If any provision or part of a provision of these Terms and Conditions is determined to be unlawful, void, or unenforceable, that provision or part of the provision is deemed severable from these Terms and Conditions and does not affect the validity and enforceability of any remaining provisions.
There is no joint venture, partnership, employment or agency relationship created between you and us as a result of these Terms and Conditions or use of the Site. You agree that these Terms and Conditions will not be construed against us by virtue of having drafted them.
You hereby waive any and all defenses you may have based on the electronic form of these Terms and Conditions and the lack of signing by the parties hereto to execute these Terms and Conditions.
            ''')

terms = st.checkbox("I accept the terms and conditions above")

#----- Disclaimer -----#
st.write('''
            DD.ai is **not** a substitution for lab drug-drug testing.
            Please view the predictions as a tool to inform your lab test prioritization.
            Just because our model did not predict an interaction, does not mean the interaction does not exist.
            Regardless of whether drugs are listed here or not, please remember to do your own diligence and further testing.
             ''')
disclaimer = st.checkbox("I understand the disclaimer above")

#----- Force users to agree to disclaimers -----#
if not terms:
    st.error("Please agree to the terms and conditions")
elif not disclaimer:
    st.error("Please agree to the disclaimer above")

#----- Create form for user Inputs -----#
else:
    # Be able to use css design file
    with open('streamlit.css') as site_design:
        #st.markdown(f'<style>{site_design.read()}</style>', unsafe_allow_html=True)
        with st.form('test', clear_on_submit = False):
            #disclaimer = st.checkbox("I understand the disclaimer above")

            # Smiles Input
            st.header("Let's add in the drug chemical structure")
            st.write("Please enter the structure in Simplified Molecular-Input Line-Entry System (SMILES) format")
            st.caption('e.g. The SMILES structure for Omeprazole is COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C')
            smiles_input = st.text_input("SMILES input:")
            m = Chem.MolFromSmiles(smiles_input, sanitize=False)

            # Pathway Input
            st.header("Let's add in an action pathway")
            st.write("Please limit to a maximum of 100 words")
            target_path = st.text_area("Action pathway:")

            # Submit Form Button
            submit_input = st.form_submit_button(label='Submit', help = 'Please submit a SMILE for the drug')

#Omeprazole smile
# COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C

#----- error handling for input submissions -----#
        if submit_input:
            # if not disclaimer:
            #     st.error("Please agree to the terms above")

            if len(smiles_input) == 0:
                st.error('Please input a SMILES structure')
            elif m is None:
                st.error('This is an invalid SMILES structure. Please try again.')
            elif len(target_path.split()) >= 100:
                word_count= len(target_path.split())
                st.error(f"The input is too long at {word_count} words. Please shorten.")

#----- Check for valid inputs and send inputs to model -----#
            elif disclaimer and m:
                st.toast('Sending SMILES to the model...')
                lottie_download = load_lottiefile("./atom.json")
                with st_lottie_spinner(lottie_download, key="download", height=150,
                    width=150):
                    st.toast('Predicting...')
                    if target_path:
                        payload = {"smiles": smiles_input, "target_pathway": target_path}
                    else:
                        payload = {"smiles": smiles_input, "target_pathway": 'Not Available'}
                    
#----- Model from AKS Endpoint -----#
                    # model_response = requests.post('https://maila.mids255.com/predict', data=json.dumps(payload))
                    # if model_response.status_code == 200:
                    ## Parse json object to get all interacting drugs
                        # pred_data = model_response.json()
                        
#----- Demo json when aks is not live -----# 
                    # Parse json object to get all interacting drugs
                    if submit_input:
                        if target_path:
                            m = open('omeprazole_model1.json')
                        else:
                            m = open('omeprazole_model2.json')
                        pred_data = json.load(m)
#----- End demo when aks is not live -----#

#----- Account for when model returns no predictions -----#
                        if len(pred_data['predictions']) == 0:
                            st.write('''
                                    Our model did not predict any interactions, but interactions may still exist.
                                    Please do your own diligence and further testing.
                                    ''')
#----- When model successfully returns predictions -----#
                        else:
                            # create dataframe from json predictions
                            pred_df = pd.DataFrame(columns=['Drug','Probability'])
                            for p in pred_data['predictions']:
                                new_row = pd.DataFrame({'Drug': p['drug_name'], 'Probability': p['score']}, index = [0])
                                pred_df = pd.concat([pred_df, new_row], axis=0, ignore_index = True)
                            # filter table to probability threshold
                            pred_df = pred_df[pred_df['Probability']>0.6]
                            pred_df.sort_values(by=['Probability'],ascending = False, inplace = True)
                            pred_df = pred_df.reset_index(drop = True)
                            size = len(pred_df)
                            st.header(f"Possible interactions with {size} drugs")

                            # Add Nutraceutical information
                            info['Type'] = np.where(info['groups'].str.contains('Nutraceutical'),'Nutraceutical','Drug')
                            pred_df = pred_df.merge(info[['generic_name','Type','conditions']], how = 'left', left_on = 'Drug', right_on = 'generic_name')
                            pred_df['Conditions'] = pred_df['conditions']
                            pred_df = pred_df.drop(['generic_name', 'conditions'], axis=1)
                            # clean up drug name field
                            pred_df['Drug'] =  pred_df['Drug'].apply(lambda x: re.sub(r'Commonly known or available as.*','', str(x)))

#----- Create objects for site (chart, tables, etc.) -----#
                            # Chart
                            chart_df = pred_df.head(20)
                            source = pd.DataFrame({
                                'Drug Name': chart_df['Drug'],
                                'Probability': chart_df['Probability'],
                                'Type': chart_df['Type'],
                                'Conditions': chart_df['Conditions']
                            })
                            # Account for when there are under 10 drug predictions
                            topten = 20
                            if len(chart_df)<10:
                                topten = len(chart_df)
                            # chart = alt.Chart(source, title = 'Top {} Highest Probability Drug Interactions'.format(topten)).mark_bar().encode(
                            #     x = alt.X('Drug Name', sort = None, axis=alt.Axis(labelAngle=280)),
                            #     y=alt.Y('Probability')
                            #     ).configure(numberFormat='%')
                            # Create the chart
                            selection = alt.selection_single()
                            chart = alt.Chart(source, title='Top {} Highest Probability Drug Interactions'.format(topten)).mark_bar().encode(
                                y=alt.Y('Drug Name:N', sort=None, axis=alt.Axis(labelAngle=0)),  # Use 'N' for nominal data
                                x=alt.X('Probability:Q', scale=alt.Scale(domain=[0.95, 1])),  # Use 'Q' for quantitative data
                                tooltip = [alt.Tooltip('Drug Name:N'),
                                    alt.Tooltip('Probability:Q'),
                                    alt.Tooltip('Conditions:N'),
                                    alt.Tooltip('Type:N'),
                                    ],
                                color=alt.condition(selection, 'Probability:Q', alt.value('grey'))
                            ).configure(numberFormat='%'
                            ).properties(
                                width=300, 
                                height=300,
                            ).add_selection(selection).interactive()


                            # For downloading table
                            def convert_df(df):
                                # IMPORTANT: Cache the conversion to prevent computation on every rerun?
                                # to do this, add cache decorator to this function: @st.cache
                                return df.to_csv().encode('utf-8')

#----- tab structure -----#
                            tab1, tab2, tab3 = st.tabs(['Top {} Drugs'.format(topten), 'Type', 'Table'])

                            # Show charts and table results when/if user inputs something
                            with tab1:
                                add_vertical_space(2)
                                #st.altair_chart(chart,use_container_width=True)
                                st.write(chart.properties(width = 600, height = 500))

                            # Add Nutraceutical/Drug Visual
                            with tab2:
                                add_vertical_space(2)
                                m1,m2 = st.columns(2)
                                m1.metric('Drug Interactions',len(pred_df[pred_df['Type']=='Drug']))
                                m2.metric('Nutraceutical Interactions',len(pred_df[pred_df['Type']!='Drug']))
                                st.divider()

                                piesource = pd.DataFrame({"Type": ['Drug', 'Nutraceutical'],
                                                    "Count": [len(pred_df[pred_df['Type']=='Drug']), len(pred_df[pred_df['Type']!='Drug'])]})
                                pie = alt.Chart(piesource).mark_arc(innerRadius=60).encode(
                                            theta="Count",
                                            color="Type"
                                        )
                                st.altair_chart(pie,use_container_width=True)

                            # update format of probability for table and show table
                            with tab3:
                                add_vertical_space(2)
                                pred_df['Probability'] = pred_df['Probability'].transform(lambda x: '{:,.2%}'.format(x))
                                st.dataframe(pred_df, use_container_width = True)

                                # Download table button
                                csv = convert_df(pred_df)
                                st.download_button(
                                    label="Download Table as CSV",
                                    data= csv,
                                    file_name="drug_interactions.csv",
                                    mime="text/csv"
                                    )
                                st.caption("This is a subset of prediction results. For a full list, please contact DD.ai.")
#----- If model response does not give status code 200 -----#
                    #else:

        st.markdown(f'<style>{site_design.read()}</style>', unsafe_allow_html=True)
