from bs4 import BeautifulSoup
from collections import OrderedDict
import pprint
import requests

# with open('../data/archive/web_scrape_omeprazole.html', 'r') as html_file:
#     content = html_file.read()
#
#     soup = BeautifulSoup(content, 'lxml')
#     # print(soup.prettify())


def get_pathway(url_link, verbose=False):
    """Parse drug pathway data from smpdc.ca website"""
    # Send GET request
    response = requests.get(url_link)

    if response.status_code == 200:
        # parse the content as HTML
        try:
            soup = BeautifulSoup(response.content, "html.parser")
            # print(soup.prettify())

        except:
            if verbose:
                print('Could not find drug information for', url_link)
            soup = None

        if soup:
            path_way = soup.find('div', id='des_content')
            path_way_text = path_way.find_all('div')[-1].text.strip().replace('\n', ' ')
            if verbose:
                print(path_way_text)
            return path_way_text


def get_drugbank_information(drugbank_id, verbose=False):
    # Initialize
    global processed_id

    # URL to scrape
    url = f"https://go.drugbank.com/drugs/{drugbank_id}"

    # Send GET request
    response = requests.get(url)

    if response.status_code == 200:
        # parse the content as HTML
        try:
            soup = BeautifulSoup(response.content, "html.parser")
            # print(soup.prettify())

        except:
            print('Could not find drug information for', drugbank_id)
            soup = None
            processed_id.append(drugbank_id)

        if soup:
            # Extract each section (dl_element) from card_content
            content_container = soup.find('div', class_='content-container')
            drug_content = content_container.find('div', class_='drug-content')
            drug_card = drug_content.find('div', class_='drug-card')
            card_content = drug_card.find('div', class_='card-content')
            dl_elements = card_content.find_all('dl')

            # Initialize drug_info dict, ensure None value is return if not found
            keys = ['drug_id', 'generic_name', 'SMILES', 'drug_summary', 'brand_names', 'type',
                    'groups', 'weight', 'chemical_formula', 'synonyms', 'indication',
                    'conditions', 'pharmacodynamics', 'mechanism_of_action',
                    'target', 'protein_binding','metabolism', 'half_life','clearance',
                    'metabolism_pathway', 'action_pathway', 'UNII', 'CAS']
            drug_info = OrderedDict.fromkeys(keys)

            # Extracting texts
            for dl in dl_elements:
                # all term elements under each section
                dts = dl.find_all('dt')  # all terms, distinguish by id
                dds = dl.find_all('dd')  # all definition/texts for each term

                # Looping through each term, definition element
                for dt, dd in zip(dts, dds):
                    try:
                        dt_id = dt["id"]
                        ## 1. IDENTIFICATION
                        # Drug summary
                        if dt_id == 'summary':
                            drug_info['drug_summary'] = dd.text.strip().replace('\n', ' ')

                        # Brand names
                        if dt_id == 'brand-names':
                            drug_info['brand_names'] = dd.text.strip().replace('\n', ', ')

                        # Generic name
                        if dt_id == 'generic-name':
                            drug_info['generic_name'] = dd.text.strip()

                        # Drugbank id
                        if dt_id == 'drugbank-accession-number':
                            drug_info['drug_id'] = dd.text.strip()

                        # Drug Type
                        if dt_id == 'type':
                            drug_info['type'] = dd.text.strip()

                        # Group
                        if dt_id == 'groups':
                            drug_info['groups'] = dd.text.strip().replace('\n', ', ')

                        # Weight
                        if dt_id == 'weight':
                            drug_info['weight'] = dd.text.strip()

                        # Chemical Formula
                        if dt_id == 'chemical-formula':
                            drug_info['chemical_formula'] = dd.text.strip()

                        # Synonyms
                        if dt_id == 'synonyms':
                            synonyms = []
                            li_elements = dd.find_all('li')
                            for li in li_elements:
                                synonym = li.text.strip()
                                synonyms.append(synonym)
                            drug_info['synonyms'] = ' | '.join(synonyms)

                        ## 2. PHARMACOLOGY
                        # Indication
                        if dt_id == 'indication':
                            drug_info['indication'] = dd.text.strip().replace('\n', '').replace('â€¢', '.')

                        # Associated Conditions
                        if dt_id == 'associated-conditions':
                            conditions = []
                            li_elements = dd.find_all('li')
                            for li in li_elements:
                                condition = li.text.strip()
                                conditions.append(condition)
                            drug_info['conditions'] = ' | '.join(conditions)

                        # Contraindications
                        # if dt_id == 'contraindications-blackbox-warnings':
                        #     drug_info['contraindications'] = dd.text \
                        #         .replace('Avoid life-threatening adverse drug events', '') \
                        #         .replace(
                        #         'Improve clinical decision support with information on contraindications & blackbox warnings, population restrictions, harmful risks, & more.',
                        #         '') \
                        #         .replace('Learn more', '') \
                        #         .replace('Avoid life-threatening adverse drug events', '') \
                        #         .replace('& improve clinical decision support.', '').strip()

                        # Pharmacodynamics
                        if dt_id == 'pharmacodynamics':
                            drug_info['pharmacodynamics'] = dd.text.strip().replace('\n', ' | ')

                        # Mechanism of action & target
                        if dt_id == 'mechanism-of-action':
                            mechanisms = []
                            p_mechanism = dd.find_all('p')  # All paragraphs
                            for p in p_mechanism:
                                mechanism = p.text.strip()
                                mechanisms.append(mechanism)
                            drug_info['mechanism_of_action'] = ' | '.join(mechanisms).replace('\n', ' | ')

                            # Target
                            drug_targets = []
                            target_table = dd.find('table')
                            target_body = target_table.find('tbody')
                            target_protein = target_body.find_all('a')
                            target_span = dd.find_all('span')
                            target_class = dd.find_all('div')

                            for _span, _protein, _class in zip(target_span, target_protein, target_class):
                                target_info = f'Category: {_span.text}, Protein: {_protein.text}, Actions: {_class.text}'
                                drug_targets.append(target_info)

                            drug_info['target'] = ' | '.join(drug_targets)

                        # Protein Binding
                        if dt_id == 'protein-binding':
                            drug_info['protein_binding'] = dd.text.strip().replace('\n', ' | ')

                        # Metabolism
                        if dt_id == 'metabolism':
                            paragraph = dd.find('p').text.strip()
                            metabolite = dd.find('li').text.strip().replace('\n\n\n', ' > ').replace('\n', ' | ')
                            if metabolite :
                                paragraph = f'{paragraph} | Metabolite: {metabolite}'
                            drug_info['metabolism'] = paragraph

                        # Half-life
                        if dt_id == 'half-life':
                            drug_info['half_life'] = dd.text.strip().replace('\n', ' | ')

                        # Clearance
                        if dt_id == 'clearance':
                            drug_info['clearance'] = dd.text.strip().replace('\n', ' | ')

                        # Pathways
                        if dt_id == 'pathways':
                            path_table = dd.find('table')
                            path_body = path_table.find('tbody')
                            path_tr = path_body.find_all('tr')
                            for each_tr in path_tr:
                                path_td = each_tr.find_all('td')
                                url_link = path_td[0].find("a")["href"]
                                url_category = path_td[1].text
                                pathway = get_pathway(url_link)
                                if url_category == 'Drug metabolism':
                                    drug_info['metabolism_pathway'] = pathway
                                if url_category == 'Drug action':
                                    drug_info['action_pathway'] = pathway

                        # UNII
                        if dt_id == 'unii':
                            drug_info['UNII'] = dd.text.strip()

                        # CAS
                        if dt_id == 'cas-number':
                            drug_info['CAS'] = dd.text.strip()

                        # SMILES
                        if dt_id == 'smiles':
                            drug_info['SMILES'] = dd.text.strip()

                    except:
                        continue

            if verbose:
                pprint.pprint(drug_info)
            processed_id.append(drugbank_id)
            return drug_info


# Test function
# drugbank_id = 'DB00338'
drugbank_id = 'DB00316'
processed_id = []
get_drugbank_information(drugbank_id, verbose=True)

# url_link = "http://smpdb.ca/view/SMP0000710?highlight[compounds][]=DB00316&highlight[proteins][]=DB00316"
# get_pathway(url_link, verbose=True)


# with open('../data/drugbank_database.xml', 'r', encoding='utf8') as xml_file:
#     content = xml_file.read()
#     soup = BeautifulSoup(content, 'xml')
#     # print(soup.prettify())
#     drugs = soup.find_all("drug") # Find all the 'drug' elements
#     for drug in drugs: # Loop through each 'drug' element
#         drug_id = drug.find("drugbank-id", primary="true").text # Get the text of the first 'drugbank-id' element
#         name = drug.find("name").text # Get the text of the 'name' element
#         print(drug_id, name) # Print the extracted values


# import xml.etree.ElementTree as ET
# tree = ET.parse("../data/drugbank_database.xml") # Parse the XML file
# root = tree.getroot() # Get the root element
# print("root tags:", tree.tag)