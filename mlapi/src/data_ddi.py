import pandas as pd
import numpy as np
import gensim
import pickle
import os
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch

# Test input smiles
example_smiles = "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C"  # Omeprazole
example_pathway = "bemeprazole is used to manage gastroesophageal reflux disease to prevent stomach ulcers , and to help manage the effects of infection . Its effects are covalently bound to the protein noncovalentlytopase enzymes because additional dose - dependent enzymes must be created to replace the ones that bind by pantoprazole."

tokenizer = AutoTokenizer.from_pretrained('./morgan-embed-bio-clinical-bert-ddi')

def get_fingerprints(smiles, verbose=False, fingerprints='Morgan'):
    # Encode to molecule object
    mol = Chem.MolFromSmiles(smiles)

    if fingerprints == 'Morgan':
        # Generate morgan fingerprint
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        except:
            fp = None

    if fingerprints == 'MACCS':
        # SMARTS-based MACCS Keys which is a simplified version (less bits) 166 keys
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
        except:
            fp = None

    if fingerprints == 'RDKit':
        # Topological Fingerprints
        try:
            fp = AllChem.RDKFingerprint(mol)
        except:
            fp = None

    if verbose:
        print('Fingerprint', fp.ToBitString())

    return fp


def ConsineSimilarity(fp1_arr, fp2_arr):
    """Calculate Cosine Similarity from fingerprints array"""
    # Convert bits array to idx of ones value
    fp1 = set(np.nonzero(fp1_arr)[0])
    fp2 = set(np.nonzero(fp2_arr)[0])

    # Count the number of bits that are both 1 in the two fingerprints
    intersection = len(fp1 & fp2)
    # Get the L1 length from two fingerprints
    product_length = np.sqrt(len(fp1)*len(fp2))
    # Return the ratio of intersection to union
    if product_length > 0:
        return round(intersection / product_length, 7)
    else:
        return 0


def TanimotoSimilarity(fp1_arr, fp2_arr):
    """Calculate Cosine Similarity from fingerprints array"""
    # Convert bits array to idx of ones value
    fp1 = set(np.nonzero(fp1_arr)[0])
    fp2 = set(np.nonzero(fp2_arr)[0])
    # Count the number of bits that are both 1 in the two fingerprints
    intersection = len(fp1 & fp2)
    # Count the Union
    union  = len(fp1) + len(fp2) - intersection
    # Return the ratio of intersection to union
    if union > 0:
        return round(intersection / union, 7)
    else:
        return 0


def embedding_fingerprints(smiles, fingerprints="Morgan", similarity="Cosine"):
    """
    Input: a SMILES string representation
    Output: Embedding from fingerprints & similarity
    Extracting Embedding Vector From Fingerprints & Similarity Type
        fingerprints: Morgan, RDKit, MACCS
        Similarity: Cosine, Tanimoto
    """
    # Extract fingerprints from smiles:
    try:
        fp1_arr = np.array(get_fingerprints(smiles, fingerprints=fingerprints))
    except:
        return None

    # Extract fingerprints basis matrix
    with open("morgan-embed-bio-clinical-bert-ddi/basis_300centroids_fingerprints.pkl", "rb") as f:
        basis = pickle.load(f)
    basis_matrix = basis[fingerprints]

    # Initialize embedding
    embedding = []

    # Calculate Similarity for drug fingerprint with the rest of the drugs in the basis
    for fp2_arr in basis_matrix:
        if similarity == "Cosine":
            drug_similarity = ConsineSimilarity(fp1_arr, fp2_arr)
        if similarity == "Tanimoto":
            drug_similarity = TanimotoSimilarity(fp1_arr, fp2_arr)
        embedding.append(drug_similarity)

    return np.array(embedding)


def trim_to_250_words(text):
    words = text.split()
    return " ".join(words[:250])


def trim_to_100_words(text):
    words = text.split()
    return " ".join(words[:100])


def clean_up_text(x):
    """Remove line breaks, special characters, punctuations within each post"""
    # Remove special characters and punctuations
    SPECIAL_CHARS_PATTERN = re.compile(r"(\*)|(\=\=)|(\~)|(\=)|(\.\.\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
    x = SPECIAL_CHARS_PATTERN.sub("", x.lower())

    # Remove different types of line breaks and white spaces
    x = re.sub(r"\n|\r|\r\n|<br\s*/?>", " ", x)

    # Remove extra white spaces
    x = re.sub(r"\s+", " ", x.strip())

    return x


def load_approved_drugs(smiles1=example_smiles, d1_pathway=example_pathway, embed_smiles="Morgan", similarity="Cosine"):
    # Import Drug2Vec Embedding - Morgan Fingerprints, Cosine Similarity
    if similarity == "Cosine" :
        drug2vec_model = gensim.models.KeyedVectors.load_word2vec_format(fname=f"./morgan-embed-bio-clinical-bert-ddi/drug2vec_300clusters_morgan_cosine.txt", binary=False)
    else:
        # Default encoding with Morgan Fingerprint & Tanimoto Similarity
        drug2vec_model = gensim.models.KeyedVectors.load_word2vec_format(fname=f"./morgan-embed-bio-clinical-bert-ddi/drug2vec_300clusters_morgan_tanimoto.txt", binary=False)
    valid_smiles = drug2vec_model.index_to_key

    # Load all approved drugs for running inference with the input SMILES
    drug_info_df= pd.read_csv('./morgan-embed-bio-clinical-bert-ddi/drug_info_smiles_pathway.csv', delimiter='\t')
    df = drug_info_df[drug_info_df["approved_drug"] == 1]

    ## FOR TESTING ONLY, ONLY Do inference for the first 128 drugs
    # df = df.head(128)

    df = df[["drug_id", "generic_name", "SMILES", "smiles_pathway"]]
    df = df.rename(columns={"drug_id": "drug2_id","generic_name": "drug2_name", "SMILES": "smiles2", "smiles_pathway": "d2_pathway"})

    # Extract Morgan Fingerprint Embedding
    # Remove the 4 SMILES causing error, only include SMILES in drug2vec.txt
    df = df[df['drug2_id'].isin(valid_smiles)]

    # Add Drug1 from input
    df["smiles1"] = smiles1
    df["d1_pathway"] = smiles1[:250] + ' ' + trim_to_100_words(clean_up_text(d1_pathway))

    # Combine Target1 and Target2 Together, using [SEP] token to distingush between drug1 & drug2
    df["pathway_features"] = df["d1_pathway"] + " [SEP] " + df["d2_pathway"]

    # Combine SMILES1 and SMILES2 Together
    df["smiles_features"] = df["smiles1"].str[:254] + " [SEP] " + df["smiles2"].str[:254] + " [SEP]"

    # Extract embedding
    if embed_smiles in ("Morgan", "RDKit", "MACCS"):
        df['d2_embedding'] = df['drug2_id'].apply(lambda x: drug2vec_model[x])
        df['d1_embedding'] = df["smiles1"].apply(lambda x: embedding_fingerprints(x, fingerprints=embed_smiles, similarity=similarity))

    else:
        # Default Morgan Fingerprint
        df['d2_embedding'] = df['drug2_id'].apply(lambda x: drug2vec_model[x])
        df['d1_embedding'] = df["smiles1"].apply(lambda x: embedding_fingerprints(x, fingerprints="Morgan", similarity=similarity))

    return df


def BuildDataLoader(smiles1=example_smiles,
                    d1_pathway=example_pathway,
                    embed_smiles="Morgan",
                    similarity="Cosine",
                    tokenizer=tokenizer,
                    model="MorganBioBert",
                    verbose=False):
    # Extract Morgan Embedding for input SMILES, and all approved drugs
    df = load_approved_drugs(smiles1=smiles1, d1_pathway=d1_pathway, embed_smiles=embed_smiles, similarity=similarity)
    drug2_ids = df['drug2_id'].values
    drug2_names = df['drug2_name'].values
    WORKERS = int(os.cpu_count())

    # Concat morgan embedding for both drugs SMILES
    if embed_smiles in ("Morgan", "RDKit", "MACCS"):
        d1_embed = np.array(df['d1_embedding'].tolist())
        d2_embed = np.array(df['d2_embedding'].tolist())
        X_smiles = np.hstack((d1_embed, d2_embed))
        smiles_embedding = torch.from_numpy(X_smiles)
        smiles_embedding = smiles_embedding.to(torch.float32)

        # Using BERT Model for Drug Target
        X_pathway = df["pathway_features"]

        # Tokenize Target
        _encodings = tokenizer(
            list(X_pathway.values),
            max_length=512,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_tensors='pt'
        )

        _ids = _encodings.input_ids
        _mask = _encodings.attention_mask

        # Build DataLoader for batch inference
        _dataset = TensorDataset(_ids, _mask, smiles_embedding)
        _loader = DataLoader(_dataset, batch_size=64, shuffle=False, num_workers=WORKERS)

        if verbose:
            print('Drug1 Embedding Dimension:',d1_embed.shape)
            print('Drug2 Embedding Dimension:',d2_embed.shape)
            print('Drug 1 Concatenate with Drug 2 Embedding Dimension:', X_smiles.shape)
            print('SMILES Morgan Embedding Dimension:', smiles_embedding.shape)
            print(smiles_embedding[0])
            print(smiles_embedding[1])
            print('Drug Action Pathway BERT Embedding Dimension:', X_pathway.shape)
            print('Encoding input_ids Dimension:', _ids.shape)
            print(_ids[0])
            print('Encoding attention_mask Dimension:', _mask.shape)
            print(_mask[0])
            print('Data Loader Len:', len(_loader))

        return _loader, drug2_ids, drug2_names

    else:
        d1_embed = np.array(df['d1_embedding'].tolist())
        d2_embed = np.array(df['d2_embedding'].tolist())
        X_smiles = np.hstack((d1_embed, d2_embed))
        smiles_embedding = torch.from_numpy(X_smiles)
        smiles_embedding = smiles_embedding.to(torch.float32)

        # Using BERT Model for Drug SMILES
        X_smiles = df["smiles_features"]

        # Tokenize
        _encodings = tokenizer(
            list(X_smiles.values),
            max_length=512,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_tensors='pt'
        )

        _ids = _encodings.input_ids
        _mask = _encodings.attention_mask

        # If Not runing Morgan BioBert Model
        if model == "BioClinicalBert":
            _dataset = TensorDataset(_ids, _mask)
        else:
            # Default with MorganBioBert Model
            _dataset = TensorDataset(_ids, _mask, smiles_embedding)

        # Data Loader
        _loader = DataLoader(_dataset, batch_size=64, shuffle=False, num_workers=WORKERS)
        if verbose:
            print('Data Loader Len:', len(_loader))

        return _loader, drug2_ids, drug2_names


## Test Embedding
# print('Embedding with Morgan Similarity:', example_smiles)
# print(embedding_fingerprints(example_smiles, fingerprints="Morgan", similarity="Cosine")[:10])
#
# Test data loader
# BuildDataLoader(embed_smiles="Morgan", verbose=True)