# Set up
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import Draw
# from rdkit.Chem import rdmolops
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys


def get_fingerprints(smiles, verbose=False, fingerprints='Morgan'):
    # Encode to molecule object
    mol = Chem.MolFromSmiles(smiles)

    if fingerprints == 'Morgan':
        # Generate morgan fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    if fingerprints == 'MACCS':
        # SMARTS-based MACCS Keys which is a simplified version (less bits) 166 keys
        fp = MACCSkeys.GenMACCSKeys(mol)

    if fingerprints == 'RDKit':
        # Topological Fingerprints
        fp = AllChem.RDKFingerprint(mol)

    if verbose:
        print('Fingerprint', fp.ToBitString())

    return fp


def BraunBlanquetSimilarity(fp1, fp2):
    """Calculate Braun-Blanquet Similarity"""
    # Count the number of bits that are both 1 in the two fingerprints
    intersection = set(fp1.GetOnBits()) & set(fp2.GetOnBits())
    # Count the number of bits that are 1 in the larger fingerprint
    union = max(fp1.GetNumOnBits(), fp2.GetNumOnBits())
    # Return the ratio of intersection to union
    return len(intersection) / union


def get_similarity(fp1, fp2, metric="Tanimoto"):
    if metric == "Tanimoto": #Jaccard
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    if metric == "Braun-Blanquet":
        return BraunBlanquetSimilarity(fp1, fp2)

    if metric == "Dice":
        return DataStructs.DiceSimilarity(fp1, fp2)

    if metric == "Cosine":
        return DataStructs.CosineSimilarity(fp1, fp2)





### TEST BLOCK
smiles1 = 'COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C'
smiles2 = 'CN(C)CCCC1(OCC2=C1C=CC(=C2)C#N)C1=CC=C(F)C=C1'

# Morgan fingerprint
print('Morgan Fingerprints:')
fp1 = get_fingerprints(smiles1, fingerprints='Morgan', verbose=True)
print()
fp2 = get_fingerprints(smiles2, fingerprints='Morgan', verbose=True)

sim_tan = DataStructs.TanimotoSimilarity(fp1, fp2)
sim_bb = BraunBlanquetSimilarity(fp1, fp2)
sim_dice = DataStructs.DiceSimilarity(fp1, fp2)
sim_cosine = DataStructs.CosineSimilarity(fp1, fp2)
# sim2 = DataStructs.FingerprintSimilarity(fp1, fp2)  ## FingerprintSimilarity is actually equal to Tanimoto

print('Tanimoto Similarity:', sim_tan)
print('Braun-Blanquet Similarity:', sim_bb)
print('Dice Similarity:', sim_dice)
print('Cosine Similarity:', sim_cosine)

# ASP fingerprints
print('\nRDKit Fingerprints:')
fp1 = get_fingerprints(smiles1, fingerprints='RDKit')
fp2 = get_fingerprints(smiles2, fingerprints='RDKit')

sim_tan = DataStructs.TanimotoSimilarity(fp1, fp2)
sim_bb = BraunBlanquetSimilarity(fp1, fp2)
sim_dice = DataStructs.DiceSimilarity(fp1, fp2)
sim_cosine = DataStructs.CosineSimilarity(fp1, fp2)
# sim2 = DataStructs.FingerprintSimilarity(fp1, fp2)  ## FingerprintSimilarity is actually equal to Tanimoto

print('Tanimoto Similarity:', sim_tan)
print('Braun-Blanquet Similarity:', sim_bb)
print('Dice Similarity:', sim_dice)
print('Cosine Similarity:', sim_cosine)