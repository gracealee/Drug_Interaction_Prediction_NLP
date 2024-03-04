import pytest
from fastapi.testclient import TestClient
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from src.main import app


@pytest.fixture
def client():
    FastAPICache.init(InMemoryBackend())
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint_pathway(client):
    data = {"smiles":"COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C",
            "target_pathway": "bemeprazole is used to manage gastroesophageal reflux disease to prevent stomach ulcers , and to help manage the effects of infection . Its effects are covalently bound to the protein noncovalentlytopase enzymes because additional dose - dependent enzymes must be created to replace the ones that bind by pantoprazole."}
    response = client.post(
        "/predict",
        json=data,
    )

    assert response.status_code == 200
    assert type(response.json()["predictions"]) is list
    assert type(response.json()["predictions"][0]) is dict
    assert set(response.json()["predictions"][0].keys()) == {"drug_id", "drug_name", "score"}
    assert response.json()["predictions"][0]["score"] >= 0.5

def test_predict_endpoint_smiles_only(client):
    data = {"smiles":"COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C",
            "target_pathway": "Not Available"}
    response = client.post(
        "/predict",
        json=data,
    )

    assert response.status_code == 200
    assert type(response.json()["predictions"]) is list
    assert type(response.json()["predictions"][0]) is dict
    assert set(response.json()["predictions"][0].keys()) == {"drug_id", "drug_name", "score"}
    assert response.json()["predictions"][0]["score"] >= 0.5


# # Test Model Output
# from model import MorganBioBertClassification, BioClinicalBertClassification, predict_scores
# from data_ddi import BuildDataLoader
#
# # model = BioClinicalBertClassification()
# # checkpoint = torch.load("./bioclinicalbertcheckpoint-cpu.pth.tar")
# model = MorganBioBertClassification()
# checkpoint = torch.load("./morgan-embed-bio-clinical-bert-ddi/morgan-bioclinicalbert-pathway-cpu.pth.tar")
# model.load_state_dict(checkpoint['state_dict'])
#
#
# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained('./morgan-embed-bio-clinical-bert-ddi')
# os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
#
# # example_smiles = "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C"  # Omeprazole
# # example_target = "bemeprazole is used to manage gastroesophageal reflux disease to prevent stomach ulcers , and to help manage the effects of infection . <n> its effects are covalently bound to the protein noncovalentlytopase enzymes because additional dose - dependent enzymes must be created to replace the ones that bind by pantoprazole."
# example_smiles = "NC(=O)C1=NC=CN=C1"  # Pyrazinamide
# example_target = "ropivacaine is an antihypertensive drug that belongs to the family of catecholamines . <n> it is used for the treatment of hypertension and angina pectoris , as well as for prophylaxis of type 2 diabetes mellitus in patients with chronic obstructive pulmonary disease ( copd).[1 ] it has been shown to exert its local anaesthetic effect by blocking voltage- and protonated sodium channels in peripheral neurons across the plasma membrane of neurons.2 in this study we"
#
# data_loader, drug_ids, drug_names  = BuildDataLoader(smiles1=example_smiles,
#                                                      d1_pathway=example_target,
#                                                     tokenizer=tokenizer,
#                                                     # embed_smiles="BioClinical",
#                                                     embed_smiles="Morgan",
#                                                     similarity="Cosine"
#                                                     )
#
# scores = predict_scores(data_loader, model,
#                         # embed_smiles="BioClinical"
#                         embed_smiles="Morgan"
#                         )
# predictions = [{"drug_id":drug_id, "drug_name":drug_name, "score": score}
#                for drug_id, drug_name, score in zip(drug_ids, drug_names, scores)
#                if score >= 0.5]
# print(predictions)