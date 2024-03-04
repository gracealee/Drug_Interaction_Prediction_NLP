import torch
from src.model import MorganBioBertClassification, predict_scores
from src.data_ddi import BuildDataLoader
from transformers import AutoTokenizer

import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel, validator
from redis import asyncio


logger = logging.getLogger(__name__)
LOCAL_REDIS_URL = "redis://redis:6379"
app = FastAPI()

# Load model for pathway + SMILES prediction
model1 = MorganBioBertClassification()
checkpoint1 = torch.load("./morgan-embed-bio-clinical-bert-ddi/morgan-bioclinicalbert-pathway-cpu.pth.tar")
model1.load_state_dict(checkpoint1['state_dict'])

# Load model for SMILES only
model2 = MorganBioBertClassification()
checkpoint2 = torch.load("./morgan-embed-bio-clinical-bert-ddi/morgan300-bioclinicalbert-cpu.pth.tar")
model2.load_state_dict(checkpoint2['state_dict'])

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('./morgan-embed-bio-clinical-bert-ddi')
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

@app.on_event("startup")
async def startup():
    try:
        REDIS_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
        logger.debug(REDIS_URL)
        redis = asyncio.from_url(REDIS_URL, encoding="utf8", decode_responses=True)
        FastAPICache.init(RedisBackend(redis), prefix="mlapi-cache")
    except Exception as e:
        print("Redis not available:", str(e))


class PredictionRequest(BaseModel):
    smiles: str | None = "COC1=CC2=C(C=C1)N=C(N2)S(=O)CC1=NC=C(C)C(OC)=C1C"
    target_pathway: str | None = "Not Available" # Impute when no target & pathway provided


class Prediction(BaseModel):
    drug_id: str
    drug_name: str
    score: float


class PredictionResponse(BaseModel):
    predictions: list[Prediction]


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
@cache(expire=5000)
async def predict(input_data: PredictionRequest):
    # Build Data Loader for all approved drugs in the database
    pathway = str(input_data.target_pathway)
    if len(pathway.split()) > 10:
        # Use MorganBioClinicalBert Model when there is pathway data input
        # Data Loader
        data_loader, drug_ids, drug_names  = BuildDataLoader(smiles1=input_data.smiles,
                                                             d1_pathway=input_data.target_pathway,
                                                             tokenizer=tokenizer,
                                                             embed_smiles="Morgan",
                                                             similarity="Cosine"
                                                             )

        # Predict
        scores = predict_scores(data_loader, model1, embed_smiles="Morgan")

    else:
        # Use BioClinicalBert Model when there is no pathway data input
        # Data Loader
        data_loader, drug_ids, drug_names  = BuildDataLoader(smiles1=input_data.smiles,
                                                             d1_pathway=input_data.target_pathway,
                                                             tokenizer=tokenizer,
                                                             embed_smiles="BioClinical",
                                                             )

        # Predict
        scores = predict_scores(data_loader, model2, embed_smiles="BioClinical")

    # Return predictions for all drugs with >= 0.5 probability of having interaction with the input drug
    predictions = [{"drug_id":drug_id, "drug_name":drug_name, "score": score}
                   for drug_id, drug_name, score in zip(drug_ids, drug_names, scores)
                   if score >= 0.5]

    return {"predictions": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
