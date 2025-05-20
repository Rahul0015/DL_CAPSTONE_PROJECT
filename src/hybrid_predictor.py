import torch
import pickle
import numpy as np
from src.ml_model import MLPClassifier
from src.symbolic_rules import SymbolicCountermeasureSuggester, KNOWLEDGE_BASE


def load_tfidf_and_model(tfidf_path, model_path, input_dim):
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return tfidf, model

def hybrid_predict(text, tfidf, model, kb):
    vec = tfidf.transform([text]).toarray()
    input_tensor = torch.tensor(vec, dtype=torch.float32)
    prob = model(input_tensor).item()
    threat_type_id = "IED_urban"  # Mocked logic for demo
    forecast = {"threat_type_id": threat_type_id, "prob": prob}
    suggester = SymbolicCountermeasureSuggester(kb)
    return forecast, suggester.suggest_countermeasures(forecast)
