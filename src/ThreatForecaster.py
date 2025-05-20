import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class ThreatForecaster:
    def __init__(self, model_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()  # Set to evaluation mode

        # Example: self.scaler = joblib.load('models/scaler.pkl') if you scale inputs
        self.scaler = None  # Optional, set to a trained scaler

    def preprocess_input(self, spatio_temporal_data: pd.DataFrame):
        """
        Accepts a DataFrame and returns a tensor for prediction.
        """
        selected_features = ['year', 'month', 'day', 'latitude', 'longitude', 'nkill', 'nwound']
        data = spatio_temporal_data[selected_features].fillna(0)

        if self.scaler:
            data = self.scaler.transform(data)
        data = torch.tensor(data.values, dtype=torch.float32).to(self.device)
        return data

    def predict(self, input_tensor: torch.Tensor):
        """
        Predicts threat scores.
        """
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.cpu().numpy()

# --- Usage Example ---
# import pandas as pd
# df = pd.read_csv("data/cleaned_india_terrorism.csv")
# forecaster = ThreatForecaster("models/torch_threat_model.pt")
# input_tensor = forecaster.preprocess_input(df.head(10))
# predictions = forecaster.predict(input_tensor)
# print(predictions)

