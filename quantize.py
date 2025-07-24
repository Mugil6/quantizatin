import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing


model = joblib.load("model.joblib")
coef = model.coef_
intercept = model.intercept_


params = {"coef": coef, "intercept": intercept}
joblib.dump(params, "unquant_params.joblib")


scale = 255 / (np.max(np.abs(coef)))
quant_coef = np.round(coef * scale).astype(np.uint8)
quant_intercept = np.round(intercept * scale).astype(np.uint8)

quant_params = {"coef": quant_coef, "intercept": quant_intercept, "scale": scale}
joblib.dump(quant_params, "quant_params.joblib")


class QuantNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


X, y = fetch_california_housing(return_X_y=True)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)


model_nn = QuantNet(X.shape[1])
model_nn.linear.weight.data = torch.tensor((quant_coef / scale).reshape(1, -1), dtype=torch.float32)
model_nn.linear.bias.data = torch.tensor([quant_intercept / scale], dtype=torch.float32)


with torch.no_grad():
    preds = model_nn(X)


from sklearn.metrics import r2_score
print("Quantized Model R² Score:", r2_score(y.numpy(), preds.numpy()))
