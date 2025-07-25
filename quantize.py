import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import os


model = joblib.load("model.joblib")


coef = model.coef_
intercept = model.intercept_


unquant_params = {"coef": coef, "intercept": intercept}
joblib.dump(unquant_params, "unquant_params.joblib")


quant_coef = coef.astype(np.float32)
quant_intercept = np.array(intercept, dtype=np.float32)

quant_params = {"coef": quant_coef, "intercept": quant_intercept}
joblib.dump(quant_params, "quant_params.joblib")

class SimpleRegressor(nn.Module):
    def __init__(self, in_features):
        super(SimpleRegressor, self).__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

model_torch = SimpleRegressor(in_features=coef.shape[0])


dequant_coef = quant_coef.astype(np.float32)
dequant_intercept = float(quant_intercept.astype(np.float32))


with torch.no_grad():
    model_torch.linear.weight.copy_(torch.tensor(dequant_coef.reshape(1, -1), dtype=torch.float32))
    model_torch.linear.bias.copy_(torch.tensor([dequant_intercept], dtype=torch.float32))

torch.save(model_torch.state_dict(), "quantized_model_float32.pth")


data = fetch_california_housing()
X, y = data.data, data.target


y_pred_original = model.predict(X)
r2_original = r2_score(y, y_pred_original)


batch_size = 512
y_preds = []
with torch.no_grad():
    for i in range(0, X.shape[0], batch_size):
        batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32)
        preds = model_torch(batch).numpy().flatten()
        y_preds.append(preds)

y_pred_torch = np.concatenate(y_preds)
r2_quantized = r2_score(y, y_pred_torch)


unquant_size = os.path.getsize("unquant_params.joblib") / 1024
quant_size = os.path.getsize("quant_params.joblib") / 1024

print(f"Original Sklearn Model R² Score: {r2_original:.4f}")
print(f"Float32 Quantized Model R² Score: {r2_quantized:.4f}")
print(f"Model size before quantization: {unquant_size:.2f} KB")
print(f"Model size after quantization:  {quant_size:.2f} KB")
