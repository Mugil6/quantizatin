
import joblib
from sklearn.datasets import fetch_california_housing

model = joblib.load("model.joblib")
X, y = fetch_california_housing(return_X_y=True)

predictions = model.predict(X[:5])
print("Predictions on test data:", predictions)
