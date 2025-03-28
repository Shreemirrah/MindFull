import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from scripts.preprocess_lifestyle import preprocess_lifestyle

# Load and preprocess data
df = pd.read_csv("../data/alzheimers_disease_data.csv")
X, y, scaler = preprocess_lifestyle(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save model and scaler
joblib.dump(model, "../models/lifestyle_model.pkl")
joblib.dump(scaler, "../models/lifestyle_scaler.pkl")
print("Model and scaler saved.")
# Copilot suggested the train-test split and model saving