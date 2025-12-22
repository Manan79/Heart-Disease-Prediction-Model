import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the exported model and scaler
loaded_rf_model = joblib.load('heart_disease_rf_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

print("Model and scaler loaded successfully!")


random_data = pd.DataFrame({
    'gender': np.random.choice([0, 1], size=1),
    'age': np.random.uniform(18, 90, size=1),
    'hypertension': np.random.choice([0, 1], size=1),
    'smoking_history': np.random.randint(0, 5, size=1), # Assuming 0-4 categories
    'bmi': np.random.uniform(15, 40, size=1),
    'HbA1c_level': np.random.uniform(4.0, 9.0, size=1),
    'blood_glucose_level': np.random.uniform(70, 200, size=1),
    'diabetes': np.random.choice([0, 1], size=1),
    'total_cholesterol': np.random.uniform(150, 300, size=1),
    'family_history': np.random.choice([0, 1], size=1),
    'physical_activity_level': np.random.randint(0, 3, size=1), # 0, 1, 2
    'alcohol_intake': np.random.randint(0, 3, size=1) # 0, 1, 2
})

print(random_data)

scaled_random_data = loaded_scaler.transform(random_data.values)

# Make predictions with the loaded model
prediction = loaded_rf_model.predict(scaled_random_data)
prediction_proba = loaded_rf_model.predict_proba(scaled_random_data)

print(f"\nPrediction for the random input: {prediction[0]} (0: No Heart Disease, 1: Heart Disease)")
print(f"Prediction probabilities: {prediction_proba[0]} (Probability of No Heart Disease, Probability of Heart Disease)")

# Visualize feature importance
# feature_importances = loaded_rf_model.feature_importances_
# features = random_data.columns
# plt.figure(figsize=(10, 6))
# plt.barh(features, feature_importances, color='skyblue')
# plt.xlabel('Importance Score')
# plt.title('Feature Importance from Random Forest Model')
# plt.show()
