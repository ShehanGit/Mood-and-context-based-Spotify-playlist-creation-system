import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

# Step 1: Load the enhanced dataset
df = pd.read_csv("enhanced_dataset.csv")

# Step 2: Preprocess the dataset
# Handle missing values (if any)
df = df.dropna()

# Define feature columns
numerical_features = ["valence", "energy", "danceability", "tempo", "acousticness", "instrumentalness", "popularity", "temperature"]
categorical_features = ["mood", "age_group", "activity", "rain"]
target = "suitability"

# Encode categorical variables
df_encoded = df[numerical_features + [target]].copy()
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    # Save the encoder for future use
    joblib.dump(le, f"{col}_encoder.pkl")

# Combine all features
X = df_encoded[numerical_features + categorical_features]
y = df_encoded[target]

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the XGBoost model (Regression)
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
}).sort_values(by="importance", ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Step 6: Save the model
joblib.dump(xgb_model, "xgb_model.pkl")
print("\nModel saved as 'xgb_model.pkl'")

# Optional: Example prediction for a new situation
# Example input: song features + context
new_data = pd.DataFrame({
    "valence": [0.75],
    "energy": [0.80],
    "danceability": [0.65],
    "tempo": [0.50],
    "acousticness": [0.10],
    "instrumentalness": [0.00],
    "popularity": [80],
    "temperature": [25.0],
    "mood": [LabelEncoder().fit_transform(["happy"])[0]],  # Load saved encoder for real use
    "age_group": [LabelEncoder().fit_transform(["young"])[0]],
    "activity": [LabelEncoder().fit_transform(["workout"])[0]],
    "rain": [LabelEncoder().fit_transform(["sunny"])[0]]
})

suitability_score = xgb_model.predict(new_data)
print(f"\nPredicted Suitability Score for Example Input: {suitability_score[0]:.4f}")

# For Classification (Optional)
# Uncomment the following to treat suitability as a classification task
"""
# Bin suitability into categories (e.g., low, medium, high)
bins = [0, 0.33, 0.66, 1.0]
labels = ["low", "medium", "high"]
y_binned = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

# Train XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(
    objective="multi:softmax",
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_classifier.fit(X_train, y_train)

# Evaluate
y_pred = xgb_classifier.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save classifier
joblib.dump(xgb_classifier, "xgb_classifier.pkl")
"""