import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
from sklearn.model_selection import GridSearchCV, LeaveOneOut

# ✅ Load the dataset
df = pd.read_csv("symptom_disease_expanded.csv")

# ✅ Extract unique symptoms
unique_symptoms = set()
for col in df.columns[1:]:  # Ignore "Disease" column
    unique_symptoms.update(df[col].dropna().unique())

full_feature_list = sorted(list(unique_symptoms))  # Convert to sorted list

# ✅ Save the corrected feature list
with open("features.json", "w") as f:
    json.dump(full_feature_list, f)

print(f"✅ Updated features.json with {len(full_feature_list)} symptoms.")

# ✅ Create binary matrix for diseases & symptoms
disease_symptom_df = pd.DataFrame(0, index=df["Disease"].unique(), columns=full_feature_list)

# ✅ Fill binary matrix
for _, row in df.iterrows():
    disease = row["Disease"]
    for symptom in row[1:]:
        if pd.notna(symptom) and symptom in full_feature_list:
            disease_symptom_df.at[disease, symptom] = 1

# ✅ Features (X) and Target (y)
X = disease_symptom_df
y = pd.Series(disease_symptom_df.index)  # Convert index to Pandas Series

# ✅ Print Disease Count
print("✅ Disease Count in Training Data:")
print(y.value_counts())

# ✅ Split Data (if possible)
if len(y) > 1:  # Ensure enough data for train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    X_train, y_train = X, y  # Use all data for training
    X_test, y_test = X, y  # Duplicate for compatibility

# ✅ Define Hyperparameter Grid
param_grid = {
    "n_estimators": [100, 500],
    "max_depth": [10, None],
    "min_samples_split": [2, 5],
}

# ✅ Use Leave-One-Out (LOO) Cross-Validation Instead of `cv=1`
cv_strategy = LeaveOneOut()

# ✅ Perform Grid Search with LOO Cross-Validation
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv_strategy, n_jobs=-1)

grid_search.fit(X_train, y_train)

# ✅ Save Best Model
best_model = grid_search.best_estimator_
joblib.dump(best_model, "symptom_model.pkl")

print("✅ Model training completed with best hyperparameters:", grid_search.best_params_)
print("✅ Model trained successfully and saved as symptom_model.pkl!")
