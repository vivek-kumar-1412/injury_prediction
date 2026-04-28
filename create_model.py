"""
One-time script to create and save the pre-trained injury prediction model.
Run this once, then delete or ignore it. The app only uses the saved .pkl file.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic training data (1500 samples, 10 features)
n_samples = 1500

# Feature distributions designed to mimic realistic football player data
age = np.random.uniform(18, 38, n_samples)                    # years
bmi = np.random.uniform(19, 30, n_samples)                    # kg/m²
total_distance = np.random.uniform(5, 14, n_samples)          # km per match
sprint_count = np.random.uniform(10, 60, n_samples)           # sprints per match
acceleration_load = np.random.uniform(50, 300, n_samples)     # arbitrary units
acwr = np.random.uniform(0.5, 2.0, n_samples)                 # acute:chronic workload ratio
yo_yo_score = np.random.uniform(14, 23, n_samples)            # yo-yo IR1 level
jump_height = np.random.uniform(25, 55, n_samples)            # cm
previous_injuries = np.random.randint(0, 8, n_samples)        # count
minutes_played = np.random.uniform(0, 90, n_samples)          # minutes per match

X = np.column_stack([
    age, bmi, total_distance, sprint_count, acceleration_load,
    acwr, yo_yo_score, jump_height, previous_injuries, minutes_played
])

# Create realistic injury risk labels based on domain knowledge
injury_score = (
    0.15 * (age - 18) / 20 +                    # older = higher risk
    0.10 * (bmi - 22) / 8 +                      # high BMI = higher risk
    0.05 * (total_distance - 9) / 5 +            # extreme distance = risk
    0.08 * (sprint_count - 30) / 30 +            # more sprints = risk
    0.12 * (acceleration_load - 150) / 150 +     # high load = risk
    0.20 * (acwr - 1.0) / 1.0 +                  # ACWR > 1.3 is danger zone
    -0.08 * (yo_yo_score - 18) / 9 +             # better fitness = lower risk
    -0.05 * (jump_height - 40) / 30 +            # better power = lower risk
    0.15 * previous_injuries / 7 +               # injury history = risk
    0.10 * (minutes_played - 45) / 45 +          # overplay = risk
    np.random.normal(0, 0.08, n_samples)         # noise
)

y = (injury_score > 0.22).astype(int)  # 1 = High Risk, 0 = Low Risk

print(f"Dataset: {n_samples} samples, {X.shape[1]} features")
print(f"Class distribution: Low Risk={np.sum(y==0)}, High Risk={np.sum(y==1)}")
print(f"Injury rate: {np.mean(y)*100:.1f}%")

# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/injury_model.pkl")
print("\nModel saved to models/injury_model.pkl")

# Quick validation
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X)
print(f"\nTraining Accuracy: {accuracy_score(y, y_pred)*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Low Risk", "High Risk"]))

# Feature importance
feature_names = [
    "Age", "BMI", "Total Distance", "Sprint Count", "Acceleration Load",
    "ACWR", "Yo-Yo Score", "Jump Height", "Previous Injuries", "Minutes Played"
]
importances = model.feature_importances_
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")
