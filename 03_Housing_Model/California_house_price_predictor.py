import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 1. Load REAL data (Houses in California)
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target # This is the price in $100,000s

# 2. Features (X) and Target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the "Random Forest"
# This is a 'Forest' of decision trees that work together
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Check Accuracy
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

print(f"--- Real World Model Evaluation ---")
print(f"Accuracy (R2 Score): {accuracy:.4f}") 
# Expect this to be around 0.80 (80% accuracy!)

# Get the 'importance' of each feature
importances = model.feature_importances_
feature_names = data.feature_names

# Create a simple chart
plt.barh(feature_names, importances)
plt.xlabel("Importance Score")
plt.title("What drives house prices in California?")
plt.show()
