# Codsoft_task2
Perfect 🎬 — let’s build a Movie Rating Prediction model step by step using Python & regression techniques.

Since we don’t have the dataset in this environment, I’ll assume you have a dataset (CSV) with columns like:

Title – Movie title

Genre – Genre of the movie

Director – Director’s name

Actors – Lead actors

Budget – Budget of the movie

Revenue – Box office revenue

Runtime – Movie duration (minutes)

Rating – IMDb rating (target variable)



---

📝 Step-by-Step Movie Rating Prediction (Regression)

1. Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


---

2. Load Dataset

# Load dataset (replace with your dataset path)
data = pd.read_csv("movies.csv")

print(data.head())
print(data.info())


---

3. Data Preprocessing

Drop Title (not useful for prediction).

Handle missing values (e.g., fill runtime or budget).

Encode categorical features (Genre, Director, Actors).

Scale numerical features.


# Drop unused columns
data = data.drop(["Title"], axis=1)

# Fill missing numeric values with median
for col in ["Budget", "Revenue", "Runtime"]:
    data[col].fillna(data[col].median(), inplace=True)

# Fill missing categorical values with mode
for col in ["Genre", "Director", "Actors"]:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Encode categorical variables
le = LabelEncoder()
data["Genre"] = le.fit_transform(data["Genre"])
data["Director"] = le.fit_transform(data["Director"])
data["Actors"] = le.fit_transform(data["Actors"])

print(data.head())


---

4. Split Features & Target

# Features (all except Rating) and target (Rating)
X = data.drop("Rating", axis=1)
y = data["Rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


---

5. Train Regression Model

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


---

6. Evaluate Model

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))


---

7. Example Visualization

# Plot actual vs predicted ratings
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()

# Feature importance (coefficients)
coef_df = pd.DataFrame(model.coef_, X.columns, columns=["Coefficient"])
print(coef_df)


---

✅ End Result:
This will give you a regression model that predicts a movie’s rating using features like genre, director, actors, budget, revenue, and runtime.

🔹 Typical performance can be improved by:

Trying Random Forest Regressor / XGBoost.

Doing One-Hot Encoding for categorical features instead of Label Encoding.

Performing feature engineering (e.g., actor popularity, director success rate).



---

👉 Do you want me to extend this (like we did for Titanic) and also show a Random Forest Regression model for better prediction accuracy?

