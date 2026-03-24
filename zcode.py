import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load merged dataset
data = pd.read_csv("merged_dataset.csv")

print("Dataset Loaded")
print(data.head())

# Select numeric columns only
data = data.select_dtypes(include=['int64', 'float64'])

# Remove missing values
data = data.dropna()

# Features and Target
X = data.drop(columns=[data.columns[-1]])
y = data[data.columns[-1]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
error = mean_absolute_error(y_test, predictions)

print("Model trained successfully!")
print("MAE:", error)