import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv("Sport car price.csv")

# Remove commas from the Price column
data['Price (in USD)'] = data['Price (in USD)'].str.replace(',', '')

# Convert the Price column to float
data['Price (in USD)'] = data['Price (in USD)'].astype(float)


# Preprocess the data
data = data.drop_duplicates()
data = data.dropna()
X = data[['Car Make', 'Car Model', 'Year', 'Engine Size (L)', 'Horsepower', 'Torque (lb-ft)', '0-60 MPH Time (seconds)']]
y = data['Price (in USD)']
encoder = OneHotEncoder()
X = encoder.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
accuracy = 1 - mae / y_test.mean()
print('Accuracy:', accuracy)

