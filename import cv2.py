# Sample weather data
data = pd.DataFrame({
    'Humidity': [80, 70, 60, 90, 85],
    'Pressure': [1012, 1010, 1008, 1015, 1013],
    'Temperature': [30, 28, 25, 32, 31]
})

# Predicting Temperature
X = data[['Humidity', 'Pressure']]
y = data['Temperature']

# Model
model = LinearRegression()
model.fit(X, y)

# Predict future temperature
future_data = pd.DataFrame({'Humidity': [75], 'Pressure': [1011]})
prediction = model.predict(future_data)
print("Predicted Temperature:", prediction)


