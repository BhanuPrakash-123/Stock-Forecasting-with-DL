# %% [markdown]
# <a href="https://colab.research.google.com/github/avns21/Quantitative-Stock-Forecasting-EE798Q/blob/main/Building_LSTM_Model.ipynb" target="_parent">
# <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
df = pd.read_csv('/content/drive/MyDrive/STOCK_INDEX.csv')

# %%
random.seed(10)

# %%
df_fixed = df.interpolate()

# %%
df2 = df_fixed.reset_index()['Close']

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df2).reshape(-1, 1))

# %%
df1.shape

# %%
training_size = int(len(df1) * 0.80)
test_size = len(df1) - training_size

train_data = df1[0:training_size, :]
test_data = df1[training_size:len(df1), :1]


# %%
def create_ds(dataset, step):
    X, Y = [], []
    for i in range(len(dataset) - step - 1):
        X.append(dataset[i:(i + step), 0])
        Y.append(dataset[i + step, 0])
    return np.array(X), np.array(Y)


# %%
look_back = 3
train_X, train_Y = create_ds(train_data, look_back)
test_X, test_Y = create_ds(test_data, look_back)

# Reshape input to [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

# %%
# Build the LSTM Model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=64))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# %%
# Train the model
epochs = 100
batch_size = 50
model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size)

# %%
# Save the model
model.save("lstm_model3.h5")

# %%
from google.colab import files
files.download("lstm_model3.h5")

# %%
# Plot Loss
loss = model.history.history['loss']
plt.plot(loss)

# %%
# Mean Squared Error
mse = model.evaluate(test_X, test_Y, verbose=0)
print("Mean Squared Error (MSE):", mse)

# Directional Accuracy
predictions = model.predict(test_X)
directional_accuracy = np.mean(
    np.sign(predictions[:, 0] - test_Y) ==
    np.sign(predictions[:, 0] - test_Y)
)
print("Directional Accuracy:", directional_accuracy)

# %%
train_predictions = model.predict(train_X)
test_predictions = model.predict(test_X)

# %%
train_predictions = scaler.inverse_transform(train_predictions)
train_Y = scaler.inverse_transform([train_Y])

test_predictions = scaler.inverse_transform(test_predictions)
test_Y = scaler.inverse_transform([test_Y])

# %%
# Plot Full Series vs Predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(train_predictions)
plt.plot(test_predictions)

# %%
# Combine predictions
combined_predictions = np.vstack((train_predictions, test_predictions))

# %%
plt.plot(scaler.inverse_transform(df1))
plt.plot(combined_predictions)

# %%
last_data = df1[-look_back:]
next_two_days = []

for _ in range(2):
    input_data = last_data[-look_back:].reshape(1, look_back, 1)
    predicted_price = model.predict(input_data)
    next_two_days.append(predicted_price[0, 0])
    last_data = np.append(last_data, predicted_price, axis=0)

# %%
predicted_prices = scaler.inverse_transform(
    np.array(next_two_days).reshape(-1, 1)
)
print(predicted_prices.flatten().tolist())

# =======================
# Second CSV Prediction
# =======================

# %%
dfs = pd.read_csv('/content/drive/MyDrive/sample2.csv')

# %%
random.seed(10)

# %%
dfs_fixed = dfs.interpolate().dropna()
dfs2 = dfs_fixed.reset_index()['Close']

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
dfs1 = scaler.fit_transform(np.array(dfs2).reshape(-1, 1))

# %%
last_data = df1[-look_back:]
next_two_days = []

for _ in range(2):
    input_data = last_data[-look_back:].reshape(1, look_back, 1)
    predicted_price = model.predict(input_data)
    next_two_days.append(predicted_price[0, 0])
    last_data = np.append(last_data, predicted_price, axis=0)

# %%
predicted_prices = scaler.inverse_transform(
    np.array(next_two_days).reshape(-1, 1)
)
print(predicted_prices.flatten().tolist())
