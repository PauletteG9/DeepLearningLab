import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

def generate_sine_data(n_samples=1000, seq_len=50):
  x = np.linspace(0, 100, n_samples)
  y = np.sin(x)
  X, Y = [], []
  for i in range(len(y) - seq_len):
    X.append(y[i:i+seq_len])
    Y.append(y[i+seq_len])
  return np.array(X)[..., np.newaxis], np.array(Y)

X_lstm, y_lstm = generate_sine_data()
split = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:split], X_lstm[split:]
y_train, y_test = y_lstm[:split], y_lstm[split:]

lstm_model = Sequential([
  layers.LSTM(64, input_shape=(50, 1), return_sequences=True, name="LSTM_Layer1"),
  layers.LSTM(32, return_sequences=False, name="LSTM_Layer2"),

  layers.Dense(16, activation='relu', name="Dense_Hidden"),
  layers.Dense(1, name="Output"),       
], name="LSTM_Sequential_Model")

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
lstm_model.summary()

history = lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
predictions = lstm_model.predict(X_test, verbose=0).flatten()

plt.figure(figsize=(10, 4))
plt.plot(y_test[:100], label='Actual', color='blue')
plt.plot(predictions[:100], label='Predicted', color='red', linestyle='--')
plt.title("LSTM: Sine Wave Prediction")
plt.xlabel("Time Step"); plt.ylabel("Value")
plt.legend(); plt.tight_layout()
plt.savefig("lstm_predictions.png", dpi=100)
plt.show()
print("Saved: lstm_predictions.png")
