# Stock Forecasting with Deep Learning

## Problem Statement
The goal of this project is to build a machine learning model that can predict the closing stock price for the next 2 days, given the stock index information of the past 50 days.

## Dataset
The model was trained on historical stock market data (`Stock_Index.csv`), which contains **2,694** daily entries (representing approximately **10 years** of trading data). The dataset includes numerical stock features such as Open, High, Low, Close, Adj Close, and Volume.

## Procedure Followed
1. **Data Preprocessing**: 
   - Handled missing values (NaNs) by using **linear interpolation** (`infer_objects(copy=False).interpolate()`) to fill in the gaps mathematically without breaking the price trends.
   - Scaled all stock prices down to a range between 0 and 1 using **MinMax Scaling** so the neural network could train efficiently.
2. **Model Building**: 
   - Built a Sequential Neural Network in TensorFlow/Keras using a two-layer **LSTM (Long Short-Term Memory)** architecture.
   - **Architecture Details:**
     - Layer 1: LSTM with 64 units (`return_sequences=True`)
     - Layer 2: LSTM with 64 units
     - Layer 3: Dense output layer with 1 unit
   - The model uses a `look_back` window of 3 days to learn temporal dependencies from the past data.
3. **Training**: 
   - Split the data sequentially into an 80% Training set and a 20% Testing set.
   - Trained the model for 100 epochs using the Adam optimizer to minimize the Mean Squared Error (MSE).
4. **Prediction**: 
   - Used an **autoregressive** forecasting loop to iteratively predict the next 2 unseen days. The model predicts Day 1, appends it to its own history, and then uses that newly updated history to predict Day 2.

## Setup & Execution

### Required Libraries
Make sure your Python environment has the following libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

### How to Run
1. Ensure the saved model (`trained_lstm_model.h5`), the input data (`sample_input.csv`), and the target data (`sample_close.txt`) are all in the same folder as the evaluation script.
2. Run the evaluation script in your terminal:
   ```bash
   python "EvaluatingMSE&DE.py"
   ```
3. The script will automatically output the final **Mean Square Error** and **Directional Accuracy** for the test sample.
