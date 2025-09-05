import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')


def predict_func(dataset):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    #taking our input testing dataset as df
    df = dataset
    #using linear interpolation to fill na values in df and then using dropna() to drop any further na values if present
    df_fixed = df.interpolate()
    df_fixed = df_fixed.dropna()
    #extracting close column from df
    df2 = df_fixed.reset_index()['Close']
    #using scaler function to scale our input testing dataset into range of (0,1)
    scaler = MinMaxScaler(feature_range = (0,1))
    dfs = scaler.fit_transform(np.array(df2).reshape(-1,1))
    #obtaining shape of our df
    dfs.shape
    #loading our prestored trained lstm model for predicting next 2 values of our dataset
    model = load_model("lstm_model3.h5")
    #as we need 2 future values so setting back_track variable as 2+1
    back_track = 3
   
    last_df = dfs[-back_track:]
    #creating an empty list to store predicted vaues
    future_2days_values = []
    #running a for loop to predict future values
    for _ in range(back_track-1):
      input_df = last_df[-back_track:].reshape(1, back_track, 1)
      predicted_value = model.predict(input_df)
      future_2days_values.append(predicted_value[0, 0])
      last_df = np.append(last_df, predicted_value, axis=0)
    #inverse transforming values to original scale
    predicted_values = scaler.inverse_transform(np.array(future_2days_values).reshape(-1, 1))
    #returning stored predicted values
    return predicted_values.flatten().tolist()

if __name__== "__main__":
    evaluate()   