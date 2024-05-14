import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

def name_get(stock,start_date,end_date):
# def name_get():
    # global df
    start=start_date
    end=end_date
    stock_symbol = stock
    # start='2010-01-01'
    # end='2019-12-31'
    # stock_symbol = 'AAPL'
    df =yf.download(stock_symbol,start,end)
#     return df

#     # close graph
# def Close(df):
    close_path = 'static/GRAPH_IMG/close_graph.png'
    plt.figure(figsize=(16,8))
    plt.title('close price histroy')
    plt.plot(df['Close'])
    plt.xlabel('Data',fontsize=18)
    plt.ylabel('CLose price USD ($)',fontsize=18)
    plt.savefig(close_path)
#     return close_path

# def MA100(df):
    # st.subheader('Closing Price vs Time Chart with 100MA')
    ma100_path = 'static/GRAPH_IMG/ma100_graph.png'
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'r')
    plt.plot(df.Close,'g')
    plt.savefig(ma100_path)
#     return ma100_path
#     # st.pyplot(fig)
    
# def MA200(df):
    # st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
    ma200_path = 'static/GRAPH_IMG/ma200_graph.png'
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'r')
    plt.plot(ma200,'b')
    plt.plot(df.Close,'g')
    plt.savefig(ma200_path)
    # st.pyplot(fig)
    # st.title('stock Trend Prediction')
#     return ma200_path

    
#         # st.subheader('Prediction')
# def prediction(df):
    data = df.filter(['Close'])
    dataset =  data.values
    trainig_data_len = math.ceil(len(dataset) * .8 )
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    
    train_data = scaled_data[0:trainig_data_len , :]
    
    x_train= []
    y_train= []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])
    
    x_train ,y_train = np.array(x_train), np.array(y_train)
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = load_model('keras_model.h5')
    
    test_data = scaled_data[trainig_data_len - 60: , :]
    
    #create data sets x_test and y_test
    x_test = []
    y_test = dataset[trainig_data_len: , :]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    
    
    x_test = np.array(x_test)
    
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    train = data[:trainig_data_len]
    valid = data[trainig_data_len:]
    valid['predictions'] = predictions

    prediction_path = 'static/GRAPH_IMG/prediction_graph.png'
    fig3=plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Data', fontsize = 18)
    plt.ylabel('Close Price USD ($)', fontsize = 18)
    plt.plot(train['Close'],label='close')
    plt.plot(valid[['predictions']])
    plt.legend(['train', 'predictions'], loc = 'lower right')
    plt.savefig(prediction_path)
    # st.pyplot(fig3)
#     return prediction_path
    
#     # st.subheader('Prediction vs Original Price')
# def prediction_orignal(df):
    

    prediction_orignal_path = 'static/GRAPH_IMG/prediction_orignal_graph.png'
    fig2=plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Data', fontsize = 18)
    plt.ylabel('Close Price USD ($)', fontsize = 18)
    plt.plot(train['Close'],label='close')
    plt.plot(valid[['Close', 'predictions']])
    plt.legend(['train', 'Val', 'predictions'], loc = 'lower right')
    plt.savefig(prediction_orignal_path)

    # A=[close_path,ma100_path,ma200_path,prediction_path,prediction_orignal_path]

    # return 
    # st.pyplot(fig2)

# name_get()
# a=name_get('AAPL','2010-01-01','2019-12-31')
# print(a)