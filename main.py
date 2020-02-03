# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:23:12 2019

@author: cyuan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation


warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 200)
#df=pd.read_csv('C:\\Users\\cyuan\\Desktop\\ML_summary\\LSTM\\passenger.csv',header=None)
#df.columns=['time','passengers']
#
#df.set_index('time',inplace=True)
#print(df.head(12))

#df['passengers'].plot()
#plt.show()

class airline_predict:
    def __init__(self, filename,sequence_length=10,split=0.8):
        self.filename=filename
        self.sequence_length=sequence_length
        self.split=split
        
    def load_data(self):
        df=pd.read_csv(self.filename,sep=',',usecols=[1],header=None)
        data_all=np.array(df).astype('float')
        print(data_all.shape)
        
        mms=MinMaxScaler()
        data_all=mms.fit_transform(data_all)
        print(data_all.shape)
        
        #LSTM的输入是需要三维数据的 (133,11,1)
        data=[]
        for i in range(len(data_all)-self.sequence_length-1):
            data.append(  data_all[i:i+self.sequence_length+1])
            
        reshaped_data=np.array(data).astype('float')
        #print(reshaped_data)
    
        np.random.shuffle(reshaped_data)
        
        x=reshaped_data[:,:-1]
        y=reshaped_data[:,-1]
        
        split_b=int(reshaped_data.shape[0]*self.split)
        train_x=x[:split_b]
        train_y=y[:split_b]
        
        test_x=x[split_b:]
        test_y=y[split_b:]

        return train_x,train_y,test_x,test_y
        
    def bulid_model(self):
        model=Sequential()
        model.add( LSTM(input_dim=1, units=50, activation='relu', return_sequences=True) )
        
        print('model layers:', model.layers)
        
        model.add(LSTM(input_dim=100,units=1,return_sequences=False))
        model.add(Dense(units=1))
        model.add(Activation('linear'))
        
        model.compile(loss='mse',optimizer='rmsprop')
        return model
    
    def train_model(self,train_x,train_y,test_x,test_y):
        model=self.bulid_model()
        
        try:
            model.fit(train_x,train_y,batch_size=512,nb_epoch=100,validation_split=0.1)
            predict=model.predict(test_x)
            predict=np.reshape(predict,(predict.size,))
            test_y=np.reshape(test_y,(test_y.size,))
            
        except KeyboardInterrupt:
            print('predict',predict)
            print('test_y', test_y)
            
        print('after predict\n',predict)
        print('test_y', test_y)
        
        
        try:
            fig1=plt.figure(1)
            plt.plot(predict,'r')
            plt.plot(test_y,'g-')
            plt.title('the picture is drawed by using standard data')
            plt.legend('predict','true')
            
        except Exception as e:
            
            print(e)
            
        return predict, test_y
            
            
    
    
filename='C:\\Users\\cyuan\\Desktop\\ML_summary\\LSTM\\passenger.csv'
airline=airline_predict(filename)
train_x,train_y,test_x,test_y=airline.load_data()

predict_y, test_y=airline.train_model(train_x,train_y,test_x,test_y)

mms=MinMaxScaler()
predict_y=mms.inverse.transform([[i] for i in predict_y])

test_y=mms.inverse.transform([[i] for i in test_y])

fig21=plt.figure(2)
plt.plot(predict_y,'g',label='prediction')
plt.plot(test_y,'r-',label='true')
plt.title('this pic ture is drawed by standard data') 
plt.legend(['predict','true'])
plt.show()   
    











