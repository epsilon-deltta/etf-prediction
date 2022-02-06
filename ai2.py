
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense , Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

#from sklearn import preprocessing

#---------------------------------------------------------------------------------
# pip install -U finance-datareader 해야지 사용할수 있는 라이브러리
# https://financedata.github.io/posts/finance-data-reader-users-guide.html  설명서
#import FinanceDataReader as fdr
def make_sequence_dataset(feature, label, window_size):
    feature_list = [] 
    label_list = []

    for i in range(len(feature) - window_size):
      feature_list.append(feature[i:i+window_size])
      label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)


# load dataset

raw_df = pd.read_csv('test4.csv')
raw_df.head()



#raw_df.isnull().sum()
#raw_df.loc[raw_df['Open'].isna()]

#raw_df['Volume']= raw_df[Volume']

#data_normalizer = preprocessing.MinMaxScaler() #데이터를 0-1범위로 수렵시키는 함수
#data_normalized = data_normalizer.fit_transform(data) 

# 정규화 작업

scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low' , 'Close', 'Adj Close', '3MA','5MA','Volume']
scaled_df = scaler.fit_transform(raw_df[scale_cols])
print(type(scaled_df),'\n')
scaled_df = pd.DataFrame(scaled_df , columns=scale_cols)
print(scaled_df)


feature_cols = [ '3MA', '5MA', 'Adj Close']
label_cols = ['Adj Close']

label_df = pd.DataFrame(scaled_df, columns=label_cols)
feature_df = pd.DataFrame(scaled_df , columns = feature_cols)

print(feature_df)
print(label_df)

label_np = label_df.to_numpy()
feature_np = feature_df.to_numpy()

window_size = 40
X,Y = make_sequence_dataset(feature_np, label_np, window_size)
print(X.shape, Y.shape)

# 트레이닝 데이터 / 테스트 데이터 분리
split = -200
x_train = X[0:split]
y_train = Y[0:split]
x_test = X[split:]
y_test = Y[split:]

print(x_train.shape , y_train.shape)
print(x_test.shape, y_test.shape)


# LSTM 모델 구축
print('lhc1 = ' , x_train[0].shape)
model = Sequential() 
model.add( LSTM(128,
             activation = 'tanh',
             input_shape = x_train[0].shape))

model.add(Dense (1, activation = 'linear'))
model.summary()
# 모델 컴파일 및 학습

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5)
model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=10,batch_size=16,callbacks=[early_stop])

pred = model.predict(x_test)
plt.figure(figsize=(12,6))
plt.title('3MA + 5MA + Adj Close, window_size=40')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')

#plt.plot(raw_df['Adj Close'], label='Adj Close', color='b')


plt.show()