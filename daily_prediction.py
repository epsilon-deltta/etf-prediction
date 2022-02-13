import FinanceDataReader as fdr 
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense , Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
# from sklearn import preprocessing
# pip install -U finance-datareader
# import FinanceDataReader as fdr
def make_sequence_dataset(feature, label, window_size):
    feature_list = [] 
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i+window_size+1]) # added label
        label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)

def survive_open(x,feature_cols,label_cols):
    
    index = [i for i,item in enumerate(feature_cols) if (item not in label_cols)]
    for i in range(len(x)):
        x[i][40][index] = np.zeros(len(index))
    return x


feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Change']
label_cols = ['High','Low']
def train_save(df,recent_date,save_dir='./models'):
    
    raw_df = df
    scaler = MinMaxScaler()
    scale_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Change']
    scaled_df = scaler.fit_transform(raw_df[scale_cols]) # scaler.inverse_transform(X)
    scaled_df = pd.DataFrame(scaled_df , columns=scale_cols)

    
    label_np   = pd.DataFrame(scaled_df, columns=label_cols).to_numpy()
    feature_np = pd.DataFrame(scaled_df, columns = feature_cols).to_numpy()

    window_size = 40
    x,y = make_sequence_dataset(feature_np, label_np, window_size)
    x = survive_open(x,feature_cols,label_cols)

    # train data 와 test data로 구분(7:3 비율) X,y : DataFrame
    
    from sklearn.model_selection import train_test_split
    xtr,xval,ytr,yval = train_test_split(x,y,train_size=0.7,shuffle=True) #train:val = 7:3

    model = Sequential()
    model.add( LSTM(128,
                 activation = 'tanh',
                 input_shape = xtr[0].shape)) #41,8

    model.add(Dense (64, activation = 'linear'))
    model.add(Dense (2, activation = 'linear'))
    # model.summary()
    # from livelossplot import PlotLossesKeras # pip install livelossplot

    model.compile(loss='mse', optimizer='adam', metrics=['mae','acc'])
    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    # model.fit(xtr,ytr, validation_data=(xval,yval), epochs=200,batch_size=32,callbacks=[early_stop,PlotLossesKeras()])
    model.fit(xtr,ytr, validation_data=(xval,yval), epochs=200,batch_size=32,callbacks=[early_stop])
    
    # save
    
    import os
    model_path = os.path.join(save_dir,f'{recent_date}_model.h5')
    model.save(model_path)
    print(f'saved to {model_path}')
    
    import joblib
    scaler_path = os.path.join(save_dir,f'{recent_date}_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f'saved to {scaler_path}')
    
    return 0

def inverse_scaling(pre,scaler,feature_cols=feature_cols,label_cols=label_cols):
    index = []
    zeros = np.zeros( (pre.shape[0],len(feature_cols)) )
    for item in label_cols:
        index.append(feature_cols.index(item) )
    zeros[:,index] = pre
    inverse_pred = scaler.inverse_transform(zeros)[:,index]
    return inverse_pred

import tensorflow as tf
import joblib
import os
import pandas as pd
def get_predate(df,recent_day):
    
    pre_idx = df.reset_index()[df.index == pd.to_datetime(today)].index.values[0]
    pre_date = df.iloc[pre_idx-1:pre_idx].index[0]
    return pre_date.date()

def load_model(df,recent_date,model_dir='./models'):
    
    pre_date = get_predate(df,recent_date)
    
    model_path  = os.path.join(model_dir,f'{str(pre_date)}_model.h5') 
    scaler_path = os.path.join(model_dir,f'{str(pre_date)}_scaler.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) ):
        train_save(df,pre_date)
    
    model  = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    
        
    return model, scaler

def predict(model,scaler,df,today):
    
    x = df.loc[:pd.to_datetime(today)][-41:]
    x = scaler.transform(x)
    x = survive_open(np.array([x]),feature_cols,label_cols)
    y = model(x)
    is_y = inverse_scaling(y,scaler) # inverse-scaled  
    high,low = is_y[0][0],is_y[0][1]
    return high,low
    

from datetime import datetime
import FinanceDataReader as fdr 
import matplotlib.pyplot as plt

today = datetime.today().date()

stock_name = "TQQQ"
df = fdr.DataReader(stock_name)
recent_date = df[-1:].index[0].date()

if __name__ == '__main__' :

    import argparse
    exam_code = '''
    e.g)  
    python daily_prediction.py
    python daily_prediction.py -date 2021-02-11
    '''
    parser = argparse.ArgumentParser("Train Mask R-CNN model",epilog=exam_code)   
    # setting
    parser.add_argument('-d'  ,'--date'      ,default=None,  help='e.g., 2022-01-21')

    args = parser.parse_args()



    from datetime import datetime           
    if args.date is not None:
        today = datetime.strptime(args.date, "%Y-%m-%d").date()
        recent_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    
    if today != recent_date: # not open
        train_save(df,recent_date)

    else: # today == recent_day # open (11:30~12:00)
        # open_price = df[today_date].Open

        # pre_date     = get_predate(df,recent_date)
        # model,scaler = load_model(pre_date)
        model,scaler = load_model(df,recent_date)

        # model,scaler = model_load(f"{str(today)}_model.onnx")

        high,low = predict(model,scaler,df,today)

        pre = pd.DataFrame({'high':[high],'low':[low]},columns=['high','low'],index=[today]);print(pre)
        pre.to_csv(f'{today}_{stock_name}.csv')