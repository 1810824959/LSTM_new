import numpy as np
from keras.layers import LSTM, Dense,Input,Dropout
from keras.models import Sequential,Model
from keras.models import load_model
from load_data import load_data
from pic_draw import draw_line


timesteps=5
feature=2


# flag=0 填入train
# flag=1 填入test (default flag=0)
def make_timesteps(array,flag=0):
    for index in range(0,len(array)-timesteps,1):
        x=array[index:index+timesteps]
        y=array[index+timesteps]
        if flag==0:
            data_X_train.append(x)
            data_Y_train.append(y)
        else:
            data_X_test.append(x)
            data_Y_test.append(y)


data_X_train=[]
data_Y_train=[]
data_X_test=[]
data_Y_test=[]

# 载入数据
train,test=load_data()

make_timesteps(train,0)
# shape(-1,5,2)
x_train=np.reshape(data_X_train,(-1,timesteps,2))
# shape(-1,2)
y_train=np.array(data_Y_train)


# inputs=Input(shape=(timesteps,feature))
# LSTM_1=LSTM(30,activation='relu',return_sequences=True)(inputs)
# Dense_1=Dense(60)(LSTM_1)
# # dropout=Dropout(0.5)(Dense_1)
# LSTM_2=LSTM(30,activation='relu')(Dense_1)
#
# output=Dense(2,activation='tanh')(LSTM_2)
#
#
# model=Model(inputs=inputs,outputs=output)
# model.compile(loss='mae', optimizer='adam')
# model.fit(x_train,y_train,batch_size=1,epochs=2,shuffle=False,verbose=1)
# model.save('model_10000_relu_2layers.h5')
model=load_model('model_10000_relu_2layers.h5')



y_predit=model.predict(x_train)

draw_line(y_train[:,0][0:30000],y_train[:,1][0:30000],y_predit[:,1][0:30000])