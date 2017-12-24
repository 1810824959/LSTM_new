import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
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






model = Sequential()
model.add(LSTM(50, activation='relu',input_shape=(timesteps,feature)))
# model.add(LSTM(50,activation='relu'))
model.add(Dense(2,activation='relu'))
model.compile(loss='mae', optimizer='adam')
model.fit(x_train,y_train,batch_size=1,epochs=5,shuffle=False,verbose=1)
model.save('model_10000_relu.h5')
# model=load_model('model_1.h5')

y_predit=model.predict(x_train)
# with open('1.txt','w')as f:
#     f.write(str(y_predit))
# y_predit=[]
# for i in x_train:
#     y=model.predict(i)
#     y_predit.append(y)
# y_predit=np.array(y_predit)
# print(y_predit.shape)


# y_predit=[]
# print(x_test)
draw_line(y_train[:,0][0:3000],y_train[:,1][0:3000],y_predit[:,1][0:3000])
