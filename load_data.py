import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler


Latitude_list=[]
Longitude_list=[]
feet_list=[]


# 接收文件路径，添加四个数组
def read_plt(read_path):
    with open(read_path,'r')as f:
        for item in f.readlines()[6:]:
            item.replace(r'\n','')
            Latitude,Longitude,_,feet,PassDay,_,_=item.split(',')

            Latitude_list.append(Latitude)
            Longitude_list.append(Longitude)


# 获得制定路径下的 plt 的文件路径
def get_plt():
    path = '/home/liyang/PycharmProjects/LSTM_practice/008/'
    for fpathe, dirs, fs in os.walk(path):
        for f in fs:
            if os.path.splitext(f)[1]=='.plt':
                read_plt(fpathe+r'/'+f)

    # list to array
    Latitude_array=np.array(Latitude_list,dtype=float)
    Longitude_array=np.array(Longitude_list,dtype=float)

    print('经度 : ',Latitude_array.shape)
    return Latitude_array,Longitude_array

def load_data():
    X,Y=get_plt()
    min_max_scaler = MinMaxScaler()
    # X = min_max_scaler.fit_transform(X)
    # Y = min_max_scaler.fit_transform(Y)

    Z=np.vstack((X,Y)).T
    Z = min_max_scaler.fit_transform(Z)
    return Z[:65000],Z[65000:77911]


if __name__ == '__main__':
    load_data()