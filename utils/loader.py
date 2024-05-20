# -*- coding: utf-8 -*-

import os
import torch.utils.data as Data
import torch
import numpy as np
import pandas as pd
from torch.utils.data.sampler import WeightedRandomSampler
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sdv.sampling import Condition
import opt
device = opt.device

#todo 修改bug
def transform_dataset(x_data,y_data,n_input, n_output):
    #n_output固定为1
    all_data = x_data
    data_size=x_data.shape[0]
    X = np.empty((data_size-n_input+1, n_input, all_data.shape[1]))
    Y = np.empty((data_size-n_input+1, y_data.shape[1]))
    for i in range(data_size - n_input+1):
        X[i] = all_data[i:i + n_input, :]
        Y[i] = y_data[i + n_input-1, :]
    return X,Y


# 准备数据
def get_data(path,step=512):
    
    train = pd.read_csv(path)
    train.columns=['speed', 'speed_med_5', 'speed_med_20', 'speed_SD_5', 'speed_SD_20',
                                  'acceleration', 'acceleration_med_5', 'acceleration_med_20', 'acceleration_SD_5', 'acceleration_SD_20',
                                  'angular_speed', 'angular_speed_med_5', 'angular_speed_med_20', 'angular_speed_SD_5', 'angular_speed_SD_20',
                                  'angular_acceleration', 'angular_acceleration_med_5', 'angular_acceleration_med_20', 'angular_acceleration_SD_5', 'angular_acceleration_SD_20',
                                  'angle_diff', 'angle_diff_med_5', 'angle_diff_med_20', 'angle_diff_SD_5', 'angle_diff_SD_20',
                                  'tag','lon','lat']
    train_data = train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]].to_numpy().astype(float)
    train_tag= train.iloc[:, [25]].to_numpy().astype(int)

    sss=ShuffleSplit(n_splits=1,test_size=opt.testRatio,random_state=0)
    for train_index,test_index in sss.split(train_data,train_tag):
        X_train,X_test=train_data[train_index],train_data[test_index]
        y_train,y_test=train_tag[train_index],train_tag[test_index]
    
    sss=ShuffleSplit(n_splits=1,test_size=opt.valRatio,random_state=0)
    for train_index,test_index in sss.split(X_train,y_train):
        X_train,X_valid=X_train[train_index],X_train[test_index]
        y_train,y_valid=y_train[train_index],y_train[test_index]

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_valid = sc.transform(X_valid)
    # X_test = sc.transform(X_test)

    # X_all=sc.fit_transform(train_data)
    X_all=train_data

    if opt.useGAN==True:
        data_train=np.c_[X_train,y_train]
        data_train = pd.DataFrame(data_train,columns=['speed', 'speed_med_5', 'speed_med_20', 'speed_SD_5', 'speed_SD_20',
                    'acceleration', 'acceleration_med_5', 'acceleration_med_20', 'acceleration_SD_5', 'acceleration_SD_20',
                    'angular_speed', 'angular_speed_med_5', 'angular_speed_med_20', 'angular_speed_SD_5', 'angular_speed_SD_20',
                    'angular_acceleration', 'angular_acceleration_med_5', 'angular_acceleration_med_20', 'angular_acceleration_SD_5', 'angular_acceleration_SD_20',
                    'angle_diff', 'angle_diff_med_5', 'angle_diff_med_20', 'angle_diff_SD_5', 'angle_diff_SD_20',
                    'lon','lat','tag'])
        weightsPath=opt.GAN_path
        if(opt.useGAN_weights and os.path.exists(weightsPath)):
            model=CTGANSynthesizer.load(weightsPath)
        else:
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data_train)
            model = CTGANSynthesizer(
                metadata,
                epochs=opt.GAN_epoch,
                verbose=True
            )
            model.fit(data_train)
            model.save(weightsPath)
        ser=data_train['tag'].value_counts()
        num_rows=abs(ser[0]-ser[1])
        cnd=0
        if ser[0]>ser[1]:
            cnd=1
        condition = Condition({'tag': cnd}, num_rows=num_rows)
        synthetic_train = model.sample_from_conditions(conditions=[condition])
        data_train=pd.concat([data_train,synthetic_train], axis=1)
        X_train = data_train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]].to_numpy().astype(float)
        y_train= data_train.iloc[:, [27]].to_numpy().astype(int).reshape(-1,1)
        # data_train.to_csv(r"./data/generate/ctgan.csv", index=False)

    X_train,y_train=transform_dataset(X_train,y_train,step,1)
    X_valid,y_valid=transform_dataset(X_valid,y_valid,step,1)
    X_test,y_test=transform_dataset(X_test,y_test,step,1)
    X_all,y_all=transform_dataset(X_all,train_tag,step,1)


    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train).to(device)
    X_valid = torch.tensor(X_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid).to(device)
    X_test=torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test).to(device)
    X_all=torch.tensor(X_all, dtype=torch.float32).to(device)
    y_all = torch.tensor(y_all).to(device)
    return X_train,y_train,X_valid,y_valid,X_test,y_test,X_all,y_all

#返回训练loader
def get_loader(path, step=512, batch_size=128,num_workers=4):
    X_train,y_train,X_valid,y_valid,X_test,y_test,X_all,y_all=get_data(path,step)
    return __dataLoader(X_train,y_train,batch_size,num_workers),__dataLoader(X_valid,y_valid,batch_size,num_workers),__dataLoader(X_test,y_test,batch_size,num_workers),__dataLoader(X_all,y_all,batch_size,num_workers)

# loader权重
def __dataLoader(X,Y,batch_size,num_workers=4):
    train_dataset = Data.TensorDataset(X.to(device), Y.to(device))
    if opt.useGAN==True:
        weights = [3 if label == 0 else 1 for label in Y]
        sampler = WeightedRandomSampler(weights,num_samples=Y.size()[0], replacement=True)
        train_loader=Data.DataLoader( 
        dataset=train_dataset, batch_size=batch_size,num_workers=num_workers,sampler=sampler
        )
    else:
        train_loader = Data.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    return train_loader

#返回predict loader
def get_predict_loader(path, step=512, batch_size=128,num_workers=4):
    train = pd.read_csv(path)
    train.columns=['speed', 'speed_med_5', 'speed_med_20', 'speed_SD_5', 'speed_SD_20',
                                  'acceleration', 'acceleration_med_5', 'acceleration_med_20', 'acceleration_SD_5', 'acceleration_SD_20',
                                  'angular_speed', 'angular_speed_med_5', 'angular_speed_med_20', 'angular_speed_SD_5', 'angular_speed_SD_20',
                                  'angular_acceleration', 'angular_acceleration_med_5', 'angular_acceleration_med_20', 'angular_acceleration_SD_5', 'angular_acceleration_SD_20',
                                  'angle_diff', 'angle_diff_med_5', 'angle_diff_med_20', 'angle_diff_SD_5', 'angle_diff_SD_20',
                                  'tag','lon','lat']
    train_data = train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]].to_numpy().astype(float)
    train_tag= train.iloc[:, [25]].to_numpy().astype(int)

    # sc = StandardScaler()
    # X_all=sc.fit_transform(train_data)

    X_all=train_data
    X_all,y_all=transform_dataset(X_all,train_tag,step,1)

    X_all=torch.tensor(X_all, dtype=torch.float32).to(device)
    y_all = torch.tensor(y_all).to(device)

    dataset = Data.TensorDataset(X_all.to(device), y_all.to(device))
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)
    
    return loader

if __name__=="__main__":
    #测试数据
    get_data(r"data\train\全部标注数据合并.csv")
