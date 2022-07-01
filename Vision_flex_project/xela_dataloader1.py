# -*- coding: utf-8 -*-

import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import deepdish as dd
from PIL import Image
import csv
import numpy as np
from time import sleep
import cv2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd




def readExcel_flex(file):
    df = pd.read_excel(file, header = None)
    df = df.T
    #print(df)
    
    # 时间
    T = df.iloc[0, 1:]
    T = T.astype(float)
    T = np.array(T, dtype=np.float64)
    
    # 电阻值
    R = df.iloc[1, 1:]   
    R = R.astype(float)
    R = np.array(R, dtype = np.float64) # Ω
    #print(R)
    
    # 压力值 
    #P = 1 / ((-1.19e-4) + R * 5e-8) #  g
    #P = 1 / ( ((-1.19e-4) + R * 5e-8) * 102 ) # N
    #print(P)
    P=R
    
    return P


def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

def load_data(path,clas,init,length,log):

    num=0
    caselist=os.listdir(path+"/"+clas+'/')
    cases=[]
    for case in caselist:
        pathTemp_visual=path+"/"+clas+'/'+case+'/'+'visual/'
        pathTemp_flex=path+"/"+clas+'/'+case+'/'+'flex/'
        num_visual=len(os.listdir(pathTemp_visual))    
        num_flex=len(os.listdir(pathTemp_flex))
        YearMonthDay,Hour,Minutes,label=case.split("-")
        YearMonthDay_true = YearMonthDay[0:8]+'-'+ Hour + '-' + Minutes  + '-' + label 
        excelName = YearMonthDay +'-'+ Hour + '-' + Minutes  + '-' + label + '.xlsx'
                   
        pathTemp_flex = path + "/" + clas + '/' + case + '/' + 'flex/'  + YearMonthDay_true + '.xlsx'
        flex_data = readExcel_flex(pathTemp_flex)
        num_flex = len(flex_data) 
        for i in range(init,num_visual-length,log):
            rowTemp=[]
            rowTemp.append(YearMonthDay)
            rowTemp.append(Hour)
            rowTemp.append(Minutes)
            rowTemp.append(label)
            for k in range(length):
                rowTemp.append(pathTemp_visual+str(i+k)+'.jpg')
                        

            flex_lenth=3
            for m in range(flex_lenth):
                index = num_flex - m - int(i/2)-1 
                rowTemp.append(flex_data[index]) 
            cases.append(rowTemp)
                     
    return cases

def train_test_dataset(path,visual_seq_length,flex_seq_length,log,flag):

    
    baishikele=load_data(path, 'baishikele', 6, visual_seq_length,log)
    jiandao=load_data(path, 'jiandao',  6, visual_seq_length, log)
    jiaodai=load_data(path, 'jiaodai', 6, visual_seq_length, log)
    lvjian=load_data(path, 'lvjian',  6, visual_seq_length, log)
    mutangchun=load_data(path, 'mutangchun', 6, visual_seq_length, log)
    tixugao=load_data(path, 'tixugao',  6, visual_seq_length, log)
    zhijiayou=load_data(path, 'zhijiayou',  6, visual_seq_length, log)
    yanjinghe=load_data(path, 'yanjinghe',  6, visual_seq_length, log)
    
    
    train_dataset=baishikele+jiandao+jiaodai+lvjian+mutangchun+tixugao+yanjinghe
    test_dataset=zhijiayou
    
    if flag == 'train':
        dataset=train_dataset
    elif flag == 'test':
        dataset=test_dataset
    return dataset



class MyDataset(Dataset):
    def __init__(self,  image_paths, visual_seq_length,flex_seq_length,transform_v,transform_f,log,flag):
        self.image_paths = image_paths
        self.visual_seq_length = visual_seq_length
        self.flex_seq_length = flex_seq_length
        self.transform_v = transform_v
        self.transform_f = transform_f
        self.label=[]
        self.visual_sequence=[]
        self.flex_sequence=[]
        self.classes=['0','1']
        self.log=log
        self.flag=flag
        self.dataset=train_test_dataset(self.image_paths,self.visual_seq_length,self.flex_seq_length,self.log, self.flag)
        le = LabelEncoder()
        le.fit(self.classes)

# convert category -> 1-hot
        action_category = le.transform(self.classes).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(action_category)
        for item in self.dataset:
            self.label.append(str(item[3]))
            visual =[]
            flex=[]
            for i in range(self.visual_seq_length):
                visual.append(item[4+i])
            for k in range(self.flex_seq_length):
                flex.append(item[k+4+self.visual_seq_length])
            self.visual_sequence.append(visual)
            self.flex_sequence.append(flex)
        self.label=labels2cat(le, self.label)
    def __getitem__(self, index):

        visuals = []
        flexs=[]
        self.flex_sequence = np.array(self.flex_sequence) 
        self.flex_sequence = torch.from_numpy(self.flex_sequence)
        for i in range(self.visual_seq_length):
            visualTemp=Image.open(self.visual_sequence[index][i])
            if self.transform_v:
                visualTemp = self.transform_v(visualTemp)
            visuals.append(visualTemp.unsqueeze(1))
        for k in range(self.flex_seq_length):
            flexTemp = self.flex_sequence[index][k] 
            flexs.append(flexTemp.unsqueeze(0))
            


        x_v = torch.cat(visuals,dim=1)
        x_f = torch.from_numpy( np.array(flexs).astype(float) )
        y = torch.tensor(self.label[index], dtype=torch.long)
        return x_v,x_f, y

    def __len__(self):
        return len(self.visual_sequence)


