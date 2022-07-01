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

# class MySampler(torch.utils.data.Sampler):
#     def __init__(self, end_idx, seq_length):
#         indices = []
#         for i in range(len(end_idx) - 1):
#             start = end_idx[i]
#             end = end_idx[i + 1] - seq_length
#             indices.append(torch.arange(start, end))
#         indices = torch.cat(indices)
#         self.indices = indices
#
#     def __iter__(self):
#         indices = self.indices[torch.randperm(len(self.indices))]
#         return iter(indices.tolist())
#
#     def __len__(self):
#         return len(self.indices)



def readExcel(file):
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
    R = np.array(R, dtype = np.float64) #  Ω
    #print(R)
    
    # 压力值 
    #P = 1 / ((-1.19e-4) + R * 5e-8) # g
    P = 1 / ( ((-1.19e-4) + R * 5e-8) * 102 ) # N
    #print(P)
    
    return P

'''
def readExcel_flex(file):
    df = pd.read_excel(file, header = None)
    df = df.T
    #print(df)
    
    # 时间
    T = df.iloc[0, 1:]
    T = T.astype(float)
    T = np.array(T, dtype=np.float64)
    
    # 电阻值
    R = df.iloc[1, 1:]   #只读取一行，转置过
    R = R.astype(float)
    R = np.array(R, dtype = np.float64) # 这里的电阻单位是 Ω
    #print(R)
    
    # 压力值 
    #P = 1 / ((-1.19e-4) + R * 5e-8) # 单位是 g
    #P = 1 / ( ((-1.19e-4) + R * 5e-8) * 102 ) # 单位是 N
    #print(P)
    P=R
    # 返回拉伸数据【一维，numpy形式】
    return P
'''

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

def load_data(path,clas,init,length,log):
#def load_data(path,clas,init,visual_seq_length,tactile_seq_length,length,log):

    num=0
    
    caselist=os.listdir(path+"/"+clas+'/')
    cases=[]
    for case in caselist:
        # print(case)
        # rowTemßp=[]
        pathTemp_visual=path+"/"+clas+'/'+case+'/'+'visual/'
        pathTemp_tactile=path+"/"+clas+'/'+case+'/'+'tactile/'
        
        num_visual=len(os.listdir(pathTemp_visual))    
        num_tactile=len(os.listdir(pathTemp_tactile))
        
        YearMonthDay,Hour,Minutes,label=case.split("-")
        YearMonthDay_true = YearMonthDay[0:8]+'-'+ Hour + '-' + Minutes  + '-' + label 
        excelName = YearMonthDay +'-'+ Hour + '-' + Minutes  + '-' + label + '.xlsx'
        
        
       
        pathTemp_tactile = path + "/" + clas + '/' + case + '/' + 'tactile/'  + YearMonthDay_true + '.xlsx'
        tactile_data = readExcel(pathTemp_tactile)
        num_tactile = len(tactile_data) 
        
        
        for i in range(init,num_visual-length,log):
            rowTemp=[]
            # print(i)
            rowTemp.append(YearMonthDay)
            rowTemp.append(Hour)
            rowTemp.append(Minutes)
            rowTemp.append(label)
            for k in range(length):
                rowTemp.append(pathTemp_visual+str(i+k)+'.jpg')
            
            
            tactile_lenth=3
            for n in range(tactile_lenth):
                index = num_tactile - n - int(i/2)-1 
                rowTemp.append(tactile_data[index]) 
                # print(path+case,i+k)
            cases.append(rowTemp)
                # print(path+case,i+k)  

            #flex_lenth=3
            #for m in range(flex_lenth):
                #index = num_flex - m - int(i/2)-1 
                #rowTemp.append(flex_data[index]) 
            #cases.append(rowTemp)
                     
    return cases
            # writer_train.writerow(rowTemp)
    # csvFile_train.close()
def train_test_dataset(path,visual_seq_length,tactile_seq_length,log,flag):
    
    
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
    def __init__(self,  image_paths, visual_seq_length,tactile_seq_length,transform_v,transform_t,log,flag):
        self.image_paths = image_paths
        self.visual_seq_length = visual_seq_length
        self.tactile_seq_length = tactile_seq_length
        #self.flex_seq_length = flex_seq_length
        self.transform_v = transform_v
        self.transform_t = transform_t
        #self.transform_f = transform_f
       # self.csvReader=csv.reader(open(image_paths))
        self.label=[]
        self.visual_sequence=[]
        self.tactile_sequence=[]
        #self.flex_sequence=[]
        self.classes=['0','1']
        self.log=log
        self.flag=flag
        self.dataset=train_test_dataset(self.image_paths,self.visual_seq_length,self.tactile_seq_length,self.log, self.flag)
        # self.tactile_sequence_length=[]
        le = LabelEncoder()
        le.fit(self.classes)

# convert category -> 1-hot
        action_category = le.transform(self.classes).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(action_category)
        for item in self.dataset:
            self.label.append(str(item[3]))
            # self.tactile_sequence_length.append(int(item[-1]))
            visual =[]
            tactile=[]
            #flex=[]
            for i in range(self.visual_seq_length):
                visual.append(item[4+i])
            for j in range(self.tactile_seq_length):
                tactile.append(item[j+4+self.visual_seq_length]) 
                #tactile.append(item[j+4])
            #for k in range(self.flex_seq_length):
                #flex.append(item[k+4+self.visual_seq_length+self.tactile_seq_length])
                #flex.append(item[k+4+self.tactile_seq_length])
            self.visual_sequence.append(visual)
            self.tactile_sequence.append(tactile)
            #self.flex_sequence.append(flex)
        self.label=labels2cat(le, self.label)
        #print(len(self.image_sequence))
    def __getitem__(self, index):

        visuals = []
        tactiles=[]
        #flexs=[]
        self.tactile_sequence = np.array(self.tactile_sequence) 
        self.tactile_sequence = torch.from_numpy(self.tactile_sequence)
        #self.flex_sequence = np.array(self.flex_sequence) 
        #self.flex_sequence = torch.from_numpy(self.flex_sequence)
        for i in range(self.visual_seq_length):
            visualTemp=Image.open(self.visual_sequence[index][i])
            if self.transform_v:
                visualTemp = self.transform_v(visualTemp)
            visuals.append(visualTemp.unsqueeze(1))
        for j in range(self.tactile_seq_length):
            tactileTemp = self.tactile_sequence[index][j] 
            tactiles.append(tactileTemp.unsqueeze(0))
        #for k in range(self.flex_seq_length):
            #flexTemp = self.flex_sequence[index][k] 
            #flexs.append(flexTemp.unsqueeze(0))
            


        x_v = torch.cat(visuals,dim=1)
        x_t = torch.from_numpy( np.array(tactiles).astype(float) )
        #x_f = torch.from_numpy( np.array(flexs).astype(float) )
        y = torch.tensor(self.label[index], dtype=torch.long)
        # print(x_v.shape,x_t.shape,y)
        return x_v,x_t, y

    def __len__(self):
        return len(self.visual_sequence)
        #return len(self.tactile_sequence)

