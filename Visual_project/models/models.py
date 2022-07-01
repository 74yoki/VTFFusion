# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import numpy
#from mypath import Path




class C3D(nn.Module):

     def __init__(self, v_dim=15, img_xv=256, img_yv=256, drop_p_v=0.2, fc_hidden_v=256, ch1_v=32,ch2_v=48,
                  ch1_t=8,ch2_t=12,t_dim=30, img_xt=4, img_yt=4, drop_p_t=0.2, fc_hidden_t=64,fc_hidden_1=128,num_classes=2):
        super(C3D, self).__init__()
        self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        self.tactile_c3d=CNN3D1(t_dim=t_dim, img_x=img_xt, img_y=img_yt, drop_p=drop_p_t, fc_hidden1=fc_hidden_t,ch1=ch1_t)
        
        self.fc1 = nn.Linear(fc_hidden_v+fc_hidden_t, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p=drop_p_v

     def forward(self,x_3d_v,x_3d_t):
         x_v=self.visual_c3d(x_3d_v)
         x_t=self.tactile_c3d(x_3d_t)
         x=torch.cat((x_v,x_t),-1)
         x=F.relu(self.fc1(x))
         x = F.dropout(x, p=self.drop_p, training=self.training)
         x = self.fc2(x)

         return x


class C3D_visual(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, pretrained=False,length=5,img_size=112):
        super(C3D_visual, self).__init__()
        self.img_size=img_size
        if length == 5:
            if img_size == 112:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1,2, 2), stride=(1, 2, 2))

                self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

                #self.fc6 = nn.Linear(8192, 4096)
                #self.fc7 = nn.Linear(4096, 4096)
                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)
            elif img_size == 32:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)
            elif img_size == 64:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)
            elif img_size ==224:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv6a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv6b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)

            elif img_size == 512:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv6a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv6b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv7a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv7b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool7 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)

        if length == 6:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),padding=(1,0,0))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(2,2, 2), stride=(2, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)
        if length == 7:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(2,2, 2), stride=(2, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)
        if length == 8:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(2,2, 2), stride=(2, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)
        if length == 3:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)
        if length == 4:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)

        # self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        #
        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        if self.img_size == 32:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            # x = self.relu(self.conv4a(x))
            # x = self.relu(self.conv4b(x))
            # x = self.pool4(x)
            # # print(x.shape)
            # x = self.relu(self.conv5a(x))
            # x = self.relu(self.conv5b(x))
            # x = self.pool5(x)
            # print(x.shape)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        elif self.img_size ==64:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            # x = self.relu(self.conv5a(x))
            # x = self.relu(self.conv5b(x))
            # x = self.pool5(x)
            # print(x.shape)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        elif self.img_size ==112:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)
            # print(x.shape)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        elif self.img_size ==224:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)
            # print(x.shape)
            x = self.relu(self.conv6a(x))
            x = self.relu(self.conv6b(x))
            x = self.pool6(x)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        elif self.img_size == 512:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)
            # print(x.shape)
            x = self.relu(self.conv6a(x))
            x = self.relu(self.conv6b(x))
            x = self.pool6(x)
            x = self.relu(self.conv7a(x))
            x = self.relu(self.conv7b(x))
            x = self.pool7(x)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        # logits = self.fc8(x)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class C3D_visual_only(nn.Module):
    def __init__(self, drop_p_v=0.2, visual_dim=4096, fc_hidden_1=128, num_classes=2):  
        super(C3D_visual_only, self).__init__()
        self.visual_c3d = C3D_visual(pretrained=False,length=6)
        self.fc1 = nn.Linear(visual_dim, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_v

    def forward(self, x_3d_v):
 
        x_v = self.visual_c3d(x_3d_v) 
        x = F.relu(self.fc1(x_v))
        x = self.fc2(x)
        return x
