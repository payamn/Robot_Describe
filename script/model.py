from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import collections
import thread
import constants
import os
import random
import shutil
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import sys
from logger import Logger
import torchvision.models as models

from utils import model_utils
from script.utility import CheckPointSaver

from torch.utils.data import DataLoader

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.05, training=True):
        super(DynamicGNoise, self).__init__()
        self.training = training
        self.std = std

    def forward(self, x):
        if not self.training: return x
        self.noise = Variable(torch.zeros(x.size()).cuda())
        self.noise.data.normal_(0, std=self.std)
        noisy_out = x - self.noise
        noisy_out.data.clamp_(0,1)
        return (noisy_out)



# class Encoder_RNN_Laser(nn.Module):
#     def __init__(self, input_size, hidden_size, n_layers=1, training=True, batch_size = 1):
#         super(Encoder_RNN_Laser, self).__init__()
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.noise = DynamicGNoise((batch_size,360), 0.05, training=training)
#         self.conv = nn.Conv1d(input_size, hidden_size, 4)
#         self.conv2 = nn.Conv1d(hidden_size, hidden_size, 5)
#
#         self.linear = nn.Linear(hidden_size*(360-3), int(self.hidden_size*3/4))
#         self.linear2 = nn.Linear(3, int(self.hidden_size/4))
#         self.lstm = nn.LSTM(hidden_size=hidden_size, input_size=hidden_size, batch_first=True)
#
#     def forward(self, speed_input, laser_input, hidden):
#         batch_size = laser_input.size()[0]
#         noisy_laser_input = self.noise(laser_input)
#         conv = self.conv(noisy_laser_input.view(batch_size, 1, 360))
#         laser_out = self.linear(conv.view(batch_size, 1, -1))
#         speed_input = speed_input.unsqueeze(1)
#         speed_out = self.linear2(speed_input)
#         output_combine = torch.cat((speed_out, laser_out),2)
#         output, hidden = self.lstm(output_combine, hidden)
#         return output, hidden
#
#     def initHidden(self, batch_size):
#         result = (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)), Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
#         if use_cuda:
#             result = tuple([i.cuda() for i in result])
#         return result


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Network_Map(nn.Module):
    def __init__(self, input_size, output_size, max_length=MAX_LENGTH, num_of_prediction=5, len_parent=4, training=True):
        super(Network_Map, self).__init__()
        self.input_size = input_size[0]
        output_feature = input_size[1]
        resnet34 = models.resnet34(pretrained=True)
        modules = list(resnet34.children())[:-3]
        print (resnet34)
        self.resnet34 = nn.Sequential(*modules)

        # print (self.resnet34)
        self.conv0 = nn.Conv2d(1, 16, 4)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(16, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 7)
        self.conv5 = nn.Conv2d(64, 64, 9 )
        self.conv6 = nn.Conv2d(64, 32, 9 )
        self.conv7 = nn.Conv2d(32, 32, 9 )
        self.conv8 = nn.Conv2d(32, 16, 9 )
        self.conv9 = nn.Conv2d(16, 24, 9 )
        self.conv10 = nn.Conv2d(24, 24, 9 )
        self.conv11 = nn.Conv2d(24, 24, 9 )

        self.max_pool = nn.MaxPool2d(4, stride=2)

        self.numb_of_prediction = num_of_prediction

        self.output_size = output_size
        self.linear = nn.Linear(256*15*15, constants.GRID_LENGTH*constants.GRID_LENGTH*2*12)
        # self.linear_pose = nn.Linear(25088,  2 * num_of_prediction)
        # self.linear_classes = nn.Linear(25088, self.output_size * num_of_prediction)
        # self.linear_classes_parents = nn.Linear(512*4*4 , self.len_parent * num_of_prediction)

        self.soft_max = nn.LogSoftmax(dim=4)
        self.tanh = nn.Tanh()
    def forward(self, map):
        batch_size = map.size()[0]
        resnet = self.resnet34(map)
        predict = self.linear(resnet.view(batch_size,-1))
        # conv = self.conv0 (map)
        # conv = self.conv1 (self.relu(conv))
        # conv = self.max_pool(self.relu(conv))
        # conv = self.conv2 (self.relu(conv))
        # conv = self.conv3 (self.relu(conv))
        # conv = self.conv4 (self.relu(conv))
        # conv = self.conv5 (self.relu(conv))
        # conv = self.conv6 (self.relu(conv))
        # conv = self.conv7 (self.relu(conv))
        # conv = self.conv8 (self.relu(conv))
        # conv = self.conv9 (self.relu(conv))
        # conv = self.conv10 (self.relu(conv))
        # # conv1 = self.conv1(resnet)
        # predict = self.conv11(conv).permute(0,2,3,1)
        # predict axis are B,W,H,Anchors,Objectness+(x,y)+classes
        predict = predict.view(batch_size, constants.GRID_LENGTH, constants.GRID_LENGTH, 2, 12)
        # poses_out = (self.tanh(self.linear_pose(resnet)) + 1.0) / 2
        # poses_out = poses_out.view(batch_size, self.numb_of_prediction, 2)
        # classes_out = self.linear_classes(resnet.view(batch_size, 1, -1))
        # classes_out = self.soft_max(classes_out.view(batch_size, 1, self.numb_of_prediction, self.output_size))
        classes_out = self.soft_max(predict[:,:,:,:,3:])

        # pose and objectness between 0 1:
        pose_objectness = predict[:,:,:,:,0:3] #(self.tanh(predict[:,:,:,:,0:3]) + 1.0) / 2

        poses = pose_objectness[:, :, :, :, 1:3]
        objectness = pose_objectness[:, :, :, :, 0]
        # return poses_out, classes_out
        return classes_out, poses, objectness


class DecoderNoRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH, num_of_prediction=5):
        super(DecoderNoRNN, self).__init__()
        self.numb_of_prediction = num_of_prediction
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.linear_classes = nn.Linear(hidden_size*(max_length+1), self.output_size * num_of_prediction)
        self.linear_poses = nn.Linear(hidden_size*(max_length+1), 2 * num_of_prediction)

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.soft_max = nn.LogSoftmax(dim=3)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.view(encoder_outputs.size()[0], 1, -1)
        hidden = torch.transpose(hidden[0], 0, 1)
        encoder_outputs = torch.cat((hidden, encoder_outputs), -1)

        # output = concat.unsqueeze(0)
        output_classes = self.linear_classes(encoder_outputs)
        output_classes = output_classes.view(encoder_outputs.size()[0], 1, self.numb_of_prediction, self.output_size)
        output_classes = self.soft_max(output_classes)
        output_poses = self.linear_poses(encoder_outputs)
        output_poses = output_poses.view(encoder_outputs.size()[0], self.numb_of_prediction, 2)
        return output_classes, output_poses

class Map_Model:
    # def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
    #     filename = os.path.join(self.project_path, filename)
    #     torch.save(state, filename)
    #     if is_best:
    #         shutil.copyfile(filename, os.path.join(self.project_path, 'model_best.pth.tar'))

    def __init__(self, dataset_train=None, dataset_validation=None, resume_path = None, learning_rate = 0.001, load_weight = True, save=True, real_time_test=False, log=True, cuda=None):
        self.start = time.time()

        self.best_epoch_acc_objectness = None
        self.best_epoch_acc_classes = None
        self.best_epoch_loss = None
        self.best_validation_acc_objecness = None
        self.best_validation_acc_classes = None
        self.best_validation_loss = None

        self.checkpoint_saver = CheckPointSaver(['epoch_loss', 'epoch_accuracy_objectness', 'epoch_accuracy_classes',
                                                'validation_loss', 'validation_accuracy_classes',
                                                'validation_accuracy_objectness'],
                                               best_metrics={'epoch_loss': self.best_epoch_loss,
                                                             'epoch_accuracy_objectness': self.best_epoch_acc_objectness,
                                                             'epoch_accuracy_classes': self.best_epoch_acc_classes,
                                                             'validation_loss': self.best_validation_loss,
                                                             'validation_accuracy_classes': self.best_validation_acc_classes,
                                                             'validation_accuracy_objectness': self.best_validation_acc_objecness
                                                             },
                                                model_dir=os.path.dirname(resume_path))

        self.dataset = dataset_train
        self.dataset_validation = dataset_validation
        self.word_encoding = model_utils.WordEncoding()
        self.learning_rate = learning_rate
        self.best_lost = sys.float_info.max
        self.distance = nn.PairwiseDistance()
        if cuda is None:
            self.use_cuda = torch.cuda.is_available()
        else:
            self.use_cuda = cuda
        # remove previous tensorboard data first
        import subprocess as sub
        if (log):
            # p = sub.Popen(['rm', '-r', './logs'])
            # p.communicate()
            self.logger = Logger('./logs')
            print ("log in on")
        else:
            print ("log is off")
            self.logger = None
        self.model = Network_Map((1,(244,244)), self.word_encoding.len_classes())
        if self.use_cuda:
            self.model = self.model.cuda()

        print("model size:")
        for parameter in self.model.parameters():
            print(parameter.size())


        # Model
        # self.weight_loss = torch.ones([len(self.word_encoding.classes)])
        # self.weight_loss[self.word_encoding.classes["noting"]] = 0.05  # nothing
        self.criterion_classes = nn.NLLLoss()# weight=self.weight_loss.cuda())
        self.criterion_poses = nn.MSELoss(size_average=False)
        self.criterion_objectness = nn.MSELoss()


        self.start_epoch = 0
        self.optimizer = optim.Adadelta(self.model.parameters())#self.model.parameters(), lr=self.learning_rate, weight_decay=0.06)
        self.project_path = "."
        if resume_path is not None and load_weight:
            self.project_path = os.path.dirname(resume_path)
            if os.path.isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path)
                self.start_epoch = checkpoint['epoch']
                # self.best_lost = checkpoint['epoch_lost']
                state = self.model.state_dict()
                state_load = checkpoint['model_dict']
                # del state_load["conv2.weight"]
                # del state_load["conv2.bias"]
                state.update(state_load)
                self.model.load_state_dict(state)
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume_path))

    def model_forward(self, batch_size, mode, iter, plot=False):
        iter_data_loader = None
        dataset = None
        dataloader = None
        if mode == "train":
            dataloader = self.dataloader
            dataset = self.dataset
        elif mode == "validation":
            dataloader = self.dataloader_validation
            dataset = self.dataset_validation
        iter_data_loader = dataloader.__iter__()
        print (mode, len(dataloader))
        epoch_loss_total = 0
        epoch_loss_classes = 0
        epoch_loss_poses = 0
        epoch_loss_objectness = 0
        epoch_accuracy_classes = []
        epoch_accuracy_objectness = []
        epoch_accuracy_poses = []
        accuracy_each_class = {i: [] for i in range(len(self.word_encoding.classes))}
        for batch, (target_classes, target_poses, target_objectness, local_map, laser_map) in enumerate(iter_data_loader):
            loss_classes = 0
            loss_poses = 0
            loss_objectness = 0
            self.optimizer.zero_grad()
            batch_accuracy_classes = []
            batch_accuracy_poses = []
            local_map = Variable(local_map).cuda() if self.use_cuda else Variable(local_map)
            laser_map = Variable(laser_map).cuda() if self.use_cuda else Variable(laser_map)
            target_classes = Variable(target_classes).cuda() if self.use_cuda else Variable(target_classes)
            target_poses = Variable(target_poses).cuda() if self.use_cuda else Variable(target_poses)
            target_objectness = Variable(target_objectness).cuda() if self.use_cuda else Variable(target_objectness)

            classes_out, poses, objectness = self.model(local_map)

            topv, topi = classes_out.data.topk(1)

            if plot:
                self.word_encoding.visualize_map(local_map[:,0,:,:], local_map[:,1,:,:], topi, poses, objectness,
                                                    target_classes, target_poses, target_objectness)
            # b = word_encoded_class != self.word_encoding.classes["noting"]
            # b = b.type(torch.FloatTensor).view(batch_size, -1, 1).repeat(1, 1, 2).cuda()
            # output_poses = output_poses * b
            # word_encoded_pose = word_encoded_pose * b

            accuracy = []
            object_acc = []
            accuracy_class = 0
            acc = topi== target_classes
            for batch_idx in range (target_classes.shape[0]):
                for x in range (target_classes.shape[1]):
                    for y in range (target_classes.shape[2]):
                        for anchor in range(target_classes.shape[3]):
                            if (target_objectness[batch_idx][x][y][anchor].item()> constants.ACCURACY_THRESHOLD and objectness[batch_idx][x][y][anchor].item()>constants.ACCURACY_THRESHOLD):
                                object_acc.append(1)
                                if acc[batch_idx][x][y][anchor]:
                                    accuracy.append(1)
                                    accuracy_class = 1
                                else:
                                    accuracy.append(0)
                                    accuracy_class = 0
                            elif (target_objectness[batch_idx][x][y][anchor].item()<= constants.ACCURACY_THRESHOLD and objectness[batch_idx][x][y][anchor].item()<=constants.ACCURACY_THRESHOLD):
                                continue
                            else:
                                accuracy.append(0)
                                object_acc.append(0)
                                accuracy_class = 0
                            accuracy_each_class[target_classes[batch_idx][x][y][anchor].item()].append(accuracy_class)

            acc_classes = np.mean(accuracy)
            acc_objectness = np.mean(object_acc)
            epoch_accuracy_classes.append(acc_classes)
            epoch_accuracy_objectness.append(acc_objectness)
            # for batch in range(batch_size):
            #     for j in range(dataset.prediction_number):
            #         accuracy = float(word_encoded_class[i][j] == topi[i][0][j][0])
            #         index = int(word_encoded_class[i][j].cpu().data)
            #         accuracy_each_class[index].append(accuracy)
            #         if word_encoded_class.data[i][j] == self.word_encoding.classes["noting"] and accuracy == 1:
            #             continue
            #         else:
            #             batch_accuracy_classes.append(accuracy)
            #             epoch_accuracy_classes.append(accuracy)
            #             distance = self.distance(word_encoded_pose[i][j].view(1, 2), output_poses[i][j].view(1, 2))
            #             # distance = distance.squeeze(1)
            #             # accuracy_poses = torch.mean(distance)
            #             batch_accuracy_poses.append(distance.item())
            #             epoch_accuracy_poses.append(distance.item())
            # output_classes = output_classes.permute(0, 3, 1, 2)
            # output_classes_parent = output_classes_parent.permute(0,3,1,2)
            # word_encoded_class = word_encoded_class.unsqueeze(1)
            # word_encoded_class_parent = word_encoded_class_parent.unsqueeze(1)
            # output_classes = output_classes.view(output_classes.size(2), output_classes.size(3))
            # output_poses = output_poses.view(output_poses.size(0), output_poses.size(2), output_poses.size(3))
            # word_encoded_class = word_encoded_class.squeeze(0)

            # for i in range (output_classes.size(1)):
            mask_objectness = target_objectness >= constants.ACCURACY_THRESHOLD


            mask_objectness = mask_objectness.type(torch.cuda.LongTensor) if self.use_cuda else mask_objectness.type(torch.LongTensor)
            if self.use_cuda:
                mask_objectness.cuda()
            target_classes = target_classes.squeeze(4)
            target_classes = target_classes * mask_objectness

            mask_objectness = mask_objectness.unsqueeze(4)
            mask_poses = mask_objectness.repeat(1,1,1,1,2).type(torch.float)
            if self.use_cuda:
                mask_poses = mask_poses.cuda()
            poses = poses * mask_poses
            target_poses = target_poses * mask_poses

            mask_classes = mask_objectness.repeat(1,1,1,1,9).type(torch.float)
            if self.use_cuda:
                mask_classes = mask_classes.cuda()
            classes_out = classes_out * mask_classes
            classes_out = classes_out.permute(0, 4, 1, 2, 3)
            target_classes = target_classes
            # landa_class = 1
            loss_classes = self.criterion_classes(classes_out, target_classes)  # +\
            # (1-landa_class)*criterion_classes_parents(output_classes_parent, word_encoded_class_parent)

            loss_poses = self.criterion_poses(poses, target_poses)
            # landa = 1.0 / 4
            # loss_total = landa * loss_poses + (1 - landa) * loss_classes


            loss_objectness = self.criterion_objectness(objectness, target_objectness)
            loss_total = (130*loss_classes+loss_poses + 100*loss_objectness)/231

            epoch_loss_classes += loss_classes.item() / classes_out.size()[0]
            epoch_loss_poses += loss_poses.item() / classes_out.size()[0]
            epoch_loss_objectness += loss_objectness.item() / classes_out.size()[0]
            epoch_loss_total += loss_total.item() #epoch_loss_classes + epoch_loss_poses + epoch_loss_objectness
            # batch_accuracy_classes = np.mean(batch_accuracy_classes)
            # batch_accuracy_poses = np.mean(batch_accuracy_poses)
            # loss = loss/batch_size
            print_loss_avg = epoch_loss_total / ((batch + 1))
            print(mode+' Epoch %d %s (%.2f%%) %.6f acc_objectness:%.4f acc_classes:%.4f' % (
            iter + 1, timeSince(self.start, (batch + 1) / (len(dataloader)))
            , float((batch + 1)) / (len(dataloader)) * 100, print_loss_avg, acc_objectness, acc_classes))

            if mode== "train":
                loss_total.backward()
                self.optimizer.step()
            if self.logger is not None:
                info = {
                    'batch_loss'+mode: loss_total.item(),
                    'batch_accuracy_classes'+mode: acc_classes,
                    'batch_accuracy_objecnes'+mode: acc_objectness,
                    # 'batch_accuracy_poses'+mode: batch_accuracy_poses,
                    'loss_poses'+mode: loss_poses.item(),
                    'batch_loss'+mode: loss_total.item(),
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iter * (batch_size) + batch)
            else:
                print ("no logger")
        epoch_accuracy_classes = np.mean(epoch_accuracy_classes)
        epoch_accuracy_objectness = np.mean(epoch_accuracy_objectness)
        # epoch_accuracy_poses = np.mean(epoch_accuracy_poses)
        if self.logger is not None:
            info = {
                'Epoch_loss_'+mode: epoch_loss_total / len(dataloader),
                'Epoch_accuracy_classes_'+mode: epoch_accuracy_classes,
                'epoch_accuracy_objectness'+mode: epoch_accuracy_objectness,
                # 'Epoch_accuracy_poses_'+mode: epoch_accuracy_poses,
                'epoch_loss_objectness_'+mode: epoch_loss_objectness / len(dataloader),
                'epoch_loss_classes_'+mode: epoch_loss_classes / len(dataloader),
                'epoch_loss_poses_'+mode: epoch_loss_poses / len(dataloader)
            }
            for i in range(len(self.word_encoding.classes)):
                info["Epoch_" + mode + "_" + self.word_encoding.get_class_char(i)] = np.mean(accuracy_each_class[i])

            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iter + 1)
        print (mode+ " Epoch %d acc_classes: %.6f acc_objectness: %.6f "%(iter, epoch_accuracy_classes, epoch_accuracy_objectness))

        return epoch_accuracy_classes, epoch_accuracy_objectness, epoch_loss_total



    def visualize_dataset(self, batch_size, dataset=None):
        if dataset is None or dataset=="train":
            dataset = self.dataset
            print ("use train data to visualize")
        else:
            dataset = self.dataset_validation
            print ("use validation data to visualize")
        self.validation(batch_size, 1, False, plot= True, dataset=dataset)

    # def calculate_encoded_pose(self, word_encoded_pose):
    #     divide = torch.FloatTensor([[1 / constants.LOCAL_MAP_DIM, 1 / (constants.LOCAL_MAP_DIM / 2.0)]])
    #     divide = divide.repeat(5, 1)
    #     divide = divide.unsqueeze(0)
    #     addition = torch.FloatTensor([[0, 1]])
    #     addition = addition.repeat(5, 1)
    #     addition = addition.unsqueeze(0)
    #     divide2 = torch.FloatTensor([[1, 0.5]])
    #     divide2 = divide2.repeat(5, 1)
    #     divide2 = divide2.unsqueeze(0)
    #     word_encoded_pose = (word_encoded_pose * divide + addition) * divide2  # to be between 0-1
    #     word_encoded_pose = Variable(word_encoded_pose).cuda() if self.use_cuda else Variable(word_encoded_pose)
    #     return word_encoded_pose

    def validation(self, batch_size, iter, save, plot=False, dataset=None):
        self.start = time.time()

        if dataset is None:
            dataset = self.dataset_validation
        self.dataloader_validation = DataLoader(dataset, shuffle=True, num_workers=5 , batch_size=batch_size, drop_last=True)

        epoch_accuracy_classes, epoch_accuracy_objectness, epoch_loss_total = self.model_forward(batch_size, "validation", iter, plot=plot)

        # print('validation loss: %.4f class_accuracy: %.4f' %
        #       (epoch_loss_total / (len(self.dataloader_validation)), epoch_accuracy_classes))

        is_best = self.best_lost > epoch_loss_total

        if (save == True and plot == False):
            thread.start_new_thread( self.checkpoint_saver.save_checkpoint,
                                     ({
                                         'epoch': iter + 1,
                                         'state_dict': self.model.state_dict().copy(),
                                         'validation_accuracy_classes': epoch_accuracy_classes,
                                         'validation_accuracy_objectness': epoch_accuracy_objectness,
                                         'validation_loss': epoch_loss_total,
                                         'optimizer': self.optimizer.state_dict().copy(),
                                     }, )
                                     )

        print('validation acc classes: %f acc objectness: %f)' % (epoch_accuracy_classes, epoch_accuracy_objectness), "\n\n")


    def train_iters(self, n_iters, print_every=1000, plot_every=100, batch_size=100, save=True):
        self.dataloader = DataLoader(self.dataset, shuffle=True, num_workers=10 , batch_size=batch_size, drop_last=True)
        n_iters = self.start_epoch + n_iters
        start = max (self.start_epoch-1, 0)
        self.validation(batch_size, start, save, plot=False)
        for iter in range(self.start_epoch, n_iters):
            self.start = time.time()
            epoch_accuracy_classes, epoch_accuracy_objectness, epoch_loss_total = self.model_forward(batch_size, "train", iter)
            if save:
                thread.start_new_thread(self.checkpoint_saver.save_checkpoint,
               ({
                    'epoch': iter + 1,
                    'state_dict': self.model.state_dict().copy(),
                    'epoch_accuracy_classes': epoch_accuracy_classes,
                    'epoch_accuracy_objectness': epoch_accuracy_objectness,
                    'epoch_loss': epoch_loss_total,
                    'optimizer': self.optimizer.state_dict().copy(),
                },))
            print (iter, " finished\n\n")
            self.validation(batch_size, iter, save, plot=False)
