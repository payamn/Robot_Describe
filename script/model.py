from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import collections
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

use_cuda = torch.cuda.is_available()
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



class Encoder_RNN_Laser(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, training=True, batch_size = 1):
        super(Encoder_RNN_Laser, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.noise = DynamicGNoise((batch_size,360), 0.05, training=training)
        self.conv = nn.Conv1d(input_size, hidden_size, 4)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 5)

        self.linear = nn.Linear(hidden_size*(360-3), int(self.hidden_size*3/4))
        self.linear2 = nn.Linear(3, int(self.hidden_size/4))
        self.lstm = nn.LSTM(hidden_size=hidden_size, input_size=hidden_size, batch_first=True)

    def forward(self, speed_input, laser_input, hidden):
        batch_size = laser_input.size()[0]
        noisy_laser_input = self.noise(laser_input)
        conv = self.conv(noisy_laser_input.view(batch_size, 1, 360))
        laser_out = self.linear(conv.view(batch_size, 1, -1))
        speed_input = speed_input.unsqueeze(1)
        speed_out = self.linear2(speed_input)
        output_combine = torch.cat((speed_out, laser_out),2)
        output, hidden = self.lstm(output_combine, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)), Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        if use_cuda:
            result = tuple([i.cuda() for i in result])
        return result


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
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        print (self.resnet18)
        self.len_parent = len_parent
        self.numb_of_prediction = num_of_prediction
        # self.conv1 = nn.Conv2d(self.input_size, 16, 6, padding=5)
        # self.conv2 = nn.Conv2d(16, 16, 5, stride=1, padding=4)
        # self.conv3 = nn.Conv2d(16, 32, 4, stride=1, padding=3)
        # self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # self.conv6 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()
        # self.relu3 = nn.ReLU()
        # self.relu4 = nn.ReLU()
        # self.relu5 = nn.ReLU()
        # self.relu6 = nn.ReLU()
        # for index in range (1, 5):
        #     conv = self.__getattr__("conv" + index.__str__())
        #
        #     kernel_size = conv.kernel_size if type(conv.kernel_size) == tuple \
        #         else (conv.kernel_size, conv.kernel_size)
        #     padding = conv.padding if type(conv.padding) == tuple \
        #         else (conv.padding, conv.padding)
        #     stride = conv.stride if type(conv.stride) == tuple \
        #         else (conv.stride, conv.stride)
        #
        #     output_feature = ((output_feature[0] - kernel_size[0] + 2 * padding[0]) // stride[0] + 1,
        #                       (output_feature[1] - kernel_size[1] + 2 * padding[0]) // stride[1] + 1)
        #     print('conv', index, ':', output_feature)

        self.output_size = output_size
        # self.linear = nn.Linear(output_feature[0]*output_feature[1]*32, 10*10)
        self.linear_pose = nn.Linear(25088,  2 * num_of_prediction)
        self.linear_classes = nn.Linear(25088, self.output_size * num_of_prediction)
        # self.linear_classes_parents = nn.Linear(512*4*4 , self.len_parent * num_of_prediction)

        self.soft_max = nn.LogSoftmax(dim=3)
        self.tanh = nn.Tanh()
    def forward(self, map):
        batch_size = map.size()[0]
        resnet = self.resnet18(map).view(batch_size, 1, -1)
        # conv = self.relu1(self.conv1(map))
        # conv = self.relu2(self.conv2(conv))
        # conv = self.relu3(self.conv3(conv))
        # conv = self.relu4(self.conv4(conv))
        # conv = self.relu5(self.conv5(conv))
        # conv = self.relu6(self.conv6(conv))
        # print (conv4.data.shape)
        # print (conv6.data.shape)
        # exit(0)

        # linear = self.linear(conv.view(batch_size, 1, -1))
        poses_out = (self.tanh(self.linear_pose(resnet)) + 1.0) / 2
        # linear_class = torch.cat((poses_out.view(batch_size, 1, -1), linear), dim=2)
        # parents_out = self.linear_classes_parents(resnet)
        # parents_out = self.soft_max(parents_out.view(batch_size, 1, self.numb_of_prediction, self.len_parent))
        poses_out = poses_out.view(batch_size, self.numb_of_prediction, 2)
        # classes_out = torch.cat((parents_out.view(batch_size, self.numb_of_prediction, self.len_parent), poses_out),2).view(batch_size, 1, -1)
        # classes_out = torch.cat((classes_out, resnet), 2)
        classes_out = self.linear_classes(resnet.view(batch_size, 1, -1))
        classes_out = self.soft_max(classes_out.view(batch_size, 1, self.numb_of_prediction, self.output_size))
        return poses_out, classes_out


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
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.project_path, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.project_path, 'model_best.pth.tar'))

    def __init__(self, dataset_train, dataset_validation, resume_path = None, learning_rate = 0.001, load_weight = True, save=True, real_time_test=False):

        self.dataset = dataset_train
        self.dataset_validation = dataset_validation
        self.learning_rate = learning_rate
        self.best_lost = sys.float_info.max
        self.distance = nn.PairwiseDistance()
        # remove previous tensorboard data first
        import subprocess as sub
        p = sub.Popen(['rm', '-r', './logs'])
        p.communicate()
        self.logger = Logger('./logs')
        self.model = Network_Map((1,(244,244)), self.dataset.word_encoding.len_classes())
        if use_cuda:
            self.model = self.model.cuda()

        print("model size:")
        for parameter in self.model.parameters():
            print(parameter.size())


        # Model
        self.weight_loss = torch.ones([len(self.dataset.word_encoding.classes)])
        self.weight_loss[self.dataset.word_encoding.classes["noting"]] = 0.05  # nothing
        self.criterion_classes = nn.NLLLoss(weight=self.weight_loss.cuda())
        self.criterion_poses = nn.MSELoss(size_average=False)


        self.start_epoch = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.06)
        self.project_path = "."
        if resume_path is not None and load_weight:
            self.project_path = os.path.dirname(resume_path)
            if os.path.isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path)
                self.start_epoch = checkpoint['epoch']
                self.best_lost = checkpoint['epoch_lost']
                self.model.load_state_dict(checkpoint['model_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume_path))

    def model_forward(self, batch_size, mode, iter):
        iter_data_loader = None
        dataset = None
        if mode == "train":
            iter_data_loader = self.dataloader.__iter__()
            dataset = self.dataset
        elif mode == "validation":
            iter_data_loader = self.dataloader_validation.__iter__()
            dataset = self.dataset_validation
        print (mode, len(dataset))
        epoch_loss_total = 0
        epoch_loss_classes = 0
        epoch_loss_poses = 0
        epoch_accuracy_classes = []
        epoch_accuracy_poses = []
        accuracy_each_class = {i: [] for i in range(len(dataset.word_encoding.classes))}
        for batch, (word_encoded_class, word_encoded_pose, local_map) in enumerate(iter_data_loader):
            loss_classes = 0
            loss_poses = 0
            self.optimizer.zero_grad()
            batch_accuracy_classes = []
            batch_accuracy_poses = []
            local_map = Variable(local_map).cuda() if use_cuda else Variable(local_map)
            word_encoded_class = Variable(word_encoded_class).cuda() if use_cuda else Variable(word_encoded_class)
            # word_encoded_class_parent = Variable(word_encoded_class_parent).cuda() if use_cuda else Variable(word_encoded_class_parent)
            word_encoded_pose = self.calculate_encoded_pose(word_encoded_pose)

            output_poses, output_classes = self.model(local_map)

            topv, topi = output_classes.data.topk(1)

            b = word_encoded_class != dataset.word_encoding.classes["noting"]
            b = b.type(torch.FloatTensor).view(batch_size, -1, 1).repeat(1, 1, 2).cuda()
            output_poses = output_poses * b
            word_encoded_pose = word_encoded_pose * b

            accuracy = 0
            for i in range(word_encoded_class.size(0)):
                for j in range(dataset.prediction_number):
                    accuracy = float(word_encoded_class[i][j] == topi[i][0][j][0])
                    index = int(word_encoded_class[i][j].cpu().data)
                    accuracy_each_class[index].append(accuracy)
                    if word_encoded_class.data[i][j] != dataset.word_encoding.classes["noting"]:
                        batch_accuracy_classes.append(accuracy)
                        epoch_accuracy_classes.append(accuracy)
                        distance = self.distance(word_encoded_pose[i][j].view(1, 2), output_poses[i][j].view(1, 2))
                        # distance = distance.squeeze(1)
                        # accuracy_poses = torch.mean(distance)
                        batch_accuracy_poses.append(distance.item())
                        epoch_accuracy_poses.append(distance.item())
            output_classes = output_classes.permute(0, 3, 1, 2)
            # output_classes_parent = output_classes_parent.permute(0,3,1,2)
            word_encoded_class = word_encoded_class.unsqueeze(1)
            # word_encoded_class_parent = word_encoded_class_parent.unsqueeze(1)
            # output_classes = output_classes.view(output_classes.size(2), output_classes.size(3))
            # output_poses = output_poses.view(output_poses.size(0), output_poses.size(2), output_poses.size(3))
            # word_encoded_class = word_encoded_class.squeeze(0)

            # for i in range (output_classes.size(1)):
            landa_class = 1
            loss_classes += landa_class * self.criterion_classes(output_classes, word_encoded_class)  # +\
            # (1-landa_class)*criterion_classes_parents(output_classes_parent, word_encoded_class_parent)

            loss_poses += self.criterion_poses(output_poses, word_encoded_pose)
            landa = 1.0 / 4
            loss_total = landa * loss_poses + (1 - landa) * loss_classes

            epoch_loss_classes += loss_classes.item() / word_encoded_class.size()[0]
            epoch_loss_poses += loss_poses.item() / word_encoded_class.size()[0]
            epoch_loss_total += epoch_loss_classes + epoch_loss_poses
            batch_accuracy_classes = np.mean(batch_accuracy_classes)
            batch_accuracy_poses = np.mean(batch_accuracy_poses)
            # loss = loss/batch_size
            print_loss_avg = epoch_loss_total / (batch_size * (batch + 1))
            print(mode+' Epoch %d %s (%.2f%%) %.4f acc_classes:%.4f acc_posses:%.4f' % (
            iter + 1, timeSince(self.start, (batch + 1) / (len(dataset) // batch_size))
            , float((batch + 1)) / (len(dataset) // batch_size) * 100, print_loss_avg, batch_accuracy_classes,
            batch_accuracy_poses))

            if mode== "train":
                loss_total.backward()
                self.optimizer.step()

            info = {
                'batch_loss'+mode: loss_total.item(),
                'batch_accuracy_classes'+mode: batch_accuracy_classes,
                'batch_accuracy_poses'+mode: batch_accuracy_poses,
                'loss_poses'+mode: loss_poses.item(),
                'batch_loss'+mode: loss_total.item(),
            }

            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iter * (len(dataset) // batch_size) + batch)
        epoch_accuracy_classes = np.mean(epoch_accuracy_classes)
        epoch_accuracy_poses = np.mean(epoch_accuracy_poses)

        info = {
            'Epoch_loss_'+mode: epoch_loss_total / ((len(dataset) // batch_size) * batch_size),
            'Epoch_accuracy_classes_'+mode: epoch_accuracy_classes,
            'Epoch_accuracy_poses_'+mode: epoch_accuracy_poses,
            'epoch_loss_classes_'+mode: epoch_loss_classes / ((len(dataset) // batch_size) * batch_size),
            'epoch_loss_poses_'+mode: epoch_loss_poses / ((len(dataset) // batch_size) * batch_size)
        }
        for i in range(len(dataset.word_encoding.classes)):
            info["Epoch_" + mode + "_" + dataset.word_encoding.get_class_char(i)] = np.mean(accuracy_each_class[i])

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, iter + 1)
        return epoch_accuracy_classes, epoch_accuracy_poses, epoch_loss_total



    def visualize_dataset(self, dataset=None):
        if dataset is None or dataset=="train":
            dataset = self.dataset
            print ("use train data to visualize")
        else:
            dataset = self.dataset_validation
            print ("use validation data to visualize")
        self.validation(1, 0, False, plot= True, dataset=dataset)

    def calculate_encoded_pose(self, word_encoded_pose):
        divide = torch.FloatTensor([[1 / constants.LOCAL_MAP_DIM, 1 / (constants.LOCAL_MAP_DIM / 2.0)]])
        divide = divide.repeat(5, 1)
        divide = divide.unsqueeze(0)
        addition = torch.FloatTensor([[0, 1]])
        addition = addition.repeat(5, 1)
        addition = addition.unsqueeze(0)
        divide2 = torch.FloatTensor([[1, 0.5]])
        divide2 = divide2.repeat(5, 1)
        divide2 = divide2.unsqueeze(0)
        word_encoded_pose = (word_encoded_pose * divide + addition) * divide2  # to be between 0-1
        word_encoded_pose = Variable(word_encoded_pose).cuda() if use_cuda else Variable(word_encoded_pose)
        return word_encoded_pose

    def validation(self, batch_size, iter, save, plot=False, dataset=None):
        if dataset is None:
            dataset = self.dataset_validation
        self.dataloader_validation = DataLoader(dataset, shuffle=True, num_workers=10, batch_size=batch_size, drop_last=True)

        epoch_accuracy_classes, epoch_accuracy_poses, epoch_loss_total = self.model_forward(batch_size, "validation", iter)

        print('validation loss: %.4f class_accuracy: %.4f pose_accuracy%.4f' %
              (epoch_loss_total / (len(self.dataset_validation) // batch_size), epoch_accuracy_classes, epoch_accuracy_poses))

        is_best = self.best_lost > epoch_loss_total
        print (is_best)
        if (save == True and plot == False):
            self.save_checkpoint({
                'epoch': iter + 1,
                'model_dict': self.model.state_dict(),
                'epoch_lost': epoch_loss_total,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)
        if is_best and plot == False:
            self.best_lost = epoch_loss_total

        print('validation acc: %f)' % epoch_accuracy_classes)


    def train_iters(self, n_iters, print_every=1000, plot_every=100, batch_size=100, save=True):
        self.dataloader = DataLoader(self.dataset, shuffle=True, num_workers=10, batch_size=batch_size, drop_last=True)
        self.start = time.time()
        n_iters = self.start_epoch + n_iters


        for iter in range(self.start_epoch, n_iters):

            epoch_accuracy_classes, epoch_accuracy_poses, epoch_loss_total = self.model_forward(batch_size, "train", iter)
            print('Iteration %s (%d %f%%) %.4f avg: %.4f acc:%.4f' %
                  (timeSince(self.start, (iter+1) / n_iters),
                  (iter + 1), (iter+1) / float(n_iters) * 100, epoch_loss_total,
                   epoch_loss_total/(len(self.dataset)//batch_size),epoch_accuracy_classes))
            print('acc: %f)' % epoch_accuracy_classes)

            self.validation(batch_size, iter, save)
