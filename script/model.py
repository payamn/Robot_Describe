from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import collections
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
    def __init__(self, input_size, output_size, max_length=MAX_LENGTH, num_of_prediction=5, training=True, batch_size = 1):
        super(Network_Map, self).__init__()
        self.input_size = input_size
        self.numb_of_prediction = num_of_prediction
        self.conv = nn.Conv2d(input_size, 256, 7)
        self.conv2 = nn.Conv2d(256, 128, 6, stride=2)
        self.conv3 = nn.Conv2d(128, 64, 5, stride=2)
        self.conv4 = nn.Conv2d(64, 32, 4, stride=1)
        self.conv5 = nn.Conv2d(32, 16, 3, stride=2)
        self.conv6 = nn.Conv2d(16, 16, 2, stride=2)

        self.output_size = output_size
        self.linear = nn.Linear(32 * 7 * 7, 16)
        self.linear2 = nn.Linear(16,  2 * num_of_prediction)
        self.linear3 = nn.Linear(16, self.output_size * num_of_prediction)
        self.soft_max = nn.LogSoftmax(dim=3)

    def forward(self, map):
        batch_size = map.size()[0]
        conv = self.conv(map)
        conv2 = self.conv2(conv)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5).view(batch_size, 1, -1)
        linear = self.linear(conv4.view(batch_size, 1, -1))
        poses_out = self.linear2(linear)
        classes_out = self.linear3(conv6)
        classes_out = self.soft_max(classes_out.view(batch_size, 1, self.numb_of_prediction, self.output_size))
        return poses_out.view(batch_size, self.numb_of_prediction, 2), classes_out


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

    def __init__(self, dataset_train, dataset_validation, resume_path = None, learning_rate = 0.001, load_weight = True, batch_size = 20, save=True, number_of_iter=0, real_time_test=False):

        self.dataset = dataset_train
        self.dataset_validation = dataset_validation
        self.dataloader = DataLoader(self.dataset, shuffle=True, num_workers=10, batch_size=batch_size, drop_last=True)
        self.dataloader_validation = DataLoader(dataset_validation, shuffle=True, num_workers=10, batch_size=batch_size, drop_last=True)
        print ("train: ", len(self.dataset))
        print (len(self.dataloader))
        print ("test: ", len(self.dataloader_validation))
        self.learning_rate = learning_rate
        self.best_lost = sys.float_info.max
        self.distance = nn.PairwiseDistance()
        # remove previous tensorboard data first
        import subprocess as sub
        p = sub.Popen(['rm', '-r', './logs'])
        p.communicate()
        self.logger = Logger('./logs')
        self.model = Network_Map(1, self.dataset.word_encoding.len_classes(), batch_size = batch_size)
        if use_cuda:
            self.model = self.model.cuda()

        print("model size:")
        for parameter in self.model.parameters():
            print(parameter.size())

        self.prev_loss = None
        self.loss = None
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
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume_path))

        self.train_iters(number_of_iter, print_every=10, save=save, batch_size=batch_size)

    def validation(self, batch_size, iter, save):
        criterion_classes = nn.NLLLoss()
        criterion_poses = nn.MSELoss()

        iter_data_loader = self.dataloader_validation.__iter__()
        epoch_loss_total = 0
        epoch_loss_classes = 0
        epoch_loss_poses = 0
        epoch_accuracy_classes = []
        epoch_accuracy_poses = []
        for batch, (word_encoded_class, word_encoded_pose, local_map) in enumerate(self.dataloader_validation):
            loss_classes = 0
            loss_poses = 0
            batch_accuracy_classes = []
            batch_accuracy_poses = []
            local_map = Variable(local_map).cuda() if use_cuda else Variable(local_map)
            local_map = local_map.unsqueeze(1)
            word_encoded_class = Variable(word_encoded_class).cuda() if use_cuda else Variable(word_encoded_class)
            word_encoded_pose = word_encoded_pose / 6 + 1 / 2  # to be between 0-1
            word_encoded_pose = Variable(word_encoded_pose).cuda() if use_cuda else Variable(word_encoded_pose)

            output_poses, output_classes = self.model(local_map)

            topv, topi = output_classes.data.topk(1)
            # print (topi)
            # print (word_encoded_class)
            # print (output_poses)
            # print (word_encoded_pose)
            accuracy = 0
            for i in range(word_encoded_class.size(0)):
                for j in range(self.dataset.prediction_number):
                    if word_encoded_class.data[i][j] != 8:
                        accuracy = float(word_encoded_class[i][j] == topi[i][0][j][0])
                        batch_accuracy_classes.append(accuracy)
                        epoch_accuracy_classes.append(accuracy)
                distance = self.distance(word_encoded_pose[i], output_poses[i])
                distance = distance.squeeze(1)
                accuracy_poses = torch.mean(distance)
                batch_accuracy_poses.append(accuracy_poses)
                epoch_accuracy_poses.append(accuracy_poses)
            output_classes = output_classes.permute(0, 3, 1, 2)
            word_encoded_class = word_encoded_class.unsqueeze(1)
            # output_classes = output_classes.view(output_classes.size(2), output_classes.size(3))
            # output_poses = output_poses.view(output_poses.size(0), output_poses.size(2), output_poses.size(3))
            # word_encoded_class = word_encoded_class.squeeze(0)

            # for i in range (output_classes.size(1)):
            loss_classes += criterion_classes(output_classes, word_encoded_class)
            loss_poses += criterion_poses(output_poses, word_encoded_pose)

            loss_total = loss_poses + 10 * loss_classes

            epoch_loss_classes += loss_classes.data[0] / word_encoded_class.size()[0]
            epoch_loss_poses += loss_poses.data[0] / word_encoded_class.size()[0]
            epoch_loss_total += epoch_loss_classes + epoch_loss_poses


        epoch_accuracy_classes = np.mean(epoch_accuracy_classes)
        epoch_accuracy_poses = np.mean(epoch_accuracy_poses)

        print('validation loss: %.4f class_accuracy: %.4f pose_accuracy%.4f' %
              (epoch_loss_total / (len(self.dataset_validation) // batch_size), epoch_accuracy_classes, epoch_accuracy_poses))

        is_best = self.best_lost > epoch_loss_total
        if (self.best_lost > epoch_loss_total and save == True):
            self.save_checkpoint({
                'epoch': iter + 1,
                'model_dict': self.model.state_dict(),
                'epoch_lost': epoch_loss_total,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)
        if is_best:
            self.best_lost = epoch_loss_total

        # if (iter % 1 == 0):
        #     self.dataset.shuffle_data()
        #     error = 0
        #     for i in range(int (len(self.dataset.list_data))-1):
        #         error += self.evaluate_check(self.dataset._max_length_laser)
        #     self.acc = 1- float(error)/((len(self.dataset.list_data))-1)
        info = {
            'Validation_loss': epoch_loss_total / ((len(self.dataset_validation) // batch_size) * batch_size),
            'Validation_accuracy_classes': epoch_accuracy_classes,
            'Validation_accuracy_poses': epoch_accuracy_poses.data[0],
            'Validation_loss_classes': epoch_loss_classes / ((len(self.dataset_validation) // batch_size) * batch_size),
            'Validation_loss_poses': epoch_loss_poses / ((len(self.dataset_validation) // batch_size) * batch_size)
        }
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, iter + 1)
        print('acc: %f)' % epoch_accuracy_classes)
        self.prev_loss = self.loss
        self.loss = epoch_loss_total

    def train_iters(self, n_iters, print_every=1000, plot_every=100, batch_size=10, save=True):
        start = time.time()
        n_iters = self.start_epoch + n_iters
        criterion_classes = nn.NLLLoss()
        criterion_poses = nn.MSELoss()

        for iter in range(self.start_epoch, n_iters):
            iter_data_loader = self.dataloader.__iter__()
            epoch_loss_total = 0
            epoch_loss_classes = 0
            epoch_loss_poses = 0
            epoch_accuracy_classes = []
            epoch_accuracy_poses = []
            for batch , (word_encoded_class, word_encoded_pose, local_map)in enumerate(self.dataloader):
                loss_classes = 0
                loss_poses = 0
                # for batch_index in range (batch_size):
                self.optimizer.zero_grad()
                batch_accuracy_classes = []
                batch_accuracy_poses = []
                local_map = Variable(local_map).cuda() if use_cuda else Variable(local_map)
                local_map = local_map.unsqueeze(1)
                word_encoded_class = Variable(word_encoded_class).cuda() if use_cuda else Variable(word_encoded_class)
                word_encoded_pose = word_encoded_pose/6 + 1/2 # to be between 0-1
                word_encoded_pose = Variable(word_encoded_pose).cuda() if use_cuda else Variable(word_encoded_pose)

                output_poses, output_classes = self.model(local_map)

                topv, topi = output_classes.data.topk(1)
                # print (topi)
                # print (word_encoded_class)
                # print (output_poses)
                # print (word_encoded_pose)
                accuracy = 0
                for i in range (word_encoded_class.size(0)):
                    for j in range (self.dataset.prediction_number):
                        if word_encoded_class.data[i][j] != 8:
                            accuracy = float(word_encoded_class[i][j] == topi[i][0][j][0])
                            batch_accuracy_classes.append(accuracy)
                            epoch_accuracy_classes.append(accuracy)
                    distance = self.distance(word_encoded_pose[i], output_poses[i])
                    distance = distance.squeeze(1)
                    accuracy_poses = torch.mean(distance)
                    batch_accuracy_poses.append(accuracy_poses)
                    epoch_accuracy_poses.append(accuracy_poses)
                output_classes = output_classes.permute(0,3,1,2)
                word_encoded_class = word_encoded_class.unsqueeze(1)
                # output_classes = output_classes.view(output_classes.size(2), output_classes.size(3))
                # output_poses = output_poses.view(output_poses.size(0), output_poses.size(2), output_poses.size(3))
                # word_encoded_class = word_encoded_class.squeeze(0)

                # for i in range (output_classes.size(1)):
                loss_classes += criterion_classes(output_classes, word_encoded_class)
                loss_poses += criterion_poses(output_poses, word_encoded_pose)

                loss_total = loss_poses + 3* loss_classes

                epoch_loss_classes += loss_classes.data[0]/word_encoded_class.size()[0]
                epoch_loss_poses += loss_poses.data[0]/word_encoded_class.size()[0]
                epoch_loss_total += epoch_loss_classes + epoch_loss_poses
                batch_accuracy_classes = np.mean(batch_accuracy_classes)
                batch_accuracy_poses = np.mean(batch_accuracy_poses)
                # loss = loss/batch_size
                print_loss_avg = epoch_loss_total / (batch_size*(batch+1))
                print('Epoch %d %s (%.2f%%) %.4f acc_classes:%.4f acc_posses:%.4f' % (iter+1, timeSince(start, (batch+1) / (len(self.dataset)//batch_size))
                                            , float((batch+1)) / (len(self.dataset)//batch_size) * 100, print_loss_avg, batch_accuracy_classes, batch_accuracy_poses))
                loss_total.backward()
                self.optimizer.step()

                info = {
                    'batch_loss': loss_total.data[0],
                    'batch_accuracy_classes': batch_accuracy_classes,
                    'batch_accuracy_poses': batch_accuracy_poses.data[0],
                    'loss_poses': loss_poses.data[0],
                    'batch_loss': loss_total.data[0],
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iter*(len(self.dataset)//batch_size)+ batch )


            epoch_accuracy_classes = np.mean(epoch_accuracy_classes)
            epoch_accuracy_poses = np.mean(epoch_accuracy_poses)

            print('Iteration %s (%d %f%%) %.4f avg: %.4f acc:%.4f' %
                  (timeSince(start, (iter+1) / n_iters),
                  (iter + 1), (iter+1) / float(n_iters) * 100, epoch_loss_total,
                   epoch_loss_total/(len(self.dataset)//batch_size),epoch_accuracy_classes))

            # is_best = self.best_lost > epoch_loss_total
            # if (self.best_lost > epoch_loss_total and save == True):
            #     self.save_checkpoint({
            #         'epoch': iter + 1,
            #         'model_dict': self.model.state_dict(),
            #         'epoch_lost': epoch_loss_total,
            #         'optimizer': self.optimizer.state_dict(),
            #     }, is_best)
            # if is_best:
            #     self.best_lost = epoch_loss_total

            self.validation(batch_size, iter, save)
            # if (iter % 1 == 0):
            #     self.dataset.shuffle_data()
            #     error = 0
            #     for i in range(int (len(self.dataset.list_data))-1):
            #         error += self.evaluate_check(self.dataset._max_length_laser)
            #     self.acc = 1- float(error)/((len(self.dataset.list_data))-1)
            info = {
                'Epoch_loss':  epoch_loss_total/((len(self.dataset)//batch_size)*batch_size),
                'Epoch_accuracy_classes': epoch_accuracy_classes,
                'Epoch_accuracy_poses': epoch_accuracy_poses.data[0],
                'epoch_loss_classes': epoch_loss_classes/((len(self.dataset)//batch_size)*batch_size),
                'epoch_loss_poses': epoch_loss_poses/((len(self.dataset)//batch_size)*batch_size)
            }
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iter + 1)
            print('acc: %f)' % epoch_accuracy_classes)
            self.prev_loss = self.loss
            self.loss = epoch_loss_total


class Model:
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.project_path,filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.project_path, 'model_best.pth.tar'))

    def __init__(self, dataset, resume_path = None, learning_rate = 0.001, load_weight = True, batch_size = 20, save=True, number_of_iter=0, real_time_test=False):

        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, shuffle=True, num_workers=10)

        plt.ion()
        hidden_size = 500
        self.learning_rate = learning_rate
        self.best_lost = sys.float_info.max
        self.distance = nn.PairwiseDistance()
        # remove previous tensorboard data first
        import subprocess as sub
        p = sub.Popen(['rm', '-r', './logs'])
        p.communicate()
        self.logger = Logger('./logs')
        self.encoder = Encoder_RNN_Laser(1, hidden_size, batch_size=batch_size)
        self.decoder = DecoderNoRNN(hidden_size, self.dataset.word_encoding.len_classes(),
                                       max_length=self.dataset.laser_window)

        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        print("encoder size:")
        for parameter in self.decoder.parameters():
            print(parameter.size())
        print("decoder size:")
        for parameter in self.decoder.parameters():
            print(parameter.size())
        self.prev_loss = None
        self.loss = None
        self.start_epoch = 0
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
        self.project_path = "."
        if resume_path is not None and load_weight:
            self.project_path = os.path.dirname(resume_path)
            if os.path.isfile(resume_path):
                print("=> loading checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path)
                self.start_epoch = checkpoint['epoch']
                self.best_lost = checkpoint['epoch_lost']
                self.encoder.load_state_dict(checkpoint['encoder_dict'])
                self.decoder.load_state_dict(checkpoint['decoder_dict'])
                self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
                self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume_path, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume_path))

        self.train_iters(number_of_iter, print_every=10, save=save, batch_size=batch_size)

        # error = 0
        # for i in range(len(self.dataset.list_data)):
        #     print ("in eval"+ str(i))
        #     error += self.evaluate_check(self.dataset._max_length_laser, plot=True)
        # accu = 1-float(error)/len(self.dataset.list_data)
        #
        # print ("accu: {} error{}".format(accu, error))
        # raw_input("end")

    def train(self, speeds, laser_data, max_length=MAX_LENGTH):

        input_length = speeds.size()[1]
        batch_size = speeds.size()[0]
        encoder_hidden = self.encoder.initHidden(batch_size = batch_size)
        encoder_outputs = Variable(torch.zeros(batch_size, max_length, self.encoder.hidden_size))

        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(speeds[:,ei], laser_data[:,ei], encoder_hidden)
            encoder_outputs[:,ei] = encoder_output[:,0].clone()

        decoder_output = self.decoder(encoder_hidden, encoder_outputs)

        return decoder_output


    def adjust_learning_rate(self):
        if self.prev_loss and self.loss and self.prev_loss <= self.loss:
            self.learning_rate = self.learning_rate * 0.96
            print ("loss didn't improve so learning rate decreasing to {}".format(self.learning_rate))


    def set_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.learning_rate
    def train_iters(self, n_iters, print_every=1000, plot_every=100, batch_size=10, save=True):
        start = time.time()
        n_iters = self.start_epoch + n_iters
        criterion_classes = nn.NLLLoss(size_average=False)
        criterion_poses = nn.MSELoss(size_average=False)

        for iter in range(self.start_epoch, n_iters):
            iter_data_loader = self.dataloader.__iter__()
            epoch_loss_total = 0
            epoch_loss_classes = 0
            epoch_loss_poses = 0
            for param_group in self.decoder_optimizer.param_groups:
                print ("lr: ", param_group['lr'])
            epoch_accuracy_classes = []
            epoch_accuracy_poses = []
            for batch in range (0, len(self.dataloader)//batch_size):
                loss_classes = 0
                loss_poses = 0
                for batch_index in range (batch_size):
                    word_encoded_class, word_encoded_pose, speeds, laser_scans = iter_data_loader.next()
                    self.encoder_optimizer.zero_grad()
                    self.decoder_optimizer.zero_grad()
                    batch_accuracy_classes = []
                    batch_accuracy_poses = []
                    speeds = Variable(speeds).cuda() if use_cuda else Variable(speeds)
                    laser_scans = Variable(laser_scans).cuda() if use_cuda else Variable(laser_scans)
                    word_encoded_class = Variable(word_encoded_class).cuda() if use_cuda else Variable(word_encoded_class)
                    word_encoded_pose = word_encoded_pose/6 + 1/2 # to be between 0-1
                    word_encoded_pose = Variable(word_encoded_pose).cuda() if use_cuda else Variable(word_encoded_pose)

                    output_classes, output_poses = self.train(speeds, laser_scans, self.dataset.laser_window)

                    topv, topi = output_classes.data.topk(1)
                    # print (topi)
                    # print (word_encoded_class)
                    # print (output_poses)
                    # print (word_encoded_pose)
                    accuracy = 0
                    for i in range (word_encoded_class.size(0)):
                        for j in range (self.dataset.prediction_number):
                            if word_encoded_class.data[i][j] != 8:
                                accuracy = float(word_encoded_class[i][j] == topi[i][0][j][0])
                                batch_accuracy_classes.append(accuracy)
                                epoch_accuracy_classes.append(accuracy)
                        distance = self.distance(word_encoded_pose[i], output_poses[i])
                        distance = distance.squeeze(1)
                        accuracy_poses = torch.mean(distance)
                        batch_accuracy_poses.append(accuracy_poses)
                        epoch_accuracy_poses.append(accuracy_poses)
                    # output_classes = output_classes.permute(0,3,1,2)
                    # word_encoded_class = word_encoded_class.unsqueeze(1)
                    output_classes = output_classes.view(output_classes.size(2), output_classes.size(3))
                    # output_poses = output_poses.view(output_poses.size(0), output_poses.size(2), output_poses.size(3))
                    word_encoded_class = word_encoded_class.squeeze(0)

                    # for i in range (output_classes.size(1)):
                    loss_classes += criterion_classes(output_classes, word_encoded_class)
                    loss_poses += criterion_poses(output_poses, word_encoded_pose)

                loss_total = loss_poses + loss_classes

                epoch_loss_classes += loss_classes.data[0]/word_encoded_class.size()[0]
                epoch_loss_poses += loss_poses.data[0]/word_encoded_class.size()[0]
                epoch_loss_total += epoch_loss_classes + epoch_loss_poses
                batch_accuracy_classes = np.mean(batch_accuracy_classes)
                batch_accuracy_poses = np.mean(batch_accuracy_poses)
                # loss = loss/batch_size
                len_data = len(self.dataloader)
                print_loss_avg = epoch_loss_total / (batch_size*(batch+1))
                print('Epoch %d %s (%.2f%%) %.4f acc_classes:%.4f acc_posses:%.4f' % (iter+1, timeSince(start, (batch+1) / (len(self.dataloader)//batch_size))
                                            , float((batch+1)) / (len(self.dataloader)//batch_size) * 100, print_loss_avg, batch_accuracy_classes, batch_accuracy_poses))
                loss_total.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                info = {
                    'batch_loss': loss_total.data[0],
                    'batch_accuracy_classes': batch_accuracy_classes,
                    'batch_accuracy_poses': batch_accuracy_poses.data[0],
                    'loss_poses': loss_poses.data[0],
                    'batch_loss': loss_total.data[0],
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iter*(len(self.dataloader)//batch_size)*batch_size+ batch )


            epoch_accuracy_classes = np.mean(epoch_accuracy_classes)
            epoch_accuracy_poses = np.mean(epoch_accuracy_poses)

            print('Iteration %s (%d %f%%) %.4f avg: %.4f acc:%.4f' %
                  (timeSince(start, (iter+1) / n_iters),
                  (iter + 1), (iter+1) / float(n_iters) * 100, epoch_loss_total,
                   epoch_loss_total/(len(self.dataloader)//batch_size),epoch_accuracy_classes))

            is_best = self.best_lost > epoch_loss_total
            if (self.best_lost > epoch_loss_total and save == True):
                self.save_checkpoint({
                    'epoch': iter + 1,
                    'encoder_dict': self.encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'epoch_lost': epoch_loss_total,
                    'encoder_optimizer': self.encoder_optimizer.state_dict(),
                    'decoder_optimizer': self.decoder_optimizer.state_dict(),
                }, is_best)
            if is_best:
                self.best_lost = epoch_loss_total


            # if (iter % 1 == 0):
            #     self.dataset.shuffle_data()
            #     error = 0
            #     for i in range(int (len(self.dataset.list_data))-1):
            #         error += self.evaluate_check(self.dataset._max_length_laser)
            #     self.acc = 1- float(error)/((len(self.dataset.list_data))-1)
            info = {
                'Epoch_loss':  epoch_loss_total/((len(self.dataloader)//batch_size)*batch_size),
                'Epoch_accuracy_classes': epoch_accuracy_classes,
                'Epoch_accuracy_poses': epoch_accuracy_poses.data[0],
                'epoch_loss_classes': epoch_loss_classes/((len(self.dataloader)//batch_size)*batch_size),
                'epoch_loss_poses': epoch_loss_poses/((len(self.dataloader)//batch_size)*batch_size)
            }
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iter + 1)
            print('acc: %f)' % epoch_accuracy_classes)
            self.prev_loss = self.loss
            self.loss = epoch_loss_total

    def evaluate(self, training_pair,  max_length=MAX_LENGTH):
        input_variable = training_pair[0]
        input_length = input_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        encoder_hidden = self.encoder.initHidden()
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_variable[ei], training_pair[1][ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        # TODO: Run network and get result

        return decoded_words

    def evaluate_check(self, max_length=MAX_LENGTH):
        training_pair = self.dataset.next_pair()
        sentences = []
        expected_out = training_pair[2].data.cpu().numpy().reshape(-1)
        for word in expected_out:
            sentences.append(self.dataset.lang.index2word[word])
        sentences = sentences[0:-1]
        decoded_words = self.evaluate(training_pair, max_length)

        output_sentence = ' '.join(decoded_words[0:-1])
        sentences = " ".join(sentences)
        if (output_sentence!=sentences):
            print("output: ", output_sentence)
            print("target: ", sentences)
            return 1

        return 0


