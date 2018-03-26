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

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.05, training=True):
        super(DynamicGNoise, self).__init__()
        self.training = training
        self.noise = Variable(torch.zeros(shape).cuda())
        self.std = std

    def forward(self, x):
        if not self.training: return x
        self.noise.data.normal_(0, std=self.std)
        noisy_out = x - self.noise
        noisy_out.data.clamp_(0,1)
        return (noisy_out)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, training=True):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.noise = DynamicGNoise(360, 0.05, training=training)
        self.conv = nn.Conv1d(input_size, hidden_size, 4)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 5)

        self.linear = nn.Linear(hidden_size*(360-3), int(self.hidden_size))
        self.linear2 = nn.Linear(3, int(self.hidden_size/4))
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, laser_input, speed_input, hidden):
        batch_size = laser_input.size()[0]
        noisy_laser_input = self.noise(laser_input)
        conv = self.conv(noisy_laser_input.view(batch_size, 1, 360))
        laser_out = self.linear(conv.view(batch_size, 1, -1))
        # speed_input = speed_input.unsqueeze(1)
        # speed_out = self.linear2(speed_input)
        # output = torch.cat((speed_out, laser_out),2)
        output = laser_out
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderNoRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH):
        super(DecoderNoRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.linear = nn.Linear(hidden_size*(max_length+1), self.output_size)

        # self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.soft_max = nn.LogSoftmax(dim=2)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.view(encoder_outputs.size()[0], 1, -1)
        hidden = torch.transpose(hidden, 0, 1)
        output = torch.cat((hidden, encoder_outputs), -1)

        # output = concat.unsqueeze(0)
        output = self.linear(output)
        output = self.soft_max(output)
        return output

class Model:
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.project_path,filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.project_path, 'model_best.pth.tar'))

    def __init__(self, robot, resume_path = None, learning_rate = 0.001, load_weight = True, batch_size = 20, save=True, number_of_iter=0, real_time_test=False):

        self.dataset = robot.dataset

        plt.ion()
        hidden_size = 500
        self.learning_rate = learning_rate
        self.best_lost = sys.float_info.max

        # remove previous tensorboard data first
        import subprocess as sub
        p = sub.Popen(['rm', '-r', './logs'])
        p.communicate()
        self.logger = Logger('./logs')
        self.encoder = EncoderRNN(1, hidden_size)
        self.decoder = DecoderNoRNN(hidden_size, self.dataset.lang.n_words,
                                       max_length=self.dataset._max_length_laser)

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

        self.train_iters(number_of_iter, print_every=10, save=save, batch_size=self.dataset.batch_size)

        self.dataset.shuffle_data()
        error = 0
        for i in range(len(self.dataset.list_data)):
            print ("in eval"+ str(i))
            error += self.evaluate_check(self.dataset._max_length_laser, plot=True)
        accu = 1-float(error)/len(self.dataset.list_data)

        print ("accu: {} error{}".format(accu, error))
        raw_input("end")

    def train(self, input_variable, target_variable, max_length=MAX_LENGTH):

        target_length = target_variable.size()[1]
        input_length = input_variable[0].size()[1]
        batch_size = input_variable[0].size()[0]
        encoder_hidden = self.encoder.initHidden(batch_size = batch_size)
        encoder_outputs = Variable(torch.zeros(batch_size, max_length, self.encoder.hidden_size))

        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[0][:,ei], input_variable[1][:,ei], encoder_hidden)
            encoder_outputs[:,ei] = encoder_output[:,0].clone()

        decoder_output = self.decoder(encoder_hidden, encoder_outputs)

        # outputs = torch.stack(outputs).squeeze(1)
        # target_variable = torch.stack(target_variable).squeeze(-1)
        decoder_output = decoder_output.squeeze(0)
        return decoder_output



        # loss.backward()

        # return loss.data[0] / target_length
        # return loss
    @staticmethod
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))

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
        criterion = nn.NLLLoss()

        for iter in range(self.start_epoch, n_iters):
            epoch_loss_total = 0
            for param_group in self.decoder_optimizer.param_groups:
                print ("lr: ", param_group['lr'])
            self.dataset.shuffle_data()
            epoch_accuracy = []
            print (int(round(len(self.dataset.list_data)/batch_size)))
            for batch in range (1,int((len(self.dataset.list_data)/batch_size))+1):

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                batch_accuracy = []
                training_pair =  self.dataset.next_batch()
                input_variable = training_pair[0:2]
                target_variable = training_pair[2]
                output = self.train(input_variable, target_variable, self.dataset._max_length_laser)
                # for d in range(1,batch_size+1):
                #     training_pair = self.dataset.next_pair()
                #     input_variable = training_pair[0:2]
                #     target_variable = training_pair[2]
                #
                #     output = self.train(input_variable, target_variable, self.dataset._max_length_laser)
                topv, topi = output.data.topk(1)
                accuracy = 0
                for i in range (target_variable.size(0)):
                    accuracy =  float(target_variable.data[i][0] == topi[i][0][0])
                    batch_accuracy.append(accuracy)
                    epoch_accuracy.append(accuracy)
                output = output.view(batch_size, output.size(2))
                loss = criterion(output, target_variable.view(batch_size))
                epoch_loss_total += loss.data[0]/target_variable.size()[0]
                batch_accuracy = np.mean(batch_accuracy)
                # loss = loss/batch_size
                print_loss_avg = epoch_loss_total / (batch_size*batch)
                print('Epoch %d %s (%.2f%%) %.4f acc:%.4f' % (iter+1, self.timeSince(start, batch / (len(self.dataset.list_data)/batch_size))
                                            , float(batch) / (len(self.dataset.list_data)/batch_size) * 100, print_loss_avg, batch_accuracy))
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                info = {
                    'batch_loss': loss.data[0],
                    'accuracy':batch_accuracy
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iter*batch_size+ batch )


            epoch_accuracy = np.mean(epoch_accuracy)

            print('Iteration %s (%d %f%%) %.4f avg: %.4f acc:%.4f' %
                  (self.timeSince(start, (iter+1) / n_iters),
                  (iter + 1), (iter+1) / float(n_iters) * 100, epoch_loss_total,
                   epoch_loss_total/((len(self.dataset.list_data)/batch_size)*batch_size),epoch_accuracy))

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
                'Epoch_loss':  epoch_loss_total/((len(self.dataset.list_data)/batch_size)*batch_size),
                'Epoch_accuracy': epoch_accuracy
            }
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iter + 1)
            print('acc: %f)' % epoch_accuracy)
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


