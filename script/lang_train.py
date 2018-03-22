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

        self.linear = nn.Linear(hidden_size*(360-3), int(self.hidden_size*2/3))
        self.linear2 = nn.Linear(3, int(self.hidden_size/3))
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, laser_input, speed_input, hidden):
        noisy_laser_input = self.noise(laser_input)
        conv = self.conv(noisy_laser_input.view(1, 1, 360))
        laser_out = conv.view(1, 1, -1)
        laser_out = self.linear(laser_out)
        speed_out = self.linear2(speed_input.view(1, 1, -1))
        output = torch.cat((speed_out, laser_out),2)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNNMy(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, dropout_p=0.15, max_length=MAX_LENGTH):
        super(DecoderRNNMy, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.linear = nn.Linear(hidden_size*(max_length+1), self.hidden_size)

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        encoder_outputs = encoder_outputs.view(1, 1, -1)
        concat = torch.cat((embedded[0], encoder_outputs[0]), 1)

        output = concat.unsqueeze(0)
        output = self.linear(output)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, None

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, dropout_p=0.15, max_length=MAX_LENGTH):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.linear = nn.Linear(hidden_size*2, self.hidden_size)

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        encoder_outputs = encoder_outputs.view(1, 1, -1)
        concat = torch.cat((embedded[0], encoder_outputs[0]), 1)

        output = concat.unsqueeze(0)
        output = self.linear(output)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, None


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, dropout_p=0.15, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        concat = torch.cat((embedded[0], hidden[0]), 1)
        attention = self.attn(concat)

        attn_weights = F.softmax(attention)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



class AttnDecoderRNN_V2(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, dropout_p=0.15, max_length=MAX_LENGTH):
        super(AttnDecoderRNN_V2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * (2) + max_length*hidden_size, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        concat = torch.cat((embedded[0], hidden[0], encoder_outputs.view( 1, -1)), 1)

        attention = self.attn(concat)

        attn_weights = F.softmax(attention)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class Model:
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        filename = os.path.join(self.project_path,filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.project_path, 'model_best.pth.tar'))

    def __init__(self, robot, model_ver=1, resume_path = None, learning_rate = 0.001, load_weight = True, batch_size = 20, teacher_forcing_ratio = 0.5, save=True, number_of_iter=0, real_time_test=False, ):

        self.dataset = robot.dataset

        plt.ion()
        hidden_size = 150
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.learning_rate = learning_rate
        self.best_lost = sys.float_info.max

        # remove previous tensorboard data first
        import subprocess as sub
        p = sub.Popen(['rm', '-r', './logs'])
        p.communicate()
        self.logger = Logger('./logs')
        self.encoder = EncoderRNN(1, hidden_size)
        if model_ver == 1:
            self.decoder = AttnDecoderRNN(hidden_size, self.dataset.lang.n_words,
                                          dropout_p=0.1, max_length=self.dataset._max_length_laser)
        if model_ver == 2:
            self.decoder = DecoderRNNMy(hidden_size, self.dataset.lang.n_words,
                                          dropout_p=0.1, max_length=self.dataset._max_length_laser)
        if model_ver == 3:
            self.decoder = DecoderRNN(hidden_size, self.dataset.lang.n_words,
                                          dropout_p=0.1, max_length=self.dataset._max_length_laser)
        if model_ver == 4:
            self.decoder = AttnDecoderRNN_V2(hidden_size, self.dataset.lang.n_words,
                                          dropout_p=0.1, max_length=self.dataset._max_length_laser)

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

        print ("model version %d" %model_ver)
        self.trainIters(number_of_iter, model_ver=model_ver, print_every=10, save=save, batch_size=batch_size)

        self.dataset.shuffle_data()
        error = 0
        for i in range(len(self.dataset.list_data)):
            print ("in eval"+ str(i))
            error += self.evaluate_check(model_ver, self.dataset._max_length_laser, plot=True)
        accu = 1-float(error)/len(self.dataset.list_data)

        print ("accu: {} error{}".format(accu, error))




        # self.dataset.shuffle_data()
        # for i in range(10):
        #     self.evaluate_check(model_ver, self.dataset._max_length_laser, plot=False)
        # self.evaluate_check(model_ver, self.dataset._max_length_laser, plot=True)
        raw_input("end")

    def train(self, input_variable, target_variable, criterion, model_ver=1,
              max_length=MAX_LENGTH):

        target_length = target_variable.size()[0]
        input_length = input_variable[0].size()[0]

        encoder_hidden = self.encoder.initHidden()
        if model_ver == 3:
            encoder_outputs = Variable(torch.zeros(1, self.encoder.hidden_size))
        else:
            encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_variable[0][ei], input_variable[1][ei], encoder_hidden)
            if model_ver == 3:
                encoder_outputs = encoder_output[0][0]
            else:
                encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden.clone()

        # decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing
                outputs.append(decoder_output)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input
                outputs.append(decoder_output)
                # loss += criterion(decoder_output, target_variable[di])
                # if ni == EOS_token:
                #     break
        outputs = torch.stack(outputs).squeeze(1)
        target_variable = torch.stack(target_variable).squeeze(-1)
        loss = criterion(outputs, target_variable)



        # loss.backward()

        # return loss.data[0] / target_length
        return loss
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
    def trainIters(self, n_iters, model_ver=1, print_every=1000, plot_every=100, batch_size=10, save=True):
        start = time.time()
        plot_losses = []
        n_iters = self.start_epoch + n_iters
        criterion = nn.NLLLoss(size_average=False)

        for iter in range(self.start_epoch, n_iters):
            epoch_loss_total = 0
            # print("learning rate: {}".format(learning_rate))
            # self.adjust_learning_rate()
            # self.set_learning_rate(self.decoder_optimizer)
            # self.set_learning_rate(self.encoder_optimizer)
            for param_group in self.decoder_optimizer.param_groups:
                print ("lr: ", param_group['lr'])
            self.dataset.shuffle_data()
            print (int(round(len(self.dataset.list_data)/batch_size)))
            for batch in range (1,int((len(self.dataset.list_data)/batch_size))+1):

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                loss = 0
                for d in range(1,batch_size+1):
                    training_pair = self.dataset.next_pair()
                    input_variable = training_pair[0:2]
                    target_variable = training_pair[2]

                    loss += self.train(input_variable, target_variable,
                                criterion,model_ver, self.dataset._max_length_laser)

                    epoch_loss_total += loss.data[0]/target_variable.size()[0]

                loss = loss/batch_size
                print_loss_avg = epoch_loss_total / (batch_size*batch)
                print('Epoch %d %s (%d %.2f%%) %.4f' % (iter+1, self.timeSince(start, batch / (len(self.dataset.list_data)/batch_size)),
                                             d, float(batch) / (len(self.dataset.list_data)/batch_size) * 100, print_loss_avg))
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                info = {
                    'batch_loss': loss.data[0],
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iter*batch_size+ batch )


            print('Iteration %s (%d %f%%) %.4f avg: %.4f' %
                  (self.timeSince(start, (iter+1) / n_iters),
                  (iter + 1), (iter+1) / float(n_iters) * 100, epoch_loss_total,
                   epoch_loss_total/((len(self.dataset.list_data)/batch_size)*batch_size)))

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


            if (iter % 1 == 0):
                self.dataset.shuffle_data()
                error = 0
                for i in range(int (len(self.dataset.list_data))-1):
                    error += self.evaluate_check(model_ver, self.dataset._max_length_laser)
                self.acc = 1- float(error)/((len(self.dataset.list_data))-1)

                print('acc: %f)' % self.acc)
                self.evaluate_check(model_ver, self.dataset._max_length_laser, plot=True)
            info = {
                'epoch_loss': epoch_loss_total/((len(self.dataset.list_data)/batch_size)*batch_size),
                'epoch_acc': self.acc
            }

            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iter + 1)
            self.prev_loss = self.loss
            self.loss = epoch_loss_total

    def showPlot(self, points):
        # plt.figure()
        # fig, ax = plt.subplots()
        # # this locator puts ticks at regular intervals
        # loc = ticker.MultipleLocator(base=0.2)
        # ax.yaxis.set_major_locator(loc)
        plt.plot(points)

    def evaluate(self, training_pair, model_ver=1, max_length=MAX_LENGTH, plot=False):
        input_variable = training_pair[0]
        input_length = input_variable.size()[0]

        if model_ver == 3:
            encoder_outputs = Variable(torch.zeros(1, self.encoder.hidden_size))
        else:
            encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        encoder_hidden = self.encoder.initHidden()
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_variable[ei], training_pair[1][ei], encoder_hidden)
            if model_ver == 3:
                encoder_outputs = encoder_output[0][0]
            else:
                encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            if (model_ver == 1 or model_ver == 4):
                decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.dataset.lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        # if ((model_ver == 1 or model_ver==4) and plot==True):
        #     ax = plt.matshow(decoder_attentions[:di + 1].numpy())
        #     plt.colorbar(ax)
        #     # fig = plt.figure()
        #
        #     # ax = fig.add_subplot(111)
        #     # cax = ax.matshow(decoder_attentions[:di + 1].numpy(), cmap='bone')
        #     # fig.colorbar(cax)
        #
        #     plt.show()
        return decoded_words

    def evaluate_check(self, model_ver=1, max_length=MAX_LENGTH, plot=False):
        training_pair = self.dataset.next_pair()
        sentences = []
        expected_out = training_pair[2].data.cpu().numpy().reshape(-1)
        for word in expected_out:
            sentences.append(self.dataset.lang.index2word[word])
        sentences = sentences[0:-1]
        decoded_words = self.evaluate(training_pair, model_ver, max_length, plot)

        corner = 0
        left = 0
        forward = 0
        two_way = 0
        streight = 0
        const = 1

        for words in [decoded_words, sentences]:
            const *= -1
            for w in words:
                if w == "continue":
                    streight += const
                    break
                if w == "left":
                    left += const
                if w == "right":
                    left -= const
                if w == "forward":
                    forward += const
                if w == "two":
                    two_way += const
                if w == "one" or w == "no":
                    two_way -= const
                if w == "intersection":
                    corner -= const
                if w == "corner":
                    corner += const

        output_sentence = ' '.join(decoded_words[0:-1])
        sentences = " ".join(sentences)
        if (output_sentence!=sentences):
            print("output: ", output_sentence)
            print("target: ", sentences)
            return 1
        if (corner != 0 or left != 0 or forward != 0 or two_way != 0 or streight != 0 or const != 1):
            if plot == True:
                print ("output: ", output_sentence)
                print ("target: ", sentences)
                print ()
            return 1
        return 0


