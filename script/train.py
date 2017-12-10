from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import collections
import os
import random
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

use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.conv = nn.Conv1d(input_size, hidden_size, 4)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 5)

        self.linear = nn.Linear(hidden_size*(90-3), self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        conv = self.conv(input.view(1, 1, 90))
        # conv = self.conv2(conv)
        linear = conv.view(1, 1, -1)
        output = self.linear(linear)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, dropout_p=0.15, max_length=MAX_LENGTH):
        super(DecoderRNN, self).__init__()
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


class Model:
    def __init__(self, dataset, save_model_path, model_ver=1, resume_path = None, teacher_forcing_ratio = 0.5, save=True):

        self.dataset = dataset
        plt.ion()
        hidden_size = 1500
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.save_path = save_model_path
        self.best_lost = sys.float_info.max
        self.encoder = EncoderRNN(1, hidden_size)
        if model_ver == 1:
            self.decoder = AttnDecoderRNN(hidden_size, self.dataset.lang.n_words,
                                          dropout_p=0.1, max_length=self.dataset._max_length_laser)
        if model_ver == 2:
            self.decoder = DecoderRNN(hidden_size, self.dataset.lang.n_words,
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

        if os.path.isfile(resume_path + "encoder"):
            self.encoder.load_state_dict(torch.load(resume_path + "encoder"))
            self.decoder.load_state_dict(torch.load(resume_path + "decoder"))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))
        print ("model version %d" %model_ver)
        self.trainIters(1000, model_ver=model_ver, print_every=10, save=save)

        self.dataset.shuffle_data()
        error = 0
        for i in range(2):
            error += self.evaluate(model_ver, self.dataset._max_length_laser, plot=True)

        print ("accu: {}", float(error)/100)
        raw_input("end")

    def train(self, input_variable, target_variable, criterion, model_ver=1,
              max_length=MAX_LENGTH):

        loss = 0
        target_length = target_variable.size()[0]
        input_length = input_variable.size()[0]

        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
                if ni == EOS_token:
                    break

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

    def adjust_learning_rate(self, optimizer, epoch, learning_rate):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        lr = learning_rate  * (0.96 ** (epoch // 2))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def trainIters(self, n_iters, model_ver=1, print_every=1000, plot_every=100, batch_size=10, learning_rate=0.001, save=True):
        start = time.time()
        plot_losses = []


        criterion = nn.NLLLoss()
        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        for iter in range(1, n_iters + 1):
            print_loss_total = 0
            plot_loss_total = 0
            # print("learning rate: {}".format(learning_rate))
            self.adjust_learning_rate(decoder_optimizer, iter, learning_rate)
            self.adjust_learning_rate(encoder_optimizer, iter, learning_rate)
            for param_group in decoder_optimizer.param_groups:
                print ("lr: ", param_group['lr'])
            self.dataset.shuffle_data()
            print (int(round(len(self.dataset.list_data)/batch_size)))
            for batch in range(1,int(round(len(self.dataset.list_data)/batch_size))+1):

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss = 0
                for d in range(1,batch_size+1):
                    training_pair = self.dataset.next_pair()
                    input_variable = training_pair[0]
                    target_variable = training_pair[1]

                    loss += self.train(input_variable, target_variable,
                                criterion,model_ver, self.dataset._max_length_laser)

                    print_loss_total += loss.data[0]/target_variable.size()[0]
                    plot_loss_total += loss.data[0]/target_variable.size()[0]


                print_loss_avg = print_loss_total / (batch_size*batch)
                print('epoch %d %s (%d %.2f%%) %.4f' % (iter, self.timeSince(start, batch / (len(self.dataset.list_data)/batch_size)),
                                             d, float(batch) / (len(self.dataset.list_data)/batch_size) * 100, print_loss_avg))
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                plot_loss = plot_loss_total
                plot_losses.append(plot_loss)
                plot_loss_total = 0

            print('%s (%d %f%%) %.4f avg: %.4f' % (self.timeSince(start, iter / n_iters),
                                         iter, iter / float(n_iters) * 100, print_loss_total, print_loss_total/len(self.dataset.list_data)))

            if (self.best_lost > print_loss_total and save == True):
                torch.save(self.encoder.state_dict(), self.save_path + str(iter) + str(print_loss_total) + "encoder")
                torch.save(self.decoder.state_dict(), self.save_path + str(iter) + str(print_loss_total) + "decoder")
                self.best_lost = print_loss_total
                torch.save(self.encoder.state_dict(), self.save_path + "_best_" + "encoder")
                torch.save(self.decoder.state_dict(), self.save_path + "_best_" + "decoder")

            if (iter % 10 == 1):
                self.dataset.shuffle_data()
                error = 0
                for i in range(100):
                    error += self.evaluate(model_ver, self.dataset._max_length_laser)
                print('acc: %f)' % (1- float(error)/100))


        self.showPlot(plot_losses)

    def showPlot(self, points):
        # plt.figure()
        # fig, ax = plt.subplots()
        # # this locator puts ticks at regular intervals
        # loc = ticker.MultipleLocator(base=0.2)
        # ax.yaxis.set_major_locator(loc)
        plt.plot(points)

    def evaluate(self, model_ver=1, max_length=MAX_LENGTH, plot=False):
        training_pair = self.dataset.next_pair()
        input_variable = training_pair[0]
        input_length = input_variable.size()[0]

        sentences = []
        expected_out = training_pair[1].data.cpu().numpy().reshape(-1)
        for word in  expected_out:
            sentences.append(self.dataset.lang.index2word[word])

        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        encoder_hidden = self.encoder.initHidden()
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            if (model_ver==1):
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
        corner = 0
        left = 0
        forward = 0
        two_way = 0
        streight = 0
        const = 1
        if (model_ver == 1 and plot==True):
            plt.matshow(decoder_attentions[:di + 1].numpy())
            fig = plt.figure()
            ax = fig.add_subplot(111)
            if (model_ver == 1):
                cax = ax.matshow(decoder_attentions[:di + 1].numpy(), cmap='bone')
                fig.colorbar(cax)

            plt.show()

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

        output_sentence = ' '.join(decoded_words)
        if corner != 0 or left != 0 or forward != 0 or two_way != 0 or streight != 0 or const != 1:
            print ("output: ", output_sentence)
            print ("target: ", " ".join(sentences))
            print ()
            return 1
        return 0


