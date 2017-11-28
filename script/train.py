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
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = 1
        self.conv = nn.Conv1d(input_size, hidden_size, 5)
        self.linear = nn.Linear(hidden_size*(90-4), self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        conv = self.conv(input.view(1, 1, 90))
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


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.     hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
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
    def __init__(self, dataset, save_model_path, resume_path = None, teacher_forcing_ratio = 0.5):
        self.dataset = dataset

        hidden_size = 1024
        self.encoder= EncoderRNN(1, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, self.dataset.lang.n_words,                                           1, dropout_p=0.1, max_length=self.dataset._max_length_laser)
        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.save_path = save_model_path
        self.best_lost = sys.float_info.max
        print ("encoder size:")
        for parameter in self.decoder.parameters():
            print (parameter.size())
        print ("decoder size:")
        for parameter in self.decoder.parameters():
            print (parameter.size())
        print (use_cuda)

        if os.path.isfile(resume_path+"encoder"):
           self.encoder.load_state_dict (torch.load(resume_path+"encoder"))
           self.decoder.load_state_dict (torch.load(resume_path+"decoder"))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))
        self.trainIters(9000, print_every=10 )

    def train(self, input_variable, target_variable, encoder_optimizer, decoder_optimizer, criterion,
              max_length=MAX_LENGTH):
        encoder_hidden = self.encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        target_length = target_variable.size()[0]
        input_length = input_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0
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
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
                if ni == EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_length

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

    def trainIters(self, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)

        criterion = nn.NLLLoss()

        for iter in range(1, n_iters + 1):
            self.dataset.shuffle_data()
            for d in range(1, len(self.dataset.list_data)+1):
                training_pair = self.dataset.next_pair()
                input_variable = training_pair[0]
                target_variable = training_pair[1]

                loss = self.train(input_variable, target_variable,
                            encoder_optimizer, decoder_optimizer, criterion, self.dataset._max_length_laser)
                print_loss_total += loss
                plot_loss_total += loss

                if d % print_every == 0:
                    print_loss_avg = print_loss_total / d
                    print('epoch %d %s (%d %.2f%%) %.4f' % (iter, self.timeSince(start, d / len(self.dataset.list_data)),
                                                 d, float(d) / len(self.dataset.list_data) * 100, print_loss_avg))

            plot_loss = plot_loss_total
            plot_losses.append(plot_loss)
            plot_loss_total = 0
            print('%s (%d %f%%) %.4f avg: %.4f' % (self.timeSince(start, iter / n_iters),
                                         iter, iter / float(n_iters) * 100, print_loss_total, print_loss_total/len(self.dataset.list_data)))
            torch.save(self.encoder.state_dict(), self.save_path +  str(iter) + str(print_loss_total) + "encoder")
            torch.save(self.decoder.state_dict(), self.save_path +  str(iter) + str(print_loss_total) + "decoder")

            if (self.best_lost > print_loss_total):
                self.best_lost = print_loss_total
                torch.save(self.encoder.state_dict(), self.save_path + "_best_" + "encoder")
                torch.save(self.decoder.state_dict(), self.save_path + "_best_" + "decoder")

            print_loss_total = 0

        self.showPlot(plot_losses)

    def showPlot(self, points):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)

    # def evaluate(self, encoder, decoder, sentence, max_length=MAX_LENGTH):
    #     input_variable = variableFromSentence(input_lang, sentence)
    #     input_length = input_variable.size()[0]
    #     encoder_hidden = encoder.initHidden()
    #
    #     encoder_outputs = (torch.zeros(max_length, encoder.hidden_size))
    #     encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    #
    #     for ei in range(input_lengtVariableh):
    #         encoder_output, encoder_hidden = encoder(input_variable[ei],
    #                                                  encoder_hidden)
    #         encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
    #
    #     decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    #     decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    #
    #     decoder_hidden = encoder_hidden
    #
    #     decoded_words = []
    #     decoder_attentions = torch.zeros(max_length, max_length)
    #
    #     for di in range(max_length):
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #             decoder_input, decoder_hidden, encoder_output, encoder_outputs)
    #         decoder_attentions[di] = decoder_attention.data
    #         topv, topi = decoder_output.data.topk(1)
    #         ni = topi[0][0]
    #         if ni == EOS_token:
    #             decoded_words.append('<EOS>')
    #             break
    #         else:
    #             decoded_words.append(output_lang.index2word[ni])
    #
    #         decoder_input = Variable(torch.LongTensor([[ni]]))
    #         decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    #
    #     return decoded_words, decoder_attentions[:di + 1]
    #

    ######################################################################
    # We can evaluate random sentences from the training set and print out the
    # input, target, and output to make some subjective quality judgements:
    #

    # def evaluateRandomly(self, encoder, decoder, n=10):
    #     for i in range(n):
    #         pair = random.choice(pairs)
    #         print('>', pair[0])
    #         print('=', pair[1])
    #         output_words, attentions = self.evaluate(encoder, decoder, pair[0])
    #         output_sentence = ' '.join(output_words)
    #         print('<', output_sentence)
    #         print('')

######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it's easier to run multiple experiments easier) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#



######################################################################
#
#
# evaluateRandomly(encoder1, attn_decoder1)


# output_words, attentions = evaluate(
#     encoder1, attn_decoder1, "je suis trop froid .")
# plt.matshow(attentions.numpy())


# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()


# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)
