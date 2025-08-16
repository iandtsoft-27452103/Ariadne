from calendar import c
import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import torch.optim.lr_scheduler
import numpy
import random
#import pickle
#import os
import re
import gru
import position
import board
import inout
import logging
#import result
import file
import rank
import bitop
import piece
import makemove
import feature
import sfen
import csa
from datetime import datetime as dt

class train_gru:
    def __init__(self):
        self.train_batchsize = 32
        self.test_batchsize = 32
        self.seq_length = 128
        self.epoch = 1
        self.iteration = 0
        self.log_file_name = 'train_gru.txt'
        self.log_file_name2 = 'train_gru_epoch.txt'
        self.file_number = 0
        self.model_file_name = 'model_gru.pth'
        self.optimizer_file_name = 'optimizer_gru.pth'
        self.scheduler_file_name = 'scheduler_gru.pth'
        self.is_load_model = 0
        self.is_load_optimizer = 0
        self.is_load_scheduler = 0
        self.is_log_init = 0
        self.lr = 0.00001# 0.01 => 
        #self.lr = 0.0001
        self.mt = 0.9#Adamでは使用しない
        self.wd = 0.0001
        self.shuffle_threshold = 12000
        self.is_use_sgd = 100
        self.lr_decay_threshold = 300#300
        self.console_out_threshold = 50
        self.record_start_number = 0
        self.record_end_number = 1000

    #棋譜からの学習
    def train_gru(self):

        #モデルとoptimizerを用意する
        device = torch.device("cuda:0")
        model = gru.gru()
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = self.lr, momentum = self.mt, weight_decay = self.wd, nesterov = True)
        #optimizer = torch.optim.RMSprop(model.parameters(), lr = self.lr, weight_decay=self.wd, momentum = self.mt)
        #optimizer = torch.optim.Adagrad(model.parameters(), weight_decay = self.wd)
        #optimizer = torch.optim.Adadelta(model.parameters())
        #optimizer = torch.optim.Adamax(model.parameters())
        #optimizer = torch.optim.Adam(model.parameters(), lr = self.lr, betas=[0.9, 0.999], weight_decay = self.wd, amsgrad=True)
        
        #各クラスのインスタンスを作成する
        bo = board.board()
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        ft = feature.feature(bo)
        pc = piece.piece()

        #モデルをロードする
        if self.is_load_model != 0:
            model.load_state_dict(torch.load(self.model_file_name))
        
        #optimizerをロードする
        if self.is_load_optimizer != 0:
            optimizer.load_state_dict(torch.load(self.optimizer_file_name))

        #継続して学習する場合、総iteration数, 総epoch数を読み込む
        #total_epoch_count = 0
        #if self.is_log_init == 0:
            #log_file = open(self.log_file_name, 'r', 1, 'UTF-8')
            #for line in log_file:
                #total_iteration_count = int(line)
            #log_file.close()
            #if self.file_number == 13 and self.record_end_number == 36727:
                #log_file = open(self.log_file_name2, 'r', 1, 'UTF-8')
                #for line in log_file:
                    #total_epoch_count = int(line)
                #log_file.close()
        #else:
            #total_iteration_count = 0
            #total_epoch_count = 0

        iteration = 0

        #epoch毎の学習のループ
        for epoch in range(self.epoch):

            #epoch開始時間を出力する
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
            tstr = "epoch start! " + tstr
            print(tstr)

            #for record_file_number in range(17):
            while True:

                #self.file_number = record_file_number
                record_file_number = self.file_number

                #棋譜を読み込む
                clsio = None
                clsio = inout.inout()
                file_name = 'records' + str(self.file_number) + '.txt'
                clsio.read_records(file_name)
                #clsio.read_records('test_records.txt')

                console_out_counter = 0

                for ith in range(self.record_start_number, self.record_end_number):
                #for ith in range(len(clsio.rec)):

                    bo.init_board(bo.board_default, bo.hand_default, bi, 0)
                    tb_list = []
                    tl_list = []
                    color = 0
                
                    for ply in range(clsio.rec[ith].ply):
                        move = clsio.rec[ith].moves[ply]

                        #現在の局面から特徴を取得する
                        #fe = ft.make_input_features4(bo, color)
                        #fe = numpy.array(fe)
                        #fe = fe.reshape(105*81)
                        #tb_list.append(fe)

                        #ラベルを保存する
                        __, direc = ft.make_output_labels(bo, move)
                        lbl = ((direc << 7) | move.ito)
                        
                        #from_sq = move.ifrom
                        #if move.ifrom == bo.square_nb:
                        #    from_sq = move.ifrom + move.piece_to_move - 1
                        #lbl = ft.label_list[move.flag_promo][from_sq][move.ito]

                        tb_list.append(lbl)
                        tl_list.append(lbl)

                        ma.makemove(bo, move, ply + 1, bi, pc, color)
                        color = color ^ 1

                    bs_list = []
                    cnt = clsio.rec[ith].ply // self.train_batchsize
                    remainder = clsio.rec[ith].ply - cnt * self.train_batchsize
                    for i in range(cnt):
                        bs_list.append(self.train_batchsize)
                    if remainder != 0:
                        bs_list.append(remainder)

                    for i in range(len(bs_list)):
                        model.batch_size = bs_list[i]
                        train_batch = numpy.zeros((bs_list[i], self.seq_length))
                        train_label = numpy.zeros((bs_list[i]))
                        ply = self.train_batchsize * i
                        for j in range(bs_list[i]):
                            train_label[j] = tl_list[ply]
                            if ply > self.seq_length - 1:
                                idx = j
                            else:
                                idx = self.seq_length - 1

                            l = self.seq_length - 1
                            for k in range(idx, -1, -1):
                                if ply <= k and ply < self.seq_length:
                                    train_batch[j][l] = 0#Null Move
                                else:
                                    train_batch[j][l] = tb_list[k]
                                l -= 1
                                if l < 0:
                                    break
                            ply += 1

                        #学習を開始する
                        model.train()

                        #バッチ毎GPUに転送する
                        train_batch = numpy.array(train_batch)
                        train_label = numpy.array(train_label)
                        x = torch.tensor(train_batch, dtype = torch.long)
                        x = x.to(device)
                        t = torch.tensor(train_label, dtype = torch.long)
                        t = t.to(device)

                        #順伝播を実行する
                        y = model.forward(x)

                        #勾配を初期化する
                        model.zero_grad()
                        optimizer.zero_grad()

                        #損失関数を計算する
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(y, t)

                        #逆伝播を実行する
                        loss.backward()

                        #パラメータを更新する
                        optimizer.step()
                        console_out_counter += 1
                        iteration += 1

                        l = loss.tolist() / bs_list[i]

                        if console_out_counter == self.console_out_threshold:
                            console_out_counter = 0
                            print('file_number = {} record_number = [{}/{}] train_loss = {a:.10f}'.format(record_file_number, ith + 1, self.record_end_number, a=l))
                break

            #epoch終了時間を出力する
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
            tstr = "epoch end! " + tstr
            print(tstr)

        #モデルを保存する
        torch.save(model.state_dict(), self.model_file_name)
        #optimizerを保存する
        torch.save(optimizer.state_dict(), self.optimizer_file_name)
        #ログファイルを保存する
        #log_file = open(self.log_file_name, 'w', 1, 'UTF-8')
        #total_iteration_count += iteration
        #log_file.write(str(total_iteration_count))
        #log_file.write('\n')
        #log_file.close()
        #if self.file_number == 13 and self.record_end_number == 36727:
            #log_file = open(self.log_file_name2, 'w', 1, 'UTF-8')
            #total_epoch_count += 1
            #log_file.write(str(total_epoch_count))
            #log_file.write('\n')
            #log_file.close()