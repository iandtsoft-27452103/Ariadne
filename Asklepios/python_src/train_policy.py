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
import policy
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
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime as dt

class train_policy:
    def __init__(self):
        self.train_batchsize = 32
        self.test_batchsize = 32
        self.epoch = 1
        self.iteration = 0
        self.file_number = 0
        self.log_file_name = 'train_policy.txt'
        self.model_file_name = 'model.pth'
        self.optimizer_file_name = 'optimizer.pth'
        self.scheduler_file_name = 'scheduler.pth'
        self.is_load_model = 0
        self.is_load_optimizer = 0
        self.is_load_scheduler = 0
        self.is_log_init = 0
        self.lr = 0.0001875#0.0001
        self.mt = 0.9#Adamでは使用しない
        self.wd = 0.0001
        self.betas0 = 0.0
        self.betas1 = 0.9
        self.shuffle_threshold = 3000
        self.is_use_sgd = 1
        self.lr_decay_threshold = 25000
        self.console_out_threshold = 100
        self.record_start_number = 0
        self.record_end_number = 1000

    #棋譜からの学習
    def train_policy(self):
        
        #グラフ描画用のList宣言
        train_loss_log = []
        train_acc_log = []
        test_loss_log = []
        test_acc_log = []

        #モデルとoptimizerを用意する
        device = torch.device("cuda:0")
        model = policy.policy()
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = self.lr, momentum = self.mt, weight_decay = self.wd, nesterov = True)
        #optimizer = torch.optim.RMSprop(model.parameters());
        #optimizer = torch.optim.Adagrad(model.parameters(), lr = self.lr, weight_decay = self.wd)
        #optimizer = torch.optim.Adadelta(model.parameters())
        #optimizer = torch.optim.Adamax(model.parameters())
        #optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd, amsgrad=True)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.98, patience = 45)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.0005, max_lr = 0.001)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = self.lr_decay_threshold, gamma = 0.99)

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
        
        #schedulerをロードする
        #if self.is_load_scheduler != 0:
            #scheduler.load_state_dict(torch.load(self.scheduler_file_name))

        #継続して学習する場合、総iteration数を読み込む
        #if self.is_log_init == 0:
            #log_file = open(self.log_file_name, 'r', 1, 'UTF-8')
            #for line in log_file:
                #total_iteration_count = int(line)
            #log_file.close()
        #else:
            #total_iteration_count = 0

        #棋譜を読み込む
        clsio = inout.inout()
        file_name = 'records' + str(self.file_number) + '.txt'
        clsio.read_records(file_name)
        #clsio.read_records('test_records.txt')

        #局面に番号を付ける
        pos = position.position()
        positions = pos.number_position(clsio.rec)

        #学習局面数とテスト局面数をセットする
        pos_count = len(positions)
        train_count = pos_count
        #train_count = int(pos_count * 0.9)
        #test_count = pos_count - train_count

        #学習局面とテスト局面を分割する
        train_pos = []
        for i in range(train_count):
            train_pos.append(positions[i])
        #test_pos = []
        #for i in range(train_count, pos_count):
            #test_pos.append(positions[i])

        iteration_max = train_count // self.train_batchsize
        #iteration_max = iteration_max // 4
        #iteration_max = self.iteration

        #epoch毎の学習のループ
        for epoch in range(self.epoch):

            #epoch開始時間を出力する
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
            tstr = "epoch start! " + tstr
            print(tstr)

            #学習局面をシャッフルする
            random.shuffle(train_pos)

            iteration = 0
            shuffle_counter = 0
            lr_decay_counter = 0
            console_out_counter = 0
            while iteration < iteration_max:

                loop_start = iteration * self.train_batchsize
                loop_end = iteration * self.train_batchsize + self.train_batchsize

                #学習用のミニバッチを生成する
                train_batch1 = []
                train_batch2 = []
                train_batch3 = []
                train_label = []

                for i in range(loop_start, loop_end):
                    #棋譜番号を取得する
                    record_no = train_pos[i].record_no

                    #手数を取得する
                    current_ply = train_pos[i].ply

                    #手を取得する
                    current_move = clsio.rec[record_no].moves[current_ply]

                    #取得した手数まで棋譜を再生する
                    bo.init_board(bo.board_default, bo.hand_default, bi, 0)
                    color = 0
                    for ply in range(current_ply):
                        move = clsio.rec[record_no].moves[ply]
                        ma.makemove(bo, move, ply + 1, bi, pc, color)
                        color = color ^ 1

                    #ラベルのリストに保存する
                    lbl, direc = ft.make_output_labels(bo, current_move)
                    train_label.append(lbl)

                    #現在の局面から特徴を取得する
                    fe = ft.make_input_features1(bo)
                    train_batch1.append(fe)
                    fe = ft.make_input_features2(bo)
                    train_batch2.append(fe)
                    fe = ft.make_input_features3(bo, color)
                    train_batch3.append(fe)

                train_loss = 0
                train_acc = 0
                tcnt = 0

                #学習を開始する
                model.train()

                #バッチ毎GPUに転送する
                train_batch1 = numpy.array(train_batch1)
                train_batch2 = numpy.array(train_batch2)
                train_batch3 = numpy.array(train_batch3)
                train_label = numpy.array(train_label)
                x1 = torch.tensor(train_batch1, dtype = torch.float)
                x1 = x1.to(device)
                x2 = torch.tensor(train_batch2, dtype = torch.float)
                x2 = x2.to(device)
                x3 = torch.tensor(train_batch3, dtype = torch.float)
                x3 = x3.to(device)
                t = torch.tensor(train_label, dtype = torch.long)
                t = t.to(device)

                #順伝播を実行する
                y = model.forward(x1, x2, x3)

                #勾配を初期化する
                model.zero_grad()
                optimizer.zero_grad()

                #損失関数を計算する
                criterion = nn.CrossEntropyLoss()
                loss = criterion(y, t)

                #ログ用のパラメータを更新する
                train_loss += loss.item()
                pre = y.max(1)[1]
                train_acc += (pre == t).sum().item()
                tcnt += t.size(0)
                avg_train_loss = train_loss / self.train_batchsize
                avg_train_acc = train_acc / tcnt

                #逆伝播を実行する
                loss.backward()

                #パラメータを更新する
                optimizer.step()

                #スケジューラを更新する
                #scheduler.step(loss)
                #scheduler.step()

                #グラフ描画用のパラメータをListに保存する
                train_loss_log.append(avg_train_loss)
                train_acc_log.append(avg_train_acc)
                #test_loss_log.append(avg_test_loss)
                #test_acc_log.append(avg_test_acc)

                #カウンタを更新する
                iteration += 1
                shuffle_counter += 1
                lr_decay_counter += 1
                console_out_counter += 1

                #コンソールに途中経過を出力する
                if console_out_counter == self.console_out_threshold:
                    console_out_counter = 0
                    print('epoch = [{}/{}], ith = [{}/{}], train_loss: {a:.10f}'.format(epoch + 1, self.epoch, iteration, iteration_max, a=avg_train_loss))
                
                #lr = scheduler.get_last_lr()
                #if lr[0] < 0.0008:
                    #print('lr is under 0.0008.')
                    #print('learning finished.')
                    #break

            #print('lr = ' + str(lr[0]))

            #epoch終了時間を出力する
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
            tstr = "epoch end! " + tstr
            print(tstr)

            #print('next lr = {a:.8f}'.format(a=self.lr))

        #モデルを保存する
        torch.save(model.state_dict(), self.model_file_name)
        #optimizerを保存する
        torch.save(optimizer.state_dict(), self.optimizer_file_name)
        #schedulerを保存する
        #torch.save(scheduler.state_dict(), self.scheduler_file_name)

        #ログファイルを保存する
        #log_file = open(self.log_file_name, 'w', 1, 'UTF-8')
        #total_iteration_count += iteration
        #log_file.write(str(total_iteration_count))
        #log_file.write('\n')
        #log_file.close()
        #グラフを描画する
        #if epoch > 1:
            #self.draw_graph(train_loss_log, train_acc_log, test_loss_log, test_acc_log)

    def draw_graph(self, train_loss_log, train_acc_log, test_loss_log, test_acc_log):
        plt.figure()
        plt.plot(range(self.epoch), train_loss_log, color="blue", linestyle="-", label="train_loss")
        plt.plot(range(self.epoch), test_loss_log, color="green", linestyle="--", label="test_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training and Testing loss")
        plt.grid()

        plt.figure()
        plt.plot(range(self.epoch), train_acc_log, color="blue", linestyle="-", label="train_accuracy")
        plt.plot(range(self.epoch), test_acc_log, color="green", linestyle="--", label="test_accuracy")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Training and Testing accuracy")
        plt.grid()

        plt.show()