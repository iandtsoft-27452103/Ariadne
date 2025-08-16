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
#import policy
import value
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

class train_value:
    def __init__(self):
        self.train_batchsize = 32
        self.test_batchsize = 32
        self.epoch = 1
        self.iteration = 0
        self.file_number = 0
        self.log_file_name = 'train_value.txt'
        self.model_file_name = 'model_value.pth'
        self.optimizer_file_name = 'optimizer_value.pth'
        self.scheduler_file_name = 'scheduler_value.pth'
        self.is_load_model = 0
        self.is_load_optimizer = 0
        self.is_load_scheduler = 0
        self.lr = 0.003#0.0001
        self.mt = 0.9#Adamでは使用しない
        self.wd = 0.0001
        self.shuffle_threshold = 3000
        self.is_use_sgd = 1
        self.lr_decay_threshold = 300
        self.console_out_threshold = 100
        self.record_start_number = 0
        self.record_end_number = 1000

    #Aperyで生成した局面からの学習
    def train_value(self):
        
        #グラフ描画用のList宣言
        train_loss_log = []
        train_acc_log = []
        test_loss_log = []
        test_acc_log = []

        #モデルとoptimizerを用意する
        device = torch.device("cuda:0")
        model = value.value()
        model = model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = self.lr, momentum = self.mt, weight_decay = self.wd, nesterov = True)
        #optimizer = torch.optim.RMSprop(model.parameters());
        #optimizer = torch.optim.Adagrad(model.parameters(), weight_decay = self.wd)
        #optimizer = torch.optim.Adadelta(model.parameters())
        #optimizer = torch.optim.Adamax(model.parameters())
        #optimizer = torch.optim.Adam(model.parameters(), weight_decay = self.wd)
        #optimizer = torch.optim.SparseAdam(model.parameters())
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.87, patience = 10)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 0.0005, max_lr = 0.001)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.935)

        #各クラスのインスタンスを作成する
        bo = board.board()
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        ft = feature.feature(bo)
        pc = piece.piece()
        #cls_sfen = sfen.sfen(bo, bi)
        #cls_csa = csa.csa()

        #モデルをロードする
        if self.is_load_model != 0:
            model.load_state_dict(torch.load(self.model_file_name))
        
        #optimizerをロードする
        if self.is_load_optimizer != 0:
            optimizer.load_state_dict(torch.load(self.optimizer_file_name))
        
        #schedulerをロードする
        #if self.is_load_scheduler != 0:
            #scheduler.load_state_dict(torch.load(self.scheduler_file_name))

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
        #iteration_max = int(iteration_max)

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

                #閾値に達したら学習局面をシャッフルする
                #if shuffle_counter == self.shuffle_threshold:
                    #shuffle_counter = 0
                    #msg = "data shuffle!"
                    #print(msg)
                    #random.shuffle(train_pos)

                #SGDの場合、閾値に達したら学習率を減衰する
                #if self.is_use_sgd == 1 and lr_decay_counter == self.lr_decay_threshold:
                    #lr_decay_counter = 0
                    #self.lr *= 0.9998
                    #optimizer = torch.optim.SGD(model.parameters(), lr = self.lr, momentum = self.mt, weight_decay = self.wd, nesterov = True)
                    #msg = "lr decay!"
                    #print(msg)

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
                    #current_move = clsio.rec[record_no].moves[current_ply]

                    #取得した手数まで棋譜を再生する
                    bo.init_board(bo.board_default, bo.hand_default, bi, 0)
                    color = 0
                    for ply in range(current_ply):
                        move = clsio.rec[record_no].moves[ply]
                        ma.makemove(bo, move, ply + 1, bi, pc, color)
                        color = color ^ 1
                
                    result = clsio.rec[record_no].result

                    #ラベルのリストに保存する
                    if color == 0:
                        if result == 'B':
                            lbl = 1
                        elif result == 'W':
                            lbl = 0
                        else:
                            lbl = 0.5
                    else:
                        if result == 'W':
                            lbl = 1
                        elif result == 'B':
                            lbl = 0
                        else:
                            lbl = 0.5

                    train_label.append(lbl)

                    #現在の局面から特徴を取得する
                    fe = ft.make_input_features1(bo)
                    train_batch1.append(fe)
                    fe = ft.make_input_features2(bo)
                    train_batch2.append(fe)
                    fe = ft.make_input_features3(bo, color)
                    train_batch3.append(fe)

                train_loss = 0
                #train_acc = 0
                #tcnt = 0

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
                t = torch.tensor(train_label, dtype = torch.float)
                t = t.to(device)

                #順伝播を実行する
                model.batch_size = self.train_batchsize
                y = model.forward(x1, x2, x3)

                #勾配を初期化する
                model.zero_grad()
                optimizer.zero_grad()

                #損失関数を計算する
                #criterion = nn.CrossEntropyLoss()
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(y, t)

                #ログ用のパラメータを更新する
                train_loss += loss.item()
                #pre = y.max(1)[1]
                #train_acc += (pre == t).sum().item()
                #tcnt += t.size(0)
                avg_train_loss = train_loss / self.train_batchsize
                #avg_train_acc = train_acc / tcnt

                #逆伝播を実行する
                loss.backward()

                #パラメータを更新する
                optimizer.step()

                #スケジューラを更新する
                #scheduler.step(loss)
                #scheduler.step()

                #コンソールに途中経過を出力する
                if console_out_counter == self.console_out_threshold:
                    console_out_counter = 0
                    print('epoch = [{}/{}], ith = [{}/{}], train_loss: {a:.10f}'.format(epoch + 1, self.epoch, iteration, iteration_max, a=avg_train_loss))

                #グラフ描画用のパラメータをListに保存する
                train_loss_log.append(avg_train_loss)
                #train_acc_log.append(avg_train_acc)
                #test_loss_log.append(avg_test_loss)
                #test_acc_log.append(avg_test_acc)

                #カウンタを更新する
                iteration += 1
                shuffle_counter += 1
                lr_decay_counter += 1
                console_out_counter += 1

            #epoch終了時間を出力する
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y-%m-%d %H:%M:%S')
            tstr = "epoch end! " + tstr
            print(tstr)

            #print('next lr = {a:.8f}'.format(a=self.lr))

            #if epoch >= 8 and avg_test_loss > test_loss_log[epoch - 1] and avg_test_loss > test_loss_log[epoch - 2] and avg_test_loss > test_loss_log[epoch - 3] and avg_test_loss > test_loss_log[epoch - 4] and avg_test_loss > test_loss_log[epoch - 5]:
                #break

        #モデルを保存する
        torch.save(model.state_dict(), self.model_file_name)
        #optimizerを保存する
        torch.save(optimizer.state_dict(), self.optimizer_file_name)
        #schedulerを保存する
        #torch.save(scheduler.state_dict(), self.scheduler_file_name)

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