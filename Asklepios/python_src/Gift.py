from os import error
import torch
import torch.nn as nn
import numpy
import bitop
import board
import color as C
import feature
import piece
import gencap
import gennocap
import gendrop
import genevasion
import sfen
#import policy
import gru
import value
import csa
#import test3
#import inout
#import analyze

def main():
    try:
        device = torch.device("cuda:0")
        model = gru.gru()
        model.load_state_dict(torch.load('model_gru.pth', weights_only=True))
        model = model.to(device)
        model_value = value.value()
        model_value.load_state_dict(torch.load('model_value.pth', weights_only=True))
        model_value = model_value.to(device)
        bo = board.board()
        ft = feature.feature(bo)
        bi = bitop.bitop(bo)
        pc = piece.piece()
        cls_sfen = sfen.sfen(bo, bi)
        cls_csa = csa.csa()
        at = ft.at
        c = C.color()
        cls_cap = gencap.gencap(bo, bi, pc, at, c)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, c)
        cm = ','
        seq_length = 128

        temp1 = torch.zeros((1,63,9,9), dtype=torch.float, device=device)
        temp2 = torch.zeros((1,56,9,9), dtype=torch.float, device=device)
        temp3 = torch.zeros((1,1,9,9), dtype=torch.float, device=device)
        temp4 = torch.zeros((1,seq_length), dtype=torch.long, device=device)
        model.eval()
        model_value.eval()

        #print('a')

        #bo = board.board()
        #clsio = inout.inout()
        #a = analyze.analyze()
        #clsio.read_records('20240804_nhk_hai.txt')
        #a.analyze_gru(clsio.rec[0], bo, 128)
        #return

        #1度実行しておくと初回の計算が速くなる。理由は不明。
        with torch.no_grad():
            model.batch_size = 1
            _ = model.forward(temp4)
            model_value.batch_size = 1
            _ = model_value.forward(temp1, temp2, temp3)

        print('Hello World.')

        cmd_line = ''
        #return
        #s0 = 'p'
        #s1 = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        #s2 = "l2g3nl/2skgr3/1p2psbpp/2pp2p2/pn3p1P1/2PPP1P2/PP1GSP2P/LK1B3R1/1NSG3NL b - 1";
        #s3 = "7nl/2k1gl1+R1/+LG2Pr2p/2ppp1p2/1s3p3/1pPP2P2/2G1B1+b1P/2G6/1K7 w 2SNL3Ps2n3p 1";
        #cmd_line = s0 + ',' + s1 + ',' + s2 + ',' + s3
        #cmd_line = 'v,1r5nl/3gk1g2/2ns1s1p1/l1pp1ppbp/4P4/p1PP2P1P/1P1S1S3/PBG6/LN1K1+r2L b Pn3p 1,lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1'
        #cmd_line = 'p,lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1,lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1'
        #cmd_line = 'p,ln1+P3Rl/9/pskp1p+B1p/1pps5/g8/6P2/P2PPPN1P/3+r1K3/L3+p1S1L b 2Pbs2n3p 1'
        #cmd_line = 'p,lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1:2726FU,lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 1:2726FU:8384FU'

        while True:
            cmd_line = input()
            cmd = cmd_line.split(',')
            if cmd[0] == 'quit':
                break
            
            str_mode = cmd[0]
            limit = len(cmd) - 1
            li1 = []
            li2 = []
            li3 = []
            li_moves = []
            li_csa_moves = []
            li_directions = []
            #is_mate = False
            str_ret = ''
            cnt = 0
            for i in range(limit):
                bo.clear_board()
                if str_mode == 'p':
                    cmd2 = cmd[i + 1].split(':')
                    cls_sfen.str_sfen = cmd2[0]
                    cls_sfen.parse_sfen()
                    limit2 = len(cmd2) - 1
                    limit3 = seq_length - limit2
                    li4 = []
                    moves = []
                    csa_moves = []
                    directions = []
                    if cls_sfen.board.color == 0:
                        if at.is_attacked(cls_sfen.board, cls_sfen.board.sq_king[0], cls_sfen.board.color ^ 1) == 0:
                            cls_cap.b_gen_captures(moves)
                            cls_nocap.b_gen_nocaptures(moves)
                            cls_drop.b_gen_drop(moves)
                        else:
                            cls_eva.b_gen_evasion(moves)
                    else:
                        if at.is_attacked(cls_sfen.board, cls_sfen.board.sq_king[1], cls_sfen.board.color ^ 1) == 0:
                            cls_cap.w_gen_captures(moves)
                            cls_nocap.w_gen_nocaptures(moves)
                            cls_drop.w_gen_drop(moves)
                        else:
                            cls_eva.w_gen_evasion(moves)
                    if len(moves) == 0:
                        li_empty = []
                        li_moves.append(li_empty)
                        li_csa_moves.append(li_empty)
                        li_directions.append(li_empty)
                    else:
                        for j in range(len(moves)):
                            _, direc = ft.make_output_labels(cls_sfen.board, moves[j])
                            directions.append(direc)
                            str_csa = cls_csa.board_to_csa(cls_sfen.board, pc, moves[j])
                            csa_moves.append(str_csa)
                        li_moves.append(moves)
                        li_csa_moves.append(csa_moves)
                        li_directions.append(directions)#csa_to_board(self, bo, pc, str_move):
                        for j in range(limit2):
                            move = cls_csa.csa_to_board(cls_sfen.board, pc, cmd2[j + 1])
                            _, direc = ft.make_output_labels(cls_sfen.board, move)
                            lbl = ((direc << 7) | move.ito)
                            li4.append(lbl)
                        for j in range(limit3):
                            li4.append(0)
                        #fe = ft.make_input_features1(cls_sfen.board)
                        #li1.append(fe)
                        #fe = ft.make_input_features2(cls_sfen.board)
                        #li2.append(fe)
                        #fe = ft.make_input_features3(cls_sfen.board, cls_sfen.board.color)
                        #li3.append(fe)
                        cnt += 1
                elif str_mode == 'v':
                    cls_sfen.str_sfen = cmd[i + 1]
                    cls_sfen.parse_sfen()
                    fe = ft.make_input_features1(cls_sfen.board)
                    li1.append(fe)
                    fe = ft.make_input_features2(cls_sfen.board)
                    li2.append(fe)
                    fe = ft.make_input_features3(cls_sfen.board, cls_sfen.board.color)
                    li3.append(fe)
                    cnt += 1
                if str_mode == 'p':
                    li1.append(li4)
            if str_mode == 'p':
                length = len(li1)
                li1 = numpy.array(li1)
                #li2 = numpy.array(li2)
                #li3 = numpy.array(li3)
                x1 = torch.tensor(li1, dtype = torch.long)
                x1 = x1.to(device)
                #x2 = torch.tensor(li2, dtype = torch.float)
                #x2 = x2.to(device)
                #x3 = torch.tensor(li3, dtype = torch.float)
                #x3 = x3.to(device)
                #yのoutput_shape = 5864
                model.eval()
                with torch.no_grad():
                    if length > 0:
                        model.batch_size = cnt
                        y = model.forward(x1)
                        #y = y.reshape(cnt, 256, 81)#アウトプット層は本来チャンネル数を32にすべきである。
                        y = y.reshape(cnt, 5864)
                        y = y.tolist()
                        #lbl = ((direc << 7) | correct_move.ito)
                    
                    k = 0
                    for i in range(len(li_moves)):
                        #print(i)
                        if len(li_moves[i]) == 0:
                            if i == 0 and len(li_moves) == 1:
                                str_ret += 'mate'
                            elif i == 0 and len(li_moves) > 1:
                                str_ret += 'mate:'
                            else:
                                if i == len(li_moves) - 1:
                                    str_ret += 'mate'
                                else:
                                    str_ret += 'mate:'
                            continue
                        #print('bbb')
                        #print(i)
                        l = []
                        for j in range(len(li_moves[i])):
                            lbl = ((li_directions[i][j] << 7) | li_moves[i][j].ito)
                            v = y[k][lbl]
                            l.append(v)
                        #print('ccc')
                        length = len(l)
                        if length > 7:
                            length = 8
                        l = numpy.array(l)
                        l = torch.tensor(l, dtype = torch.float)
                        l = torch.softmax(l, 0, dtype = torch.float)
                        l_copy = l
                        indexes = []
                        for _ in range(length):
                            idx = int(torch.argmax(l_copy))
                            indexes.append(idx)
                            l_copy = l_copy.tolist()
                            l_copy[idx] = 0
                            l_copy = torch.tensor(l_copy, dtype = torch.float)

                        l = l.tolist()
                        vl = []
                        for j in range(length):
                            vl.append(l[indexes[j]])
                        s = ''
                        for j in range(length):
                            if j != 0:
                                s += cm
                            s += li_csa_moves[i][indexes[j]]
                            s += " " + str(vl[j])
                            #if i != cnt - 1 and j == length - 1:
                            #s += ":"
                        str_ret += s
                        if i != len(li_moves) - 1:
                            str_ret += ":"
                        k += 1
            elif str_mode == 'v':
                li1 = numpy.array(li1)
                li2 = numpy.array(li2)
                li3 = numpy.array(li3)
                x1 = torch.tensor(li1, dtype = torch.float)
                x1 = x1.to(device)
                x2 = torch.tensor(li2, dtype = torch.float)
                x2 = x2.to(device)
                x3 = torch.tensor(li3, dtype = torch.float)
                x3 = x3.to(device)                
                model_value.eval()
                with torch.no_grad():
                    model_value.batch_size = cnt
                    y = model_value.forward(x1, x2, x3)
                    y = torch.sigmoid(y)
                    y = y.tolist()
                    for i in range(cnt):
                        if i != 0:
                            str_ret += cm
                        str_ret += str(y[i])
            if str_ret != "":
                print(str_ret)
                #break
    except Exception:
        file = open('error_log.txt', 'w', 1, 'UTF-8')
        file.write(cmd_line)
        file.close()

if __name__ == "__main__":
    main()
