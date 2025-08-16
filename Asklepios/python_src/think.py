import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import policy
import file
import rank
import makemove
import unmakemove
import gencap
import gennocap
import gendrop
import genevasion
import attack
import color
import usi
import numpy

class think:
    def __init__(self):
        self.model = policy.policy()
        self.model.load_state_dict(torch.load('model.pth'))

    def think(self, bo, bi, ft, c, pc):
        f = file.file()
        r = rank.rank()
        ma = makemove.makemove()
        um = unmakemove.unmakemove()
        #at = attack.attack()#ToDo:これは重いのでここで呼ばない方がよい
        at = ft.at
        cls_color = color.color()
        cls_usi = usi.usi()

        cls_cap = gencap.gencap(bo, bi, pc, at, cls_color)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, cls_color)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, cls_color)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, cls_color)

        device = torch.device("cuda:0")
        self.model = self.model.to(device)

        ply_max = 16

        pv = []
        str_pv = []

        co = c
        for ply in range(1, ply_max + 1):

            li = []
            fe = ft.make_input_features(bo, co)
            li.append(fe)

            li = numpy.array(li)

            self.model.eval()

            with torch.no_grad():
                x = torch.tensor(li, dtype = torch.float)
                x = x.to(device)
                y = self.model.forward(x)
                y = y.to("cpu")
                z = y.data[0].tolist()

            temp = []

            if at.is_attacked(bo, bo.sq_king[co], co ^ 1) == 0:
                if co == cls_color.black:
                    cls_cap.b_gen_captures(temp)
                    cls_nocap.b_gen_nocaptures(temp)
                    cls_drop.b_gen_drop(temp)
                else:
                    cls_cap.w_gen_captures(temp)
                    cls_nocap.w_gen_nocaptures(temp)
                    cls_drop.w_gen_drop(temp)
            else:
                if co == cls_color.black:
                    cls_eva.b_gen_evasion(temp)
                else:
                    cls_eva.w_gen_evasion(temp)

            moves = []
            for i in range(len(temp)):
                ma.makemove(bo, temp[i], ply, bi, pc, co)
                if at.is_attacked(bo, bo.sq_king[co], co ^ 1) == 0:
                    moves.append(temp[i])
                um.unmakemove(bo, temp[i], ply, bi, pc, co)

            if len(moves) == 0:
                break

            digits = []
            #best_digit = 0
            move_index = 0
            for i in range(len(moves)):
                a, b = ft.make_output_labels(bo, moves[i])
                d = z[b][bo.rank_table[moves[i].ito]][bo.file_table[moves[i].ito]]
                if i == 0:
                    best_digit = d
                digits.append(d)
                if d > best_digit:
                    best_digit = d
                    move_index = i

            pv.append(moves[move_index])
            str_usi = cls_usi.board_to_usi(bo, co, pc, moves[move_index])
            str_pv.append(str_usi)

            ma.makemove(bo, moves[move_index], ply, bi, pc, co)

            co ^= 1

        if len(pv) == 0:
            print('bestmove resign\n')
            return

        s = 'info depth ' + (str)(len(pv)) + ' pv'
        for i in range(len(pv)):
            s += ' ' + str_pv[i]

        s += '\n'
        print(s)
        s = 'bestmove ' + str_pv[0] + '\n'
        print(s)