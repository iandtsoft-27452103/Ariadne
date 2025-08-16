import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import policy
import value
import pv_nn
import file
import rank
import piece
import bitop
import makemove
import unmakemove
import feature
import gencap
import gennocap
import gendrop
import genevasion
import attack
import sfen
import csa
import move
import color as C
import numpy

class test3:
    def test(self, records, bo):
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        ft = feature.feature(bo)
        pc = piece.piece()
        #at = attack.attack()
        at = ft.at
        c = C.color()
        cls_cap = gencap.gencap(bo, bi, pc, at, c)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, c)

        device = torch.device("cuda:0")
        model = policy.policy()
        model.load_state_dict(torch.load('model.pth'))
        model = model.to(device)

        total_move_count = 0
        total_match_count = 0

        for cnt in range(len(records)):
            color = 0
            move_count = 0
            match_count = 0
            bo.init_board(bo.board_default, bo.hand_default, bi, 0)
            for ply in range(1, len(records[cnt].moves) + 1):
                current = records[cnt].moves[ply - 1]
                li = []
                fe = ft.make_input_features1(bo)
                li.append(fe)
                li2 = []
                fe = ft.make_input_features2(bo)
                li2.append(fe)
                li3 = []
                fe = ft.make_input_features3(bo, color)
                li3.append(fe)

                li = numpy.array(li)
                li2 = numpy.array(li2)
                li3 = numpy.array(li3)

                model.eval()

                with torch.no_grad():
                    x1 = torch.tensor(li, dtype = torch.float)
                    x1 = x1.to(device)
                    x2 = torch.tensor(li2, dtype = torch.float)
                    x2 = x2.to(device)
                    x3 = torch.tensor(li3, dtype = torch.float)
                    x3 = x3.to(device)
                    y = model.forward(x1, x2, x3)
                    y = y.to("cpu")

                    moves = []
                    if color == 0:
                        if at.is_attacked(bo, bo.sq_king[0], color ^ 1) == 0:
                            cls_cap.b_gen_captures(moves)
                            cls_nocap.b_gen_nocaptures(moves)
                            cls_drop.b_gen_drop(moves)
                        else:
                            cls_eva.b_gen_evasion(moves)
                    else:
                        if at.is_attacked(bo, bo.sq_king[1], color ^ 1) == 0:
                            cls_cap.w_gen_captures(moves)
                            cls_nocap.w_gen_nocaptures(moves)
                            cls_drop.w_gen_drop(moves)
                        else:
                            cls_eva.w_gen_evasion(moves)

                    correct_move = current
                    correct_label, direc = ft.make_output_labels(bo, correct_move)
                    z = y.data[0].tolist()

                    #correct_digit = z[direc][bo.file_table[correct_move.ito]][bo.rank_table[correct_move.ito]]
                    correct_digit = z[direc][bo.rank_table[correct_move.ito]][bo.file_table[correct_move.ito]]

                    digits = []
                    flag = False
                    for i in range(len(moves)):
                        a, b = ft.make_output_labels(bo, moves[i])
                        #c = z[b][bo.file_table[moves[i].ito]][bo.rank_table[moves[i].ito]]
                        c = z[b][bo.rank_table[moves[i].ito]][bo.file_table[moves[i].ito]]
                        digits.append(c)
                        if c > correct_digit:
                            flag = True

                    ma.makemove(bo, correct_move, ply, bi, pc, color)

                    if flag == False:
                        match_count += 1
                color = color ^ 1
                move_count += 1

            matching_rate = match_count / move_count
            print("[record ", cnt + 1, "]")
            print(match_count, "/", move_count, " ", matching_rate)
            print("")

            total_match_count += match_count
            total_move_count += move_count
        
        total_matching_rate = total_match_count / total_move_count
        print("[total]")
        print(total_match_count, "/", total_move_count, total_matching_rate)

    def test2(self, test_sfen, bo):
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ft = feature.feature(bo)
        pc = piece.piece()
        #at = attack.attack()
        at = ft.at
        c = C.color()
        cls_sfen = sfen.sfen(bo, bi)
        cls_csa = csa.csa()
        cls_cap = gencap.gencap(bo, bi, pc, at, c)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, c)

        device = torch.device("cuda:0")
        model = policy.policy()
        model.load_state_dict(torch.load('model.pth'))
        model = model.to(device)

        total_move_count = len(test_sfen)
        total_match_count = 0

        for cnt in range(len(test_sfen)):
            #bo.init_board(bo.board_default, bo.hand_default, bi, 0)
            bo.clear_board()
            cls_sfen.str_sfen = test_sfen[cnt].str_sfen
            cls_sfen.parse_fen()

            correct_move = move.move(bo, pc)
            correct_move = cls_csa.csa_to_board(bo, pc, test_sfen[cnt].bestmove)
            color = cls_sfen.board.color
            li = []
            fe = ft.make_input_features1(bo)
            li.append(fe)
            li2 = []
            fe = ft.make_input_features2(bo)
            li2.append(fe)
            li3 = []
            fe = ft.make_input_features3(bo, color)
            li3.append(fe)

            li = numpy.array(li)
            li2 = numpy.array(li2)
            li3 = numpy.array(li3)

            model.eval()

            with torch.no_grad():
                x1 = torch.tensor(li, dtype = torch.float)
                x1 = x1.to(device)
                x2 = torch.tensor(li2, dtype = torch.float)
                x2 = x2.to(device)
                x3 = torch.tensor(li3, dtype = torch.float)
                x3 = x3.to(device)
                y = model.forward(x1, x2, x3)
                y = y.to("cpu")

                moves = []
                if color == 0:
                    if at.is_attacked(bo, bo.sq_king[0], color ^ 1) == 0:
                        cls_cap.b_gen_captures(moves)
                        cls_nocap.b_gen_nocaptures(moves)
                        cls_drop.b_gen_drop(moves)
                    else:
                        cls_eva.b_gen_evasion(moves)
                else:
                    if at.is_attacked(bo, bo.sq_king[1], color ^ 1) == 0:
                        cls_cap.w_gen_captures(moves)
                        cls_nocap.w_gen_nocaptures(moves)
                        cls_drop.w_gen_drop(moves)
                    else:
                        cls_eva.w_gen_evasion(moves)

                correct_label, direc = ft.make_output_labels(bo, correct_move)

                z = y.data[0].tolist()

                correct_digit = z[direc][bo.file_table[correct_move.ito]][bo.rank_table[correct_move.ito]]

                digits = []
                flag = False
                for i in range(len(moves)):
                    a, b = ft.make_output_labels(bo, moves[i])
                    c = z[b][bo.file_table[moves[i].ito]][bo.rank_table[moves[i].ito]]
                    digits.append(c)
                    if c > correct_digit:
                        flag = True

                if flag == False:
                    total_match_count += 1
        
        total_matching_rate = total_match_count / total_move_count
        print("[total]")
        print(total_match_count, "/", total_move_count, total_matching_rate)

    #Value Network検証用
    def test3(self, record, bo):
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        uma = unmakemove.unmakemove()
        ft = feature.feature(bo)
        pc = piece.piece()
        #at = attack.attack()
        at = ft.at
        c = C.color()
        cls_cap = gencap.gencap(bo, bi, pc, at, c)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, c)

        device = torch.device("cuda:0")
        model = value.value()
        model.load_state_dict(torch.load('model_value.pth'))
        model = model.to(device)

        color = 0
        match_count = 0
        bo.init_board(bo.board_default, bo.hand_default, bi, 0)
        for ply in range(1, len(record.moves) + 1):
            current = record.moves[ply - 1]
            moves = []
            if color == 0:
                if at.is_attacked(bo, bo.sq_king[0], color ^ 1) == 0:
                    cls_cap.b_gen_captures(moves)
                    cls_nocap.b_gen_nocaptures(moves)
                    cls_drop.b_gen_drop(moves)
                else:
                    cls_eva.b_gen_evasion(moves)
            else:
                if at.is_attacked(bo, bo.sq_king[1], color ^ 1) == 0:
                    cls_cap.w_gen_captures(moves)
                    cls_nocap.w_gen_nocaptures(moves)
                    cls_drop.w_gen_drop(moves)
                else:
                    cls_eva.w_gen_evasion(moves)

            correct_move = current

            batch_size = 96
            counter = len(moves) // batch_size
            
            if counter == 0:
                counter = 1
            elif len(moves) > batch_size and len(moves) <= (batch_size * 2):
                counter = 2
            elif len(moves) > (batch_size * 2) and len(moves) <= (batch_size * 3):
                counter = 3
            elif len(moves) > (batch_size * 3) and len(moves) <= (batch_size * 4):
                counter = 4

            if len(moves) < batch_size:
                batch_size = len(moves)

            k = 0
            total = []
            flag = False
            for i in range(counter):
                li = []
                li2 = []
                li3 = []
                correct_index = 0
                j = 0
                while j < batch_size:
                    if k == len(moves):
                        break
                    ma.makemove(bo, moves[k], ply, bi, pc, color)
                    fe = ft.make_input_features1(bo)
                    li.append(fe)
                    fe = ft.make_input_features2(bo)
                    li2.append(fe)
                    fe = ft.make_input_features3(bo, color ^ 1)
                    li3.append(fe)
                    uma.unmakemove(bo, moves[k], ply, bi, pc, color)
                    ret = self.is_correct_move(correct_move, moves[k])
                    if ret == True:
                        correct_index = j + i * batch_size
                        #print("ply={a:.1f}".format(a=ply))
                    j += 1
                    k += 1
                li = numpy.array(li)
                li2 = numpy.array(li2)
                li3 = numpy.array(li3)

                model.eval()

                with torch.no_grad():
                    x1 = torch.tensor(li, dtype = torch.float)
                    x1 = x1.to(device)
                    x2 = torch.tensor(li2, dtype = torch.float)
                    x2 = x2.to(device)
                    x3 = torch.tensor(li3, dtype = torch.float)
                    x3 = x3.to(device)
                    model.batch_size = len(li)
                    y = model.forward(x1, x2, x3)
                    y = y.to("cpu")
                    y = y.sigmoid()
                    y = y.data.tolist()
                    for x in y:
                        total.append(x)
            for l in range(len(total)):
                if l == correct_index:
                    continue
                if total[correct_index] > total[l]:
                    flag = True
                    break
            if flag == False:
                match_count += 1
                print("match!")
            else:
                print("unmatch...")
            ma.makemove(bo, correct_move, ply, bi, pc, color)
            color ^= 1
        print(str(match_count) + " / " + str(len(record.moves)))
        s = match_count / len(record.moves)
        print(s)

    def test4(self, records, bo):
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        ft = feature.feature(bo)
        pc = piece.piece()
        #at = attack.attack()
        at = ft.at
        c = C.color()
        cls_cap = gencap.gencap(bo, bi, pc, at, c)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, c)

        device = torch.device("cuda:0")
        model = policy.policy()
        model.load_state_dict(torch.load('model.pth'))
        model = model.to(device)

        total_move_count = 0
        total_match_count = 0

        for cnt in range(len(records)):

            bs_list = []
            batch_size = 64
            batch_cnt = records[cnt].ply // batch_size
            remainder = records[cnt].ply - batch_cnt * batch_size
            for i in range(batch_cnt):
                bs_list.append(batch_size)
            if remainder != 0:
                bs_list.append(remainder)

            color = 0
            move_count = 0
            match_count = 0
            batch_number = 0
            i = 0
            fe_list1 = []
            fe_list2 = []
            fe_list3 = []
            result_list = []
            bo.init_board(bo.board_default, bo.hand_default, bi, 0)
            for ply in range(1, len(records[cnt].moves) + 1):
                current = records[cnt].moves[ply - 1]
                fe = ft.make_input_features1(bo)
                fe_list1.append(fe)
                fe = ft.make_input_features2(bo)
                fe_list2.append(fe)
                fe = ft.make_input_features3(bo, color)
                fe_list3.append(fe)

                ma.makemove(bo, current, ply, bi, pc, color)

                color = color ^ 1

                i += 1
                if i == bs_list[batch_number]:
                    x1 = numpy.array(fe_list1)
                    x2 = numpy.array(fe_list2)
                    x3 = numpy.array(fe_list3)

                    model.eval()
                    with torch.no_grad():
                        x1 = torch.tensor(x1, dtype = torch.float)
                        x1 = x1.to(device)
                        x2 = torch.tensor(x2, dtype = torch.float)
                        x2 = x2.to(device)
                        x3 = torch.tensor(x3, dtype = torch.float)
                        x3 = x3.to(device)
                        y = model.forward(x1, x2, x3)
                        y = y.to("cpu")
                        for j in range(bs_list[batch_number]):
                            z = y[j].tolist()
                            result_list.append(z)

                    batch_number += 1
                    i = 0
                    fe_list1 = []
                    fe_list2 = []
                    fe_list3 = []

            color = 0    
            bo.init_board(bo.board_default, bo.hand_default, bi, 0)        
            for ply in range(1, len(result_list) + 1):
                current = records[cnt].moves[ply - 1]

                moves = []
                if color == 0:
                    if at.is_attacked(bo, bo.sq_king[0], color ^ 1) == 0:
                        cls_cap.b_gen_captures(moves)
                        cls_nocap.b_gen_nocaptures(moves)
                        cls_drop.b_gen_drop(moves)
                    else:
                        cls_eva.b_gen_evasion(moves)
                else:
                    if at.is_attacked(bo, bo.sq_king[1], color ^ 1) == 0:
                        cls_cap.w_gen_captures(moves)
                        cls_nocap.w_gen_nocaptures(moves)
                        cls_drop.w_gen_drop(moves)
                    else:
                        cls_eva.w_gen_evasion(moves)

                correct_move = current
                correct_label, direc = ft.make_output_labels(bo, correct_move)
                z = result_list[ply - 1]

                correct_digit = z[direc][bo.rank_table[correct_move.ito]][bo.file_table[correct_move.ito]]

                digits = []
                flag = False
                for i in range(len(moves)):
                    a, b = ft.make_output_labels(bo, moves[i])
                    c = z[b][bo.rank_table[moves[i].ito]][bo.file_table[moves[i].ito]]
                    digits.append(c)
                    if c > correct_digit:
                        flag = True

                ma.makemove(bo, correct_move, ply, bi, pc, color)

                if flag == False:
                    match_count += 1

                color = color ^ 1
                move_count += 1

            matching_rate = match_count / move_count
            print("[record ", cnt + 1, "]")
            print(match_count, "/", move_count, " ", matching_rate)
            print("")

            total_match_count += match_count
            total_move_count += move_count

        total_matching_rate = total_match_count / total_move_count
        print("[total]")
        print(total_match_count, "/", total_move_count, total_matching_rate)

        #ログファイルを保存する（仮）
        log_file = open('matching_rate_log.txt', 'w', 1, 'UTF-8')
        s = str(total_match_count) + '/' + str(total_move_count)
        log_file.write(s)
        s = str(total_matching_rate)
        log_file.write('\n')
        log_file.write(s)
        log_file.write('\n')
        log_file.close()

    def test5(self, records, bo):
        f = file.file()
        r = rank.rank()
        bi = bitop.bitop(bo)
        ma = makemove.makemove()
        ft = feature.feature(bo)
        pc = piece.piece()
        #at = attack.attack()
        at = ft.at
        c = C.color()
        cls_cap = gencap.gencap(bo, bi, pc, at, c)
        cls_nocap = gennocap.gennocap(bo, bi, pc, at, c)
        cls_drop = gendrop.gendrop(bo, bi, pc, at, c)
        cls_eva = genevasion.genevasion(bo, bi, pc, at, c)

        device = torch.device("cuda:0")
        model = pv_nn.pv_nn()
        model.load_state_dict(torch.load('model_pv_nn.pth'))
        model = model.to(device)

        total_move_count = 0
        total_match_count = 0

        for cnt in range(len(records)):

            bs_list = []
            batch_size = 64
            batch_cnt = records[cnt].ply // batch_size
            remainder = records[cnt].ply - batch_cnt * batch_size
            for i in range(batch_cnt):
                bs_list.append(batch_size)
            if remainder != 0:
                bs_list.append(remainder)

            color = 0
            move_count = 0
            match_count = 0
            batch_number = 0
            i = 0
            fe_list1 = []
            fe_list2 = []
            fe_list3 = []
            result_list = []
            bo.init_board(bo.board_default, bo.hand_default, bi, 0)
            for ply in range(1, len(records[cnt].moves) + 1):
                current = records[cnt].moves[ply - 1]
                fe = ft.make_input_features1(bo)
                fe_list1.append(fe)
                fe = ft.make_input_features2(bo)
                fe_list2.append(fe)
                fe = ft.make_input_features3(bo, color)
                fe_list3.append(fe)

                ma.makemove(bo, current, ply, bi, pc, color)

                color = color ^ 1

                i += 1
                if i == bs_list[batch_number]:
                    x1 = numpy.array(fe_list1)
                    x2 = numpy.array(fe_list2)
                    x3 = numpy.array(fe_list3)

                    model.eval()
                    with torch.no_grad():
                        x1 = torch.tensor(x1, dtype = torch.float)
                        x1 = x1.to(device)
                        x2 = torch.tensor(x2, dtype = torch.float)
                        x2 = x2.to(device)
                        x3 = torch.tensor(x3, dtype = torch.float)
                        x3 = x3.to(device)
                        model.batch_size = i
                        y, _ = model.forward(x1, x2, x3)
                        y = y.to("cpu")
                        for j in range(bs_list[batch_number]):
                            z = y[j].tolist()
                            result_list.append(z)

                    batch_number += 1
                    i = 0
                    fe_list1 = []
                    fe_list2 = []
                    fe_list3 = []

            color = 0    
            bo.init_board(bo.board_default, bo.hand_default, bi, 0)        
            for ply in range(1, len(result_list) + 1):
                current = records[cnt].moves[ply - 1]

                moves = []
                if color == 0:
                    if at.is_attacked(bo, bo.sq_king[0], color ^ 1) == 0:
                        cls_cap.b_gen_captures(moves)
                        cls_nocap.b_gen_nocaptures(moves)
                        cls_drop.b_gen_drop(moves)
                    else:
                        cls_eva.b_gen_evasion(moves)
                else:
                    if at.is_attacked(bo, bo.sq_king[1], color ^ 1) == 0:
                        cls_cap.w_gen_captures(moves)
                        cls_nocap.w_gen_nocaptures(moves)
                        cls_drop.w_gen_drop(moves)
                    else:
                        cls_eva.w_gen_evasion(moves)

                correct_move = current
                correct_label, direc = ft.make_output_labels(bo, correct_move)
                z = result_list[ply - 1]

                correct_digit = z[direc][bo.rank_table[correct_move.ito]][bo.file_table[correct_move.ito]]

                digits = []
                flag = False
                for i in range(len(moves)):
                    a, b = ft.make_output_labels(bo, moves[i])
                    c = z[b][bo.rank_table[moves[i].ito]][bo.file_table[moves[i].ito]]
                    digits.append(c)
                    if c > correct_digit:
                        flag = True

                ma.makemove(bo, correct_move, ply, bi, pc, color)

                if flag == False:
                    match_count += 1

                color = color ^ 1
                move_count += 1

            matching_rate = match_count / move_count
            print("[record ", cnt + 1, "]")
            print(match_count, "/", move_count, " ", matching_rate)
            print("")

            total_match_count += match_count
            total_move_count += move_count

        total_matching_rate = total_match_count / total_move_count
        print("[total]")
        print(total_match_count, "/", total_move_count, total_matching_rate)

        #ログファイルを保存する（仮）
        #log_file = open('matching_rate_log.txt', 'w', 1, 'UTF-8')
        #s = str(total_match_count) + '/' + str(total_move_count)
        #log_file.write(s)
        #s = str(total_matching_rate)
        #log_file.write('\n')
        #log_file.write(s)
        #log_file.write('\n')
        #log_file.close()

    def is_correct_move(self, cm, m):
        if cm.ito != m.ito:
            return False
        if cm.ifrom != m.ifrom:
            return False
        if cm.flag_promo != m.flag_promo:
            return False
        if cm.piece_to_move != m.piece_to_move:
            return False
        if cm.cap_to_move != m.cap_to_move:
            return False
        return True