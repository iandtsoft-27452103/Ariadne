from string import whitespace
import numpy as np
#import inout
#import result
import bitop
import label
import direction
import piece
import attack

class feature:
    def __init__(self, bo):
        self.bi = bitop.bitop(bo)
        self.bb_full = self.bi.bb_ini()
        bb = self.bi.bb_ini()
        for i in range(len(bo.board)):
            bb = self.bi.xor(i, bb)
        self.bb_full = bb
        self.l = label.label()
        self.pc = piece.piece()
        self.di = direction.direction()
        self.at = attack.attack()
        # from     : 1010111
        # to       : 1010001 << 7
        # is_promo : 1 << 14
        num = ((1 << 14) + (80 << 7) + 87) + 1
        self.label_table = [self.l.EMPTY] * num
        self.set_label_table(bo)

    def make_input_features1(self, bo):
        features = []
        attacks_count = [[0 for j in range(bo.square_nb)] for i in range(2)]

        #総計 56 + 7 = 63個

        #先後の駒と利き 28 * 2 = 56個
        for i in range(2):
            #盤上の駒 14個
            bb = bo.bb_pawn[i]
            self.loop(bo, features, bb)
            bb = bo.bb_lance[i]
            self.loop(bo, features, bb)
            bb = bo.bb_knight[i]
            self.loop(bo, features, bb)
            bb = bo.bb_silver[i]
            self.loop(bo, features, bb)
            bb = bo.bb_gold[i]
            self.loop(bo, features, bb)
            bb = bo.bb_bishop[i]
            self.loop(bo, features, bb)
            bb = bo.bb_rook[i]
            self.loop(bo, features, bb)
            ft = np.zeros(bo.nfile * bo.nrank)
            ft[bo.sq_king[i]] = 1
            features.append(ft.reshape((bo.nfile, bo.nrank)))
            bb = bo.bb_pro_pawn[i]
            self.loop(bo, features, bb)
            bb = bo.bb_pro_lance[i]
            self.loop(bo, features, bb)
            bb = bo.bb_pro_knight[i]
            self.loop(bo, features, bb)
            bb = bo.bb_pro_silver[i]
            self.loop(bo, features, bb)
            bb = bo.bb_horse[i]
            self.loop(bo, features, bb)
            bb = bo.bb_dragon[i]
            self.loop(bo, features, bb)

            #駒の利き 14個

            #歩の利き
            bb_attacks = bo.bb_pawn_attacks[i]
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #香の利き
            bb = bo.bb_lance[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.get_file_attacks(bo.bb_occupied, sq)
                if i == 0:
                    bb2 = self.bi.bb_and(bb2, self.at.abb_minus_rays[sq])
                else:
                    bb2 = self.bi.bb_and(bb2, self.at.abb_plus_rays[sq])
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #桂の利き
            bb = bo.bb_knight[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.abb_knight_attacks[i][sq]
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #銀の利き
            bb = bo.bb_silver[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.abb_silver_attacks[i][sq]
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #金の利き
            bb = bo.bb_gold[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.abb_gold_attacks[i][sq]
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #角の利き
            bb = bo.bb_bishop[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.get_bishop_attacks(bo.bb_occupied, sq)
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #飛の利き
            bb = bo.bb_rook[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.get_rook_attacks(bo.bb_occupied, sq)
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop(bo, features, bb_attacks)

            #玉の利き
            bb_attacks = self.at.abb_king_attacks[bo.sq_king[i]]
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #と金の利き
            bb = bo.bb_pro_pawn[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.abb_gold_attacks[i][sq]
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #成香の利き
            bb = bo.bb_pro_lance[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.abb_gold_attacks[i][sq]
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #成桂の利き
            bb = bo.bb_pro_knight[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.abb_gold_attacks[i][sq]
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #成銀の利き
            bb = bo.bb_pro_lance[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.abb_gold_attacks[i][sq]
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #馬の利き
            bb = bo.bb_horse[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.get_bishop_attacks(bo.bb_occupied, sq)
                bb2 = self.bi.bb_or(bb2, self.at.abb_king_attacks[sq])
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

            #龍の利き
            bb = bo.bb_dragon[i]
            bb_attacks = self.bi.bb_ini()
            while bb != 0:
                sq = self.bi.first_one(bb)
                bb = self.bi.xor(sq, bb)
                bb2 = self.at.get_rook_attacks(bo.bb_occupied, sq)
                bb2 = self.bi.bb_or(bb2, self.at.abb_king_attacks[sq])
                bb_attacks = self.bi.bb_or(bb_attacks, bb2)
            self.loop3(bo, features, bb_attacks, attacks_count[i])

        #Square毎の利きの勝ち負け 7個
        sq_list = []
        for i in range(7):
            li = np.zeros(bo.square_nb)
            sq_list.append(li)

        for i in range(bo.square_nb):
            black_attacks_count = attacks_count[0][i]
            white_attacks_count = attacks_count[1][i]
            x = black_attacks_count - white_attacks_count
            if x >= 3:
                sq_list[0][i] = 1
            elif x == 2:
                sq_list[1][i] = 1
            elif x == 1:
                sq_list[2][i] = 1
            elif x == 0:
                sq_list[3][i] = 1
            elif x == -1:
                sq_list[4][i] = 1
            elif x == -2:
                sq_list[5][i] = 1
            elif x <= -3:
                sq_list[6][i] = 1

        for i in range(len(sq_list)):
            features.append(sq_list[i].reshape(bo.nfile, bo.nrank))

        return features

    def make_input_features2(self, bo):
        features = []
        for i in range(2):
            #持ち駒 28 * 2 = 56個
            self.loop2(bo, features, bo.hand_pawn[i], 8)#8枚のみ評価
            self.loop2(bo, features, bo.hand_lance[i], 4)
            self.loop2(bo, features, bo.hand_knight[i], 4)
            self.loop2(bo, features, bo.hand_silver[i], 4)
            self.loop2(bo, features, bo.hand_gold[i], 4)
            self.loop2(bo, features, bo.hand_bishop[i], 2)
            self.loop2(bo, features, bo.hand_rook[i], 2)

        return features

    def make_input_features3(self, bo, color):
        features = []

        #手番 1個
        if color == 0:
            ft = np.ones(bo.nfile * bo.nrank)
        else:
            ft = np.zeros(bo.nfile * bo.nrank)
        features.append(ft.reshape((bo.nfile, bo.nrank)))

        return features

    def loop(self, bo, features, bb):
        ft = np.zeros(bo.nfile * bo.nrank)
        while bb != 0:
            sq = self.bi.first_one(bb)
            bb = self.bi.xor(sq, bb)
            ft[sq] = 1
        features.append(ft.reshape((bo.nfile, bo.nrank)))

    def loop2(self, bo, features, hand, count):
        h = hand
        for i in range(count):
            if h > 0:
                #持ち駒があったら1埋め
                ft = np.ones(bo.nfile * bo.nrank)
                h -= 1
            else:
                #持ち駒がなかったら0埋め
                ft = np.zeros(bo.nfile * bo.nrank)
            features.append(ft.reshape((bo.nfile, bo.nrank)))

    def loop3(self, bo, features, bb, attacks_count):
        ft = np.zeros(bo.nfile * bo.nrank)
        while bb != 0:
            sq = self.bi.first_one(bb)
            bb = self.bi.xor(sq, bb)
            ft[sq] = 1
            attacks_count[sq] += 1
        features.append(ft.reshape((bo.nfile, bo.nrank)))

    def make_output_labels(self, bo, m):
        if m.ifrom < bo.square_nb:
            h = ((m.flag_promo << 14) + (m.ito << 7) + m.ifrom)
        else:
            h = ((m.flag_promo << 14) + (m.ito << 7) + (m.ifrom + m.piece_to_move - 1))
        move_direction = self.label_table[h]
        a = np.zeros(bo.nfile * bo.nrank)
        a[m.ito] = move_direction
        move_label = a.reshape((bo.nfile, bo.nrank))
        return move_label, move_direction

    def set_label_table(self, bo):
        for flag_promo in range(2):
            for ifrom in range(bo.square_nb + 7):
                for ito in range(bo.square_nb):
                    if ifrom == ito:
                        continue
                    if ifrom < bo.square_nb:
                        d = ito - ifrom
                        if flag_promo == 0:
                            if d == self.di.north_west or d == (self.di.north_west * 2) or d == (self.di.north_west * 3) or d == (self.di.north_west * 4) or d == (self.di.north_west * 5) or d == (self.di.north_west * 6) or d == (self.di.north_west * 7) or d == (self.di.north_west * 8):
                                move_direction = self.l.UP_LEFT
                            elif d == self.di.north or d == (self.di.north * 2) or d == (self.di.north * 3) or d == (self.di.north * 4) or d == (self.di.north * 5) or d == (self.di.north * 6) or d == (self.di.north * 7) or d == (self.di.north * 8):
                                move_direction = self.l.UP
                            elif d == self.di.north_east or d == (self.di.north_east * 2) or d == (self.di.north_east * 3) or d == (self.di.north_east * 4) or d == (self.di.north_east * 5) or d == (self.di.north_east * 6) or d == (self.di.north_east * 7) or d == (self.di.north_east * 8):
                                move_direction = self.l.UP_RIGHT
                            elif d == self.di.east or d == (self.di.east * 2) or d == (self.di.east * 3) or d == (self.di.east * 4) or d == (self.di.east * 5) or d == (self.di.east * 6) or d == (self.di.east * 7) or d == (self.di.east * 8):
                                move_direction = self.l.RIGHT
                            elif d == self.di.south_east or d == (self.di.south_east * 2) or d == (self.di.south_east * 3) or d == (self.di.south_east * 4) or d == (self.di.south_east * 5) or d == (self.di.south_east * 6) or d == (self.di.south_east * 7) or d == (self.di.south_east * 8):
                                move_direction = self.l.DOWN_RIGHT
                            elif d == self.di.south or d == (self.di.south * 2) or d == (self.di.south * 3) or d == (self.di.south * 4) or d == (self.di.south * 5) or d == (self.di.south * 6) or d == (self.di.south * 7) or d == (self.di.south * 8):
                                move_direction = self.l.DOWN
                            elif d == self.di.south_west or d == (self.di.south_west * 2) or d == (self.di.south_west * 3) or d == (self.di.south_west * 4) or d == (self.di.south_west * 5) or d == (self.di.south_west * 6) or d == (self.di.south_west * 7) or d == (self.di.south_west * 8):
                                move_direction = self.l.DOWN_LEFT
                            elif d == self.di.west or d == (self.di.west * 2) or d == (self.di.west * 3) or d == (self.di.west * 4) or d == (self.di.west * 5) or d == (self.di.west * 6) or d == (self.di.west * 7) or d == (self.di.west * 8):
                                move_direction = self.l.LEFT
                            elif d == self.di.knight_north_west:
                                move_direction = self.l.UP_LEFT_KNIGHT
                            elif d == self.di.knight_north_east:
                                move_direction = self.l.UP_RIGHT_KNIGHT
                            elif d == self.di.knight_south_east:
                                move_direction = self.l.DOWN_RIGHT_KNIGHT
                            elif d == self.di.knight_south_west:
                                move_direction = self.l.DOWN_LEFT_KNIGHT
                        else:
                            if d == self.di.north_west or d == (self.di.north_west * 2) or d == (self.di.north_west * 3) or d == (self.di.north_west * 4) or d == (self.di.north_west * 5) or d == (self.di.north_west * 6) or d == (self.di.north_west * 7) or d == (self.di.north_west * 8):
                                move_direction = self.l.UP_LEFT_PRO
                            elif d == self.di.north or d == (self.di.north * 2) or d == (self.di.north * 3) or d == (self.di.north * 4) or d == (self.di.north * 5) or d == (self.di.north * 6) or d == (self.di.north * 7) or d == (self.di.north * 8):
                                move_direction = self.l.UP_PRO
                            elif d == self.di.north_east or d == (self.di.north_east * 2) or d == (self.di.north_east * 3) or d == (self.di.north_east * 4) or d == (self.di.north_east * 5) or d == (self.di.north_east * 6) or d == (self.di.north_east * 7) or d == (self.di.north_east * 8):
                                move_direction = self.l.UP_RIGHT_PRO
                            elif d == self.di.east or d == (self.di.east * 2) or d == (self.di.east * 3) or d == (self.di.east * 4) or d == (self.di.east * 5) or d == (self.di.east * 6) or d == (self.di.east * 7) or d == (self.di.east * 8):
                                move_direction = self.l.RIGHT_PRO
                            elif d == self.di.south_east or d == (self.di.south_east * 2) or d == (self.di.south_east * 3) or d == (self.di.south_east * 4) or d == (self.di.south_east * 5) or d == (self.di.south_east * 6) or d == (self.di.south_east * 7) or d == (self.di.south_east * 8):
                                move_direction = self.l.DOWN_RIGHT_PRO
                            elif d == self.di.south or d == (self.di.south * 2) or d == (self.di.south * 3) or d == (self.di.south * 4) or d == (self.di.south * 5) or d == (self.di.south * 6) or d == (self.di.south * 7) or d == (self.di.south * 8):
                                move_direction = self.l.DOWN_PRO
                            elif d == self.di.south_west or d == (self.di.south_west * 2) or d == (self.di.south_west * 3) or d == (self.di.south_west * 4) or d == (self.di.south_west * 5) or d == (self.di.south_west * 6) or d == (self.di.south_west * 7) or d == (self.di.south_west * 8):
                                move_direction = self.l.DOWN_LEFT_PRO
                            elif d == self.di.west or d == (self.di.west * 2) or d == (self.di.west * 3) or d == (self.di.west * 4) or d == (self.di.west * 5) or d == (self.di.west * 6) or d == (self.di.west * 7) or d == (self.di.west * 8):
                                move_direction = self.l.LEFT_PRO
                            elif d == self.di.knight_north_west:
                                move_direction = self.l.UP_LEFT_KNIGHT_PRO
                            elif d == self.di.knight_north_east:
                                move_direction = self.l.UP_RIGHT_KNIGHT_PRO
                            elif d == self.di.knight_south_east:
                                move_direction = self.l.DOWN_RIGHT_KNIGHT_PRO
                            elif d == self.di.knight_south_west:
                                move_direction = self.l.DOWN_LEFT_KNIGHT_PRO
                    else:
                        if ifrom == bo.square_nb:
                            move_direction = self.l.DROP_PAWN
                        elif ifrom == bo.square_nb + 1:
                            move_direction = self.l.DROP_LANCE
                        elif ifrom == bo.square_nb + 2:
                            move_direction = self.l.DROP_KNIGHT
                        elif ifrom == bo.square_nb + 3:
                            move_direction = self.l.DROP_SILVER
                        elif ifrom == bo.square_nb + 4:
                            move_direction = self.l.DROP_GOLD
                        elif ifrom == bo.square_nb + 5:
                            move_direction = self.l.DROP_BISHOP
                        elif ifrom == bo.square_nb + 6:
                            move_direction = self.l.DROP_ROOK
                    h = ((flag_promo << 14) + (ito << 7) + ifrom)
                    self.label_table[h] = move_direction
