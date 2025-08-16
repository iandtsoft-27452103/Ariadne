import piece
import color
import move
import makemove
import usi

class sfen:
    def __init__(self, bo, bi):
        self.board = bo
        self.bitop = bi
        self.piece = piece.piece()
        self.color = color.color()
        self.str_sfen = 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1'

    #関数名がsfenでなく、fenになっているのに気付いたが…
    def parse_sfen(self):
        str_temp = self.str_sfen.split(' ')

        #盤面のparse
        str_pos = str_temp[0]
        length = len(str_pos)
        i = 0
        pos = 0
        flag_promo = 0
        while pos < length:
            s = str_pos[pos]
            if s == 'P' and flag_promo == 0:
                self.board.board[i] = self.piece.pawn
                self.board.bb_pawn[0] = self.bitop.xor(i, self.board.bb_pawn[0])
                self.board.bb_pawn_attacks[0] = self.bitop.xor(i - 9, self.board.bb_pawn_attacks[0])
                self.set_occupied(0, i)
                pos += 1
            elif s == 'P' and flag_promo == 1:
                self.board.board[i] = self.piece.pro_pawn
                self.board.bb_pro_pawn[0] = self.bitop.xor(i, self.board.bb_pro_pawn[0])
                self.board.bb_total_gold[0] = self.bitop.xor(i, self.board.bb_total_gold[0])
                self.set_occupied(0, i)
                flag_promo = 0
                pos += 1
            elif s == 'L' and flag_promo == 0:
                self.board.board[i] = self.piece.lance
                self.board.bb_lance[0] = self.bitop.xor(i, self.board.bb_lance[0])
                self.set_occupied(0, i)
                pos += 1
            elif s == 'L' and flag_promo == 1:
                self.board.board[i] = self.piece.pro_lance
                self.board.bb_pro_lance[0] = self.bitop.xor(i, self.board.bb_pro_lance[0])
                self.board.bb_total_gold[0] = self.bitop.xor(i, self.board.bb_total_gold[0])
                self.set_occupied(0, i)
                flag_promo = 0
                pos += 1
            elif s == 'N' and flag_promo == 0:
                self.board.board[i] = self.piece.knight
                self.board.bb_knight[0] = self.bitop.xor(i, self.board.bb_knight[0])
                self.set_occupied(0, i)
                pos += 1
            elif s == 'N' and flag_promo == 1:
                self.board.board[i] = self.piece.pro_knight
                self.board.bb_pro_knight[0] = self.bitop.xor(i, self.board.bb_pro_knight[0])
                self.board.bb_total_gold[0] = self.bitop.xor(i, self.board.bb_total_gold[0])
                self.set_occupied(0, i)
                flag_promo = 0
                pos += 1
            elif s == 'S' and flag_promo == 0:
                self.board.board[i] = self.piece.silver
                self.board.bb_silver[0] = self.bitop.xor(i, self.board.bb_silver[0])
                self.set_occupied(0, i)
                pos += 1
            elif s == 'S' and flag_promo == 1:
                self.board.board[i] = self.piece.pro_silver
                self.board.bb_pro_silver[0] = self.bitop.xor(i, self.board.bb_pro_silver[0])
                self.board.bb_total_gold[0] = self.bitop.xor(i, self.board.bb_total_gold[0])
                self.set_occupied(0, i)
                flag_promo = 0
                pos += 1
            elif s == 'G':
                self.board.board[i] = self.piece.gold
                self.board.bb_gold[0] = self.bitop.xor(i, self.board.bb_gold[0])
                self.board.bb_total_gold[0] = self.bitop.xor(i, self.board.bb_total_gold[0])
                self.set_occupied(0, i)
                pos += 1
            elif s == 'B' and flag_promo == 0:
                self.board.board[i] = self.piece.bishop
                self.board.bb_bishop[0] = self.bitop.xor(i, self.board.bb_bishop[0])
                self.board.bb_bh[0] = self.bitop.xor(i, self.board.bb_bh[0])
                self.set_occupied(0, i)
                pos += 1
            elif s == 'B' and flag_promo == 1:
                self.board.board[i] = self.piece.horse
                self.board.bb_horse[0] = self.bitop.xor(i, self.board.bb_horse[0])
                self.board.bb_bh[0] = self.bitop.xor(i, self.board.bb_bh[0])
                self.board.bb_hdk[0] = self.bitop.xor(i, self.board.bb_hdk[0])
                self.set_occupied(0, i)
                flag_promo = 0
                pos += 1
            elif s == 'R' and flag_promo == 0:
                self.board.board[i] = self.piece.rook
                self.board.bb_rook[0] = self.bitop.xor(i, self.board.bb_rook[0])
                self.board.bb_rd[0] = self.bitop.xor(i, self.board.bb_rd[0])
                self.set_occupied(0, i)
                pos += 1
            elif s == 'R' and flag_promo == 1:
                self.board.board[i] = self.piece.dragon
                self.board.bb_dragon[0] = self.bitop.xor(i, self.board.bb_dragon[0])
                self.board.bb_rd[0] = self.bitop.xor(i, self.board.bb_rd[0])
                self.board.bb_hdk[0] = self.bitop.xor(i, self.board.bb_hdk[0])
                self.set_occupied(0, i)
                flag_promo = 0
                pos += 1
            elif s == 'K':
                self.board.board[i] = self.piece.king
                self.board.sq_king[0] = i
                self.board.bb_hdk[0] = self.bitop.xor(i, self.board.bb_hdk[0])
                self.set_occupied(0, i)
                pos += 1
            elif s == 'p' and flag_promo == 0:
                self.board.board[i] = -self.piece.pawn
                self.board.bb_pawn[1] = self.bitop.xor(i, self.board.bb_pawn[1])
                self.board.bb_pawn_attacks[1] = self.bitop.xor(i + 9, self.board.bb_pawn_attacks[1])
                self.set_occupied(1, i)
                pos += 1
            elif s == 'p' and flag_promo == 1:
                self.board.board[i] = -self.piece.pro_pawn
                self.board.bb_pro_pawn[1] = self.bitop.xor(i, self.board.bb_pro_pawn[1])
                self.board.bb_total_gold[1] = self.bitop.xor(i, self.board.bb_total_gold[1])
                self.set_occupied(1, i)
                flag_promo = 0
                pos += 1
            elif s == 'l' and flag_promo == 0:
                self.board.board[i] = -self.piece.lance
                self.board.bb_lance[1] = self.bitop.xor(i, self.board.bb_lance[1])
                self.set_occupied(1, i)
                pos += 1
            elif s == 'l' and flag_promo == 1:
                self.board.board[i] = -self.piece.pro_lance
                self.board.bb_pro_lance[1] = self.bitop.xor(i, self.board.bb_pro_lance[1])
                self.board.bb_total_gold[1] = self.bitop.xor(i, self.board.bb_total_gold[1])
                self.set_occupied(1, i)
                flag_promo = 0
                pos += 1
            elif s == 'n' and flag_promo == 0:
                self.board.board[i] = -self.piece.knight
                self.board.bb_knight[1] = self.bitop.xor(i, self.board.bb_knight[1])
                self.set_occupied(1, i)
                pos += 1
            elif s == 'n' and flag_promo == 1:
                self.board.board[i] = -self.piece.pro_knight
                self.board.bb_pro_knight[1] = self.bitop.xor(i, self.board.bb_pro_knight[1])
                self.board.bb_total_gold[1] = self.bitop.xor(i, self.board.bb_total_gold[1])
                self.set_occupied(1, i)
                flag_promo = 0
                pos += 1
            elif s == 's' and flag_promo == 0:
                self.board.board[i] = -self.piece.silver
                self.board.bb_silver[1] = self.bitop.xor(i, self.board.bb_silver[1])
                self.set_occupied(1, i)
                pos += 1
            elif s == 's' and flag_promo == 1:
                self.board.board[i] = -self.piece.pro_silver
                self.board.bb_pro_silver[1] = self.bitop.xor(i, self.board.bb_pro_silver[1])
                self.board.bb_total_gold[1] = self.bitop.xor(i, self.board.bb_total_gold[1])
                self.set_occupied(1, i)
                flag_promo = 0
                pos += 1
            elif s == 'g':
                self.board.board[i] = -self.piece.gold
                self.board.bb_gold[1] = self.bitop.xor(i, self.board.bb_gold[1])
                self.board.bb_total_gold[1] = self.bitop.xor(i, self.board.bb_total_gold[1])
                self.set_occupied(1, i)
                pos += 1
            elif s == 'b' and flag_promo == 0:
                self.board.board[i] = -self.piece.bishop
                self.board.bb_bishop[1] = self.bitop.xor(i, self.board.bb_bishop[1])
                self.board.bb_bh[1] = self.bitop.xor(i, self.board.bb_bh[1])
                self.set_occupied(1, i)
                pos += 1
            elif s == 'b' and flag_promo == 1:
                self.board.board[i] = -self.piece.horse
                self.board.bb_horse[1] = self.bitop.xor(i, self.board.bb_horse[1])
                self.board.bb_bh[1] = self.bitop.xor(i, self.board.bb_bh[1])
                self.board.bb_hdk[1] = self.bitop.xor(i, self.board.bb_hdk[1])
                self.set_occupied(1, i)
                flag_promo = 0
                pos += 1
            elif s == 'r' and flag_promo == 0:
                self.board.board[i] = -self.piece.rook
                self.board.bb_rook[1] = self.bitop.xor(i, self.board.bb_rook[1])
                self.board.bb_rd[1] = self.bitop.xor(i, self.board.bb_rd[1])
                self.set_occupied(1, i)
                pos += 1
            elif s == 'r' and flag_promo == 1:
                self.board.board[i] = -self.piece.dragon
                self.board.bb_dragon[1] = self.bitop.xor(i, self.board.bb_dragon[1])
                self.board.bb_rd[1] = self.bitop.xor(i, self.board.bb_rd[1])
                self.board.bb_hdk[1] = self.bitop.xor(i, self.board.bb_hdk[1])
                self.set_occupied(1, i)
                flag_promo = 0
                pos += 1
            elif s == 'k':
                self.board.board[i] = -self.piece.king
                self.board.sq_king[1] = i
                self.board.bb_hdk[1] = self.bitop.xor(i, self.board.bb_hdk[1])
                self.set_occupied(1, i)
                pos += 1
            elif s == '1' or s == '2' or s == '3' or s == '4' or s == '5' or s == '6' or s == '7' or s == '8' or s == '9':
                j = int(s)
                for k in range(j):
                    self.board.board[i] = self.piece.empty
                    i += 1
                pos += 1
                continue
            elif s == '+':
                flag_promo = 1
                pos += 1
                continue
            elif s == '/':
                pos += 1
                continue
            else:
                pos += 1
                continue
            i += 1

        #手番のparse
        if str_temp[1] == 'b':
            self.board.color = self.color.black
        else:
            self.board.color = self.color.white

        #持ち駒のparse
        str_pos = str_temp[2]
        length = len(str_pos)
        pos = 0
        hand_count = 0
        flag = 0
        while pos < length:
            s = str_pos[pos]
            if s == '-':
                break
            elif s == '0':
                if flag == 1:
                    hand_count = 10
                flag = 0
            elif s == '1':
                flag = 1
            elif s == '2':
                if flag == 0:
                    hand_count = 2
                else:
                    hand_count = 12
                flag = 0
            elif s == '3':
                if flag == 0:
                    hand_count = 3
                else:
                    hand_count = 13
                flag = 0
            elif s == '4':
                if flag == 0:
                    hand_count = 4
                else:
                    hand_count = 14
                flag = 0
            elif s == '5':
                if flag == 0:
                    hand_count = 5
                else:
                    hand_count = 15
                flag = 0
            elif s == '6':
                if flag == 0:
                    hand_count = 6
                else:
                    hand_count = 16
                flag = 0
            elif s == '7':
                if flag == 0:
                    hand_count = 7
                else:
                    hand_count = 17
                flag = 0
            elif s == '8':
                if flag == 0:
                    hand_count = 8
                else:
                    hand_count = 18
                flag = 0
            elif s == '9':
                hand_count = 9
            elif s == 'P':
                if hand_count > 1:
                    self.board.hand_pawn[0] = hand_count
                else:
                    self.board.hand_pawn[0] = 1
                hand_count = 0
            elif s == 'L':
                if hand_count > 1:
                    self.board.hand_lance[0] = hand_count
                else:
                    self.board.hand_lance[0] = 1
                hand_count = 0
            elif s == 'N':
                if hand_count > 1:
                    self.board.hand_knight[0] = hand_count
                else:
                    self.board.hand_knight[0] = 1
                hand_count = 0
            elif s == 'S':
                if hand_count > 1:
                    self.board.hand_silver[0] = hand_count
                else:
                    self.board.hand_silver[0] = 1
                hand_count = 0
            elif s == 'G':
                if hand_count > 1:
                    self.board.hand_gold[0] = hand_count
                else:
                    self.board.hand_gold[0] = 1
                hand_count = 0
            elif s == 'B':
                if hand_count > 1:
                    self.board.hand_bishop[0] = hand_count
                else:
                    self.board.hand_bishop[0] = 1
                hand_count = 0
            elif s == 'R':
                if hand_count > 1:
                    self.board.hand_rook[0] = hand_count
                else:
                    self.board.hand_rook[0] = 1
                hand_count = 0
            elif s == 'p':
                if hand_count > 1:
                    self.board.hand_pawn[1] = hand_count
                else:
                    self.board.hand_pawn[1] = 1
                hand_count = 0
            elif s == 'l':
                if hand_count > 1:
                    self.board.hand_lance[1] = hand_count
                else:
                    self.board.hand_lance[1] = 1
                hand_count = 0
            elif s == 'n':
                if hand_count > 1:
                    self.board.hand_knight[1] = hand_count
                else:
                    self.board.hand_knight[1] = 1
                hand_count = 0
            elif s == 's':
                if hand_count > 1:
                    self.board.hand_silver[1] = hand_count
                else:
                    self.board.hand_silver[1] = 1
                hand_count = 0
            elif s == 'g':
                if hand_count > 1:
                    self.board.hand_gold[1] = hand_count
                else:
                    self.board.hand_gold[1] = 1
                hand_count = 0
            elif s == 'b':
                if hand_count > 1:
                    self.board.hand_bishop[1] = hand_count
                else:
                    self.board.hand_bishop[1] = 1
                hand_count = 0
            elif s == 'r':
                if hand_count > 1:
                    self.board.hand_rook[1] = hand_count
                else:
                    self.board.hand_rook[1] = 1
                hand_count = 0

            pos += 1

        if len(str_temp) == 4:
            return

        #指し手のparse
        ma = makemove.makemove()
        cls_usi = usi.usi()
        c = self.board.color
        ply = 0
        for i in range(5, len(str_temp)):
            str_move = str_temp[i]
            m = move.move(self.board, self.piece)
            if len(str_move) == 5:
                m.flag_promo = 1
            cls_usi.usi_to_move(self.board, self.piece, str_move, m)
            ma.makemove(self.board, m, ply, self.bitop, self.piece, c)
            c ^= 1
            ply += 1

    def set_occupied(self, color, i):
        self.board.bb_occ_color[color] = self.bitop.xor(i, self.board.bb_occ_color[color])
        self.board.bb_occupied = self.bitop.xor(i, self.board.bb_occupied)