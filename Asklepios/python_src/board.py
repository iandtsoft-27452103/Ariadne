import piece
import color

class board:
    def __init__(self):
        #piece全部をまとめてもよかったが、分かりにくいので分けた
        self.bb_pawn = [0, 0]
        self.bb_pawn_attacks = [0, 0]
        self.bb_lance = [0, 0]
        self.bb_knight = [0, 0]
        self.bb_silver = [0, 0]
        self.bb_gold = [0, 0]
        self.bb_bishop = [0, 0]
        self.bb_rook = [0, 0]
        self.bb_pro_pawn = [0, 0]
        self.bb_pro_lance = [0, 0]
        self.bb_pro_knight = [0, 0]
        self.bb_pro_silver = [0, 0]
        self.bb_horse = [0, 0]
        self.bb_dragon = [0, 0]
        self.bb_total_gold = [0, 0]
        self.bb_bh = [0, 0]
        self.bb_rd = [0, 0]
        self.bb_hdk = [0, 0]
        self.sq_king = [0, 0]
        self.bb_occ_color = [0, 0]
        self.bb_occupied = 0
        self.hand_pawn = [0, 0]
        self.hand_lance = [0, 0]
        self.hand_knight = [0, 0]
        self.hand_silver = [0, 0]
        self.hand_gold = [0, 0]
        self.hand_bishop = [0, 0]
        self.hand_rook = [0, 0]
        self.square_nb = 81
        pc = piece.piece()
        self.board = [pc.empty] * self.square_nb
        self.nfile = 9
        self.nrank = 9
        self.npawn = 18
        self.nlance = 4
        self.nknight = 4
        self.nsilver = 4
        self.ngold = 4
        self.nbishop = 2
        self.nrook = 2
        self.file_table = []
        self.init_file_table()
        self.rank_table = []
        self.init_rank_table()
        self.color = 0
        self.board_default = [-pc.lance, -pc.knight, -pc.silver, -pc.gold, -pc.king, -pc.gold, -pc.silver, -pc.knight, -pc.lance,
                              pc.empty, -pc.rook, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, -pc.bishop, pc.empty,
                              -pc.pawn, -pc.pawn, -pc.pawn, -pc.pawn, -pc.pawn, -pc.pawn, -pc.pawn, -pc.pawn, -pc.pawn,
                              pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty,
                              pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty,
                              pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty,
                              pc.pawn, pc.pawn, pc.pawn, pc.pawn, pc.pawn, pc.pawn, pc.pawn, pc.pawn, pc.pawn,
                              pc.empty, pc.bishop, pc.empty, pc.empty, pc.empty, pc.empty, pc.empty, pc.rook, pc.empty,
                              pc.lance, pc.knight, pc.silver, pc.gold, pc.king, pc.gold, pc.silver, pc.knight, pc.lance]
        self.hand_default = [0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0]

    def init_file_table(self):
        for r in range(self.nrank):
            for f in range(self.nfile):
                self.file_table.append(f)
    
    def init_rank_table(self):
        for r in range(self.nrank):
            for f in range(self.nfile):
                self.rank_table.append(r)

    def init_board(self, board_array, hand_array, bitop, c):
        pc = piece.piece()
        co = color.color()

        self.color = c
        self.bb_pawn = [0, 0]
        self.bb_pawn_attacks = [0, 0]
        self.bb_lance = [0, 0]
        self.bb_knight = [0, 0]
        self.bb_silver = [0, 0]
        self.bb_gold = [0, 0]
        self.bb_bishop = [0, 0]
        self.bb_rook = [0, 0]
        self.bb_pro_pawn = [0, 0]
        self.bb_pro_lance = [0, 0]
        self.bb_pro_knight = [0, 0]
        self.bb_pro_silver = [0, 0]
        self.bb_horse = [0, 0]
        self.bb_dragon = [0, 0]
        self.bb_total_gold = [0, 0]
        self.bb_bh = [0, 0]
        self.bb_rd = [0, 0]
        self.bb_hdk = [0, 0]
        self.sq_king = [0, 0]
        self.bb_occ_color = [0, 0]
        self.bb_occupied = 0
        self.hand_pawn = [0, 0]
        self.hand_lance = [0, 0]
        self.hand_knight = [0, 0]
        self.hand_silver = [0, 0]
        self.hand_gold = [0, 0]
        self.hand_bishop = [0, 0]
        self.hand_rook = [0, 0]

        for sq in range(self.square_nb):
            self.board[sq] = board_array[sq]
            if board_array[sq] > 0:
                self.init_board_child(bitop, co.black, sq, pc, 1)
            else:
                self.init_board_child(bitop, co.white, sq, pc, -1)

        self.init_hand(co.black, hand_array, 0)
        self.init_hand(co.white, hand_array, 7)

    def init_board_child(self, bitop, color, sq, pc, sign):
        if self.board[sq] == (sign * pc.pawn):
            self.bb_pawn[color] = bitop.xor(sq, self.bb_pawn[color])
            self.bb_pawn_attacks[color] = bitop.xor(sq + (-sign * 9), self.bb_pawn_attacks[color])
        elif self.board[sq] == (sign * pc.lance):
            self.bb_lance[color] = bitop.xor(sq, self.bb_lance[color])
        elif self.board[sq] == (sign * pc.knight):
            self.bb_knight[color] = bitop.xor(sq, self.bb_knight[color])
        elif self.board[sq] == (sign * pc.silver):
            self.bb_silver[color] = bitop.xor(sq, self.bb_silver[color])
        elif self.board[sq] == (sign * pc.gold):
            self.bb_gold[color] = bitop.xor(sq, self.bb_gold[color])
            self.bb_total_gold[color] = bitop.xor(sq, self.bb_total_gold[color])
        elif self.board[sq] == (sign * pc.bishop):
            self.bb_bishop[color] = bitop.xor(sq, self.bb_bishop[color])
            self.bb_bh[color] = bitop.xor(sq, self.bb_bh[color])
        elif self.board[sq] == (sign * pc.rook):
            self.bb_rook[color] = bitop.xor(sq, self.bb_rook[color])
            self.bb_rd[color] = bitop.xor(sq, self.bb_rd[color])
        elif self.board[sq] == (sign * pc.king):
            self.sq_king[color] = sq
            self.bb_hdk[color] = bitop.xor(sq, self.bb_hdk[color])
        elif self.board[sq] == (sign * pc.pro_pawn):
            self.bb_pro_pawn[color] = bitop.xor(sq, self.bb_pro_pawn[color])
            self.bb_total_gold[color] = bitop.xor(sq, self.bb_bb_total_gold[color])
        elif self.board[sq] == (sign * pc.pro_lance):
            self.bb_pro_lance[color] = bitop.xor(sq, self.bb_pro_lance[color])
            self.bb_total_gold[color] = bitop.xor(sq, self.bb_bb_total_gold[color])
        elif self.board[sq] == (sign * pc.pro_knight):
            self.bb_pro_knight[color] = bitop.xor(sq, self.bb_pro_knight[color])
            self.bb_total_gold[color] = bitop.xor(sq, self.bb_total_gold[color])
        elif self.board[sq] == (sign * pc.pro_silver):
            self.bb_pro_silver[color] = bitop.xor(sq, self.bb_pro_silver[color])
            self.bb_total_gold[color] = bitop.xor(sq, self.bb_total_gold[color])
        elif self.board[sq] == (sign * pc.horse):
            self.bb_horse[color] = bitop.xor(sq, self.bb_horse[color])
            self.bb_bh[color] = bitop.xor(sq, self.bb_bh[color])
            self.bb_hdk[color] = bitop.xor(sq, self.bb_hdk[color])
        elif self.board[sq] == (sign * pc.dragon):
            self.bb_dragon[color] = bitop.xor(sq, self.bb_dragon[color])
            self.bb_rd[color] = bitop.xor(sq, self.bb_rd[color])
            self.bb_hdk[color] = bitop.xor(sq, self.bb_hdk[color])
        
        if self.board[sq] != pc.empty:
            self.bb_occ_color[color] = bitop.xor(sq, self.bb_occ_color[color])
            self.bb_occupied = bitop.xor(sq, self.bb_occupied)

    def init_hand(self, color, hand_array, i):
        self.hand_pawn[color] = hand_array[i]
        self.hand_lance[color] = hand_array[i + 1]
        self.hand_knight[color] = hand_array[i + 2]
        self.hand_silver[color] = hand_array[i + 3]
        self.hand_gold[color] = hand_array[i + 4]
        self.hand_bishop[color] = hand_array[i + 5]
        self.hand_rook[color] = hand_array[i + 6]

    def clear_board(self):
        self.bb_pawn = [0, 0]
        self.bb_pawn_attacks = [0, 0]
        self.bb_lance = [0, 0]
        self.bb_knight = [0, 0]
        self.bb_silver = [0, 0]
        self.bb_gold = [0, 0]
        self.bb_bishop = [0, 0]
        self.bb_rook = [0, 0]
        self.bb_pro_pawn = [0, 0]
        self.bb_pro_lance = [0, 0]
        self.bb_pro_knight = [0, 0]
        self.bb_pro_silver = [0, 0]
        self.bb_horse = [0, 0]
        self.bb_dragon = [0, 0]
        self.bb_total_gold = [0, 0]
        self.bb_bh = [0, 0]
        self.bb_rd = [0, 0]
        self.bb_hdk = [0, 0]
        self.sq_king = [0, 0]
        self.bb_occ_color = [0, 0]
        self.bb_occupied = 0
        self.hand_pawn = [0, 0]
        self.hand_lance = [0, 0]
        self.hand_knight = [0, 0]
        self.hand_silver = [0, 0]
        self.hand_gold = [0, 0]
        self.hand_bishop = [0, 0]
        self.hand_rook = [0, 0]
        self.color = 0