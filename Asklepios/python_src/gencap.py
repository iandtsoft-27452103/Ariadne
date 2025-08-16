import move

class gencap:
    def __init__(self, board, bitop, pc, attack, color):
        self.bo = board
        self.bi = bitop
        self.pc = pc
        self.at = attack
        self.co = color

    #先手の駒を取る手の生成
    def b_gen_captures(self, moves):
        bb_full = self.bi.bb_full
        bb_capture = self.bo.bb_occ_color[1]
        bb_movable = self.bi.bb_not_and(bb_full, self.bo.bb_occ_color[0])

        #歩で駒を取る手（空成りも生成する）
        bb_desti0 = self.bo.bb_pawn_attacks[0] & bb_movable & ((0x7FFFFFF) << 54)
        bb_desti12 = self.bo.bb_pawn_attacks[0] & bb_capture & (0x3FFFFFFFFFFFFF)
        bb_desti = self.bi.bb_or(bb_desti0, bb_desti12)
        while bb_desti != 0:
            ito = self.bi.last_one(bb_desti)
            bb_desti = self.bi.xor(ito, bb_desti)
            ifrom = ito + 9
            m = move.move(self.bo, self.pc)
            if ito < 27:
                self.set_move(m, ifrom, ito, self.pc.pawn, 1)
            else:
                self.set_move(m, ifrom, ito, self.pc.pawn, 0)
            moves.append(m)

        #銀で駒を取る手
        bb_piece = self.bo.bb_silver[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_capture, self.at.abb_silver_attacks[0][ifrom])
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.silver, 0)
                moves.append(m)
                if ito < 27 or ifrom < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.silver, 1)
                    moves.append(m)

        #金、成金で駒を取る手
        self.gen_gold_moves(moves, self.co.black, bb_capture)

        #玉で駒を取る手
        self.gen_king_moves(moves, self.co.black, bb_capture)

        #角で駒を取る手
        bb_piece = self.bo.bb_bishop[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_attacks = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti0 = bb_attacks & bb_movable & ((0x7FFFFFF) << 54)
            if ifrom < 27:
                bb_desti12 = bb_attacks & bb_movable & (0x3FFFFFFFFFFFFF)
            else:
                bb_desti12 = bb_attacks & bb_capture & (0x3FFFFFFFFFFFFF)
            bb_desti = self.bi.bb_or(bb_desti0, bb_desti12)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito < 27 or ifrom < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.bishop, 1)
                    moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.bishop, 0)
                    moves.append(m)

        #飛で駒を取る手
        bb_piece = self.bo.bb_rook[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_attacks = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti0 = bb_attacks & bb_movable & ((0x7FFFFFF) << 54)
            if ifrom < 27:
                bb_desti12 = bb_attacks & bb_movable & (0x3FFFFFFFFFFFFF)
            else:
                bb_desti12 = bb_attacks & bb_capture & (0x3FFFFFFFFFFFFF)
            bb_desti = self.bi.bb_or(bb_desti0, bb_desti12)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito < 27 or ifrom < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.rook, 1)
                    moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.rook, 0)
                    moves.append(m)

        #馬で駒を取る手
        self.gen_horse_moves(moves, self.co.black, bb_capture)

        #龍で駒を取る手
        self.gen_dragon_moves(moves, self.co.black, bb_capture)

        #香で駒を取る手
        bb_piece = self.bo.bb_lance[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_attacks = self.at.get_file_attacks(self.bo.bb_occupied, ifrom)
            bb_attacks = self.bi.bb_and(bb_attacks, self.at.abb_minus_rays[ifrom])
            bb_desti0 = bb_attacks & bb_movable & ((0x7FFFFFF) << 54)
            bb_desti12 = bb_attacks & bb_capture & (0x3FFFFFFFFFFFFF)
            bb_desti = self.bi.bb_or(bb_desti0, bb_desti12)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito < 18:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 1)
                    moves.append(m)
                elif ito >= 18 and ito < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 1)
                    moves.append(m)
                    if self.bo.board[ito] < 0:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.lance, 0)
                        moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 0)
                    moves.append(m)

        #桂で駒を取る手
        bb_piece = self.bo.bb_knight[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_attacks = self.at.abb_knight_attacks[0][ifrom]
            bb_desti0 = bb_attacks & bb_movable & ((0x7FFFFFF) << 54)
            bb_desti12 = bb_attacks & bb_capture & (0x3FFFFFFFFFFFFF)
            bb_desti = self.bi.bb_or(bb_desti0, bb_desti12)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito < 18:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 1)
                    moves.append(m)
                elif ito >= 18 and ito < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 1)
                    moves.append(m)
                    if self.bo.board[ito] < 0:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.knight, 0)
                        moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 0)
                    moves.append(m)

    #後手の駒を取る手の生成
    def w_gen_captures(self, moves):
        bb_full = self.bi.bb_full
        bb_capture = self.bo.bb_occ_color[0]
        bb_movable = self.bi.bb_not_and(bb_full, self.bo.bb_occ_color[1])

        #歩で駒を取る手（空成りも生成する）
        bb_desti01 = self.bo.bb_pawn_attacks[1] & bb_capture & ((0x3FFFFFFFFFFFFF) << 27)
        bb_desti2 = self.bo.bb_pawn_attacks[1] & bb_movable & (0x7FFFFFF)
        bb_desti = self.bi.bb_or(bb_desti01, bb_desti2)
        while bb_desti != 0:
            ito = self.bi.first_one(bb_desti)
            bb_desti = self.bi.xor(ito, bb_desti)
            ifrom = ito - 9
            m = move.move(self.bo, self.pc)
            if ito > 53:
                self.set_move(m, ifrom, ito, self.pc.pawn, 1)
            else:
                self.set_move(m, ifrom, ito, self.pc.pawn, 0)
            moves.append(m)

        #銀で駒を取る手
        bb_piece = self.bo.bb_silver[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_capture, self.at.abb_silver_attacks[1][ifrom])
            while bb_desti != 0:
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.silver, 0)
                moves.append(m)
                if ito > 53 or ifrom > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.silver, 1)
                    moves.append(m)

        #金、成金で駒を取る手
        self.gen_gold_moves(moves, self.co.white, bb_capture)

        #玉で駒を取る手
        self.gen_king_moves(moves, self.co.white, bb_capture)

        #角で駒を取る手
        bb_piece = self.bo.bb_bishop[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_attacks = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti2 = bb_attacks & bb_movable & (0x7FFFFFF)
            if ifrom > 53:
                bb_desti01 = bb_attacks & bb_movable & ((0x3FFFFFFFFFFFFF) << 27)
            else:
                bb_desti01 = bb_attacks & bb_capture & ((0x3FFFFFFFFFFFFF) << 27)
            bb_desti = self.bi.bb_or(bb_desti2, bb_desti01)
            while bb_desti != 0:
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito > 53 or ifrom > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.bishop, 1)
                    moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.bishop, 0)
                    moves.append(m)

        #飛で駒を取る手
        bb_piece = self.bo.bb_rook[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_attacks = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti2 = bb_attacks & bb_movable & (0x7FFFFFF)
            if ifrom > 53:
                bb_desti01 = bb_attacks & bb_movable & ((0x3FFFFFFFFFFFFF) << 27)
            else:
                bb_desti01 = bb_attacks & bb_capture & ((0x3FFFFFFFFFFFFF) << 27)
            bb_desti = self.bi.bb_or(bb_desti2, bb_desti01)
            while bb_desti != 0:
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito > 53 or ifrom > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.rook, 1)
                    moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.rook, 0)
                    moves.append(m)

        #馬で駒を取る手
        self.gen_horse_moves(moves, self.co.white, bb_capture)

        #龍で駒を取る手
        self.gen_dragon_moves(moves, self.co.white, bb_capture)

        #香で駒を取る手
        bb_piece = self.bo.bb_lance[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_attacks = self.at.get_file_attacks(self.bo.bb_occupied, ifrom)
            bb_attacks = self.bi.bb_and(bb_attacks, self.at.abb_plus_rays[ifrom])
            bb_desti2 = bb_attacks & bb_movable & (0x7FFFFFF)
            bb_desti01 = bb_attacks & bb_capture & ((0x3FFFFFFFFFFFFF) << 27)
            bb_desti = self.bi.bb_or(bb_desti2, bb_desti01)
            while bb_desti != 0:
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito > 62:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 1)
                    moves.append(m)
                elif ito <= 62 and ito > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 1)
                    moves.append(m)
                    if self.bo.board[ito] > 0:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.lance, 0)
                        moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 0)
                    moves.append(m)

        #桂で駒を取る手
        bb_piece = self.bo.bb_knight[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_attacks = self.at.abb_knight_attacks[1][ifrom]
            bb_desti2 = bb_attacks & bb_movable & (0x7FFFFFF)
            bb_desti01 = bb_attacks & bb_capture & ((0x3FFFFFFFFFFFFF) << 27)
            bb_desti = self.bi.bb_or(bb_desti2, bb_desti01)
            while bb_desti != 0:
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito > 62:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 1)
                    moves.append(m)
                elif ito <= 62 and ito > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 1)
                    moves.append(m)
                    if self.bo.board[ito] > 0:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.knight, 0)
                        moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 0)
                    moves.append(m)

    def set_move(self, m, ifrom, ito, pc, promo):
        m.ifrom = ifrom
        m.ito = ito
        m.piece_to_move = pc
        m.cap_to_move = abs(self.bo.board[ito])
        m.flag_promo = promo

    #金、成金で駒を取る手
    def gen_gold_moves(self, moves, color, bb_capture):
        bb_piece = self.bo.bb_total_gold[color]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_capture, self.at.abb_gold_attacks[color][ifrom])
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, abs(self.bo.board[ifrom]), 0)
                moves.append(m)

    #玉で駒を取る手
    def gen_king_moves(self, moves, color, bb_capture):
        ifrom = self.bo.sq_king[color]
        bb_desti = self.bi.bb_and(bb_capture, self.at.abb_king_attacks[ifrom])
        while bb_desti != 0:
            ito = self.bi.last_one(bb_desti)
            bb_desti = self.bi.xor(ito, bb_desti)
            m = move.move(self.bo, self.pc)
            self.set_move(m, ifrom, ito, self.pc.king, 0)
            moves.append(m)

    #馬で駒を取る手
    def gen_horse_moves(self, moves, color, bb_capture):
        bb_piece = self.bo.bb_horse[color]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_or(bb_desti, self.at.abb_king_attacks[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_capture)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.horse, 0)
                moves.append(m)

    #龍で駒を取る手
    def gen_dragon_moves(self, moves, color, bb_capture):
        bb_piece = self.bo.bb_dragon[color]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_or(bb_desti, self.at.abb_king_attacks[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_capture)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.dragon, 0)
                moves.append(m)