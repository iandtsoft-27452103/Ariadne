import move

class gennocap:
    def __init__(self, board, bitop, pc, attack, color):
        self.bo = board
        self.bi = bitop
        self.pc = pc
        self.at = attack
        self.co = color

    #先手の駒を取らない手の生成
    def b_gen_nocaptures(self, moves):
        bb_full = self.bi.bb_full
        bb_empty = self.bi.bb_not_and(bb_full, self.bo.bb_occupied)
        
        #歩を動かす手（成りは生成しない）
        bb_mask = 0x3FFFFFFFFFFFFF
        bb_piece = self.bo.bb_pawn_attacks[0] & bb_mask & bb_empty
        while bb_piece != 0:
            ito = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ito, bb_piece)
            ifrom = ito + 9
            m = move.move(self.bo, self.pc)
            self.set_move(m, ifrom, ito, self.pc.pawn, 0)
            moves.append(m)

        #銀を動かす手（成りも生成する）
        bb_piece = self.bo.bb_silver[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_empty, self.at.abb_silver_attacks[self.co.black][ifrom])
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.silver, 0)
                moves.append(m)
                if ifrom < 27 or ito < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.silver, 1)
                    moves.append(m)

        #金、成金を動かす手
        self.gen_gold_moves(moves, self.co.black, bb_empty)

        #玉を動かす手
        self.gen_king_moves(moves, self.co.black, bb_empty)

        #角を動かす手（成りは生成しない）
        self.gen_bishop_moves(moves, self.co.black, bb_empty, bb_mask)
        
        #飛を動かす手（成りは生成しない）
        self.gen_rook_moves(moves, self.co.black, bb_empty, bb_mask)

        #馬を動かす手
        self.gen_horse_moves(moves, self.co.black, bb_empty)

        #龍を動かす手
        self.gen_dragon_moves(moves, self.co.black, bb_empty)

        #香を動かす手（成りは生成しない）
        bb_mask = 0x7FFFFFFFFFFFFFFF
        self.gen_lance_moves(moves, self.co.black, bb_empty, bb_mask)

        #桂を動かす手（成りは生成しない）
        self.gen_knight_moves(moves, self.co.black, bb_empty, bb_mask)
    
    #後手の駒を取らない手の生成
    def w_gen_nocaptures(self, moves):
        bb_full = self.bi.bb_full
        bb_empty = self.bi.bb_not_and(bb_full, self.bo.bb_occupied)

        #歩を動かす手（成りは生成しない）
        bb_mask = bb_full ^ 0x7FFFFFF
        bb_piece = self.bo.bb_pawn_attacks[1] & bb_mask & bb_empty
        while bb_piece != 0:
            ito = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ito, bb_piece)
            ifrom = ito - 9
            m = move.move(self.bo, self.pc)
            self.set_move(m, ifrom, ito, self.pc.pawn, 0)
            moves.append(m)

        #銀を動かす手（成りも生成する）
        bb_piece = self.bo.bb_silver[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_empty, self.at.abb_silver_attacks[self.co.white][ifrom])
            while bb_desti != 0:
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.silver, 0)
                moves.append(m)
                if ifrom > 53 or ito > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.silver, 1)
                    moves.append(m)

        #金、成金を動かす手
        self.gen_gold_moves(moves, self.co.white, bb_empty)

        #玉を動かす手
        self.gen_king_moves(moves, self.co.white, bb_empty)

        #角を動かす手（成りは生成しない）
        self.gen_bishop_moves(moves, self.co.white, bb_empty, bb_mask)
        
        #飛を動かす手（成りは生成しない）
        self.gen_rook_moves(moves, self.co.white, bb_empty, bb_mask)

        #馬を動かす手
        self.gen_horse_moves(moves, self.co.white, bb_empty)

        #龍を動かす手
        self.gen_dragon_moves(moves, self.co.white, bb_empty)

        #香を動かす手（成りは生成しない）
        bb_mask = bb_full ^ 0x3FFFF
        self.gen_lance_moves(moves, self.co.white, bb_empty, bb_mask)

        #桂を動かす手（成りは生成しない）
        self.gen_knight_moves(moves, self.co.white, bb_empty, bb_mask)

    def set_move(self, m, ifrom, ito, pc, promo):
        m.ifrom = ifrom
        m.ito = ito
        m.piece_to_move = pc
        m.flag_promo = promo

    #金、成金を動かす手
    def gen_gold_moves(self, moves, color, bb_empty):
        bb_piece = self.bo.bb_total_gold[color]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_empty, self.at.abb_gold_attacks[color][ifrom])
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, abs(self.bo.board[ifrom]), 0)
                moves.append(m)

    #玉を動かす手
    def gen_king_moves(self, moves, color, bb_empty):
        ifrom = self.bo.sq_king[color]
        bb_desti = self.bi.bb_and(bb_empty, self.at.abb_king_attacks[ifrom])
        while bb_desti != 0:
            ito = self.bi.last_one(bb_desti)
            bb_desti = self.bi.xor(ito, bb_desti)
            m = move.move(self.bo, self.pc)
            self.set_move(m, ifrom, ito, self.pc.king, 0)
            moves.append(m)

    #角を動かす手（成りは生成しない）
    def gen_bishop_moves(self, moves, color, bb_empty, bb_mask):
        bb_piece = self.bo.bb_bishop[color] & bb_mask
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_and(bb_desti, bb_empty)
            bb_desti = self.bi.bb_and(bb_desti, bb_mask)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.bishop, 0)
                moves.append(m)

    #飛を動かす手（成りは生成しない）
    def gen_rook_moves(self, moves, color, bb_empty, bb_mask):
        bb_piece = self.bo.bb_rook[color] & bb_mask
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_and(bb_desti, bb_empty)
            bb_desti = self.bi.bb_and(bb_desti, bb_mask)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.rook, 0)
                moves.append(m)

    #馬を動かす手
    def gen_horse_moves(self, moves, color, bb_empty):
        bb_piece = self.bo.bb_horse[color]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_or(bb_desti, self.at.abb_king_attacks[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_empty)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.horse, 0)
                moves.append(m)

    #龍を動かす手
    def gen_dragon_moves(self, moves, color, bb_empty):
        bb_piece = self.bo.bb_dragon[color]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_or(bb_desti, self.at.abb_king_attacks[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_empty)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.dragon, 0)
                moves.append(m)

    #香を動かす手（成りは生成しない）
    def gen_lance_moves(self, moves, color, bb_empty, bb_mask):
        bb_piece = self.bo.bb_lance[color] & bb_mask
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_file_attacks(self.bo.bb_occupied, ifrom)
            if color == 0:
                bb_rays = self.at.abb_minus_rays[ifrom]
            else:
                bb_rays = self.at.abb_plus_rays[ifrom]
            bb_desti = self.bi.bb_and(bb_desti, bb_rays)
            bb_desti = self.bi.bb_and(bb_desti, bb_empty)
            bb_desti = self.bi.bb_and(bb_desti, bb_mask)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.lance, 0)
                moves.append(m)

    #桂を動かす手（成りは生成しない）
    def gen_knight_moves(self, moves, color, bb_empty, bb_mask):
        bb_piece = self.bo.bb_knight[color] & bb_mask
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_empty, self.at.abb_knight_attacks[color][ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_mask)
            while bb_desti != 0:
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.knight, 0)
                moves.append(m)