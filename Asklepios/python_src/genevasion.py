import move

class genevasion:
    def __init__(self, board, bitop, pc, attack, color):
        self.bo = board
        self.bi = bitop
        self.pc = pc
        self.at = attack
        self.at.cls_board = self.bo
        self.co = color

    #先手が王手を避ける手を生成する
    def b_gen_evasion(self, moves):
        sq_bk = self.bo.sq_king[0]
        bb_boccupy = self.bi.xor(sq_bk, self.bo.bb_occ_color[0])
        bb_desti = self.bi.bb_not_and(self.at.abb_king_attacks[sq_bk], bb_boccupy)

        self.bo.bb_occupied = self.bi.xor(sq_bk, self.bo.bb_occupied)

        #Kingを動かす手
        while bb_desti != 0:
            ito = self.bi.last_one(bb_desti)
            if self.at.is_attacked(self.bo, ito, self.co.white) == 0:
                m = move.move(self.bo, self.pc)
                m.ifrom = sq_bk
                m.piece_to_move = self.pc.king
                m.ito = ito
                m.cap_to_move = abs(self.bo.board[ito])
                moves.append(m)
            bb_desti = self.bi.xor(ito, bb_desti)

        self.bo.bb_occupied = self.bi.xor(sq_bk, self.bo.bb_occupied)

        #両王手の場合は駒を動かすしかないので、return
        bb_checker = self.at.w_attacks_to_piece(self.bo, sq_bk)
        nchecker = self.bi.popu_count(bb_checker)
        if nchecker == 2:
            return

        sq_check = self.bi.last_one(bb_checker)
        bb_inter = self.at.abb_obstacle[sq_bk][sq_check]

        #他の駒を動かす（王手している駒を取る、王手している龍・飛・馬・角の間に駒を動かす）
        bb_target = self.bi.bb_or(bb_inter, bb_checker)

        #歩を動かす手
        bb_desti = self.bi.bb_and(bb_target, self.bo.bb_pawn_attacks[0])
        while bb_desti != 0:
            ito = self.bi.last_one(bb_desti)
            bb_desti = self.bi.xor(ito, bb_desti)
            ifrom = ito + 9
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:
                m = move.move(self.bo, self.pc)
                if ito < 27:
                    self.set_move(m, ifrom, ito, self.pc.pawn, 1)
                else:
                    self.set_move(m, ifrom, ito, self.pc.pawn, 0)
                moves.append(m)

        #香を動かす手
        bb_piece = self.bo.bb_lance[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_file_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_and(bb_desti, self.at.abb_minus_rays[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_target)
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:
                ito = self.bi.last_one(bb_desti)
                if ito < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 1)
                    moves.append(m)
                if ito >= 18:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 0)
                    moves.append(m)

        #桂を動かす手
        bb_piece = self.bo.bb_knight[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_target, self.at.abb_knight_attacks[0][ifrom])
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 1)
                    moves.append(m)
                if ito >= 18:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 0)
                    moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.last_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    if ito < 27:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.knight, 1)
                        moves.append(m)
                    if ito >= 18:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.knight, 0)
                        moves.append(m)

        #銀を動かす手
        bb_piece = self.bo.bb_silver[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_target, self.at.abb_silver_attacks[0][ifrom])
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.silver, 0)
                moves.append(m)
                if ifrom < 27 or ito < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.silver, 1)
                    moves.append(m)
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

        #金・成金を動かす手
        bb_piece = self.bo.bb_total_gold[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_target, self.at.abb_gold_attacks[0][ifrom])
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, abs(self.bo.board[ifrom]), 0)
                moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.last_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, abs(self.bo.board[ifrom]), 0)
                    moves.append(m)

        #角を動かす手
        bb_piece = self.bo.bb_bishop[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_and(bb_desti, bb_target)            
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ifrom < 27 or ito < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.bishop, 1)
                    moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.bishop, 0)
                    moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.last_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    if ifrom < 27 or ito < 27:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.bishop, 1)
                        moves.append(m)
                    else:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.bishop, 0)
                        moves.append(m)

        #飛を動かす手
        bb_piece = self.bo.bb_rook[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_and(bb_desti, bb_target)            
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)

                if ifrom < 27 or ito < 27:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.rook, 1)
                    moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.rook, 0)
                    moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.last_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    if ifrom < 27 or ito < 27:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.rook, 1)
                        moves.append(m)
                    else:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.rook, 0)
                        moves.append(m)

        #馬を動かす手
        bb_piece = self.bo.bb_horse[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_or(bb_desti, self.at.abb_king_attacks[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_target)            
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.horse, 0)
                moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.last_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.horse, 0)
                    moves.append(m)

        #龍を動かす手
        bb_piece = self.bo.bb_dragon[0]
        while bb_piece != 0:
            ifrom = self.bi.last_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_or(bb_desti, self.at.abb_king_attacks[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_target)            
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_bk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.black) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.last_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.dragon, 0)
                moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.last_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.dragon, 0)
                    moves.append(m)

        #持ち駒がない場合はここで終了
        if self.bo.hand_pawn[0] == 0 and self.bo.hand_lance[0] == 0 and self.bo.hand_knight[0] == 0 and self.bo.hand_silver[0] == 0 and self.bo.hand_gold[0] == 0 and self.bo.hand_bishop[0] == 0 and self.bo.hand_rook[0] == 0:
            return

        #合駒できない場合はここで終了
        if bb_inter == 0:
            return

        #駒打ちの手（合駒）
        bb_target = bb_inter
        ubb_target0a = bb_target & ((0x7fc0000) << 54)
        ubb_target0b = bb_target & ((0x003fe00) << 54)
        bb_target &= 0x7FFFFFFFFFFFFFFF
        
        #歩は二歩と打ち歩詰めにならないようにする
        if self.bo.hand_pawn[0] > 0:
            ais_pawn = []
            for i in range(self.bo.nfile):
                temp = self.bo.bb_pawn_attacks[0] & (self.bi.mask_file1 >> i)
                ais_pawn.append(temp)
                
            while bb_target != 0:
                ito = self.bi.last_one(bb_target)
                if ais_pawn[self.bo.file_table[ito]] == 0 and self.at.is_mate_pawn_drop(self.bo, ito, self.co.black) != 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.pawn
                    moves.append(m)
                if self.bo.hand_lance[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.lance
                    moves.append(m)
                if self.bo.hand_knight[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.knight
                    moves.append(m)
                if self.bo.hand_silver[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.silver
                    moves.append(m)
                if self.bo.hand_gold[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.gold
                    moves.append(m)
                if self.bo.hand_bishop[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.bishop
                    moves.append(m)
                if self.bo.hand_rook[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.rook
                    moves.append(m)
                bb_target = self.bi.xor(ito, bb_target)
            while ubb_target0b != 0:
                ito = self.bi.last_one(ubb_target0b)
                if ais_pawn[self.bo.file_table[ito]] == 0 and self.at.is_mate_pawn_drop(self.bo, ito, self.co.black) != 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.pawn
                    moves.append(m)
                if self.bo.hand_lance[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.lance
                    moves.append(m)
                if self.bo.hand_silver[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.silver
                    moves.append(m)
                if self.bo.hand_gold[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.gold
                    moves.append(m)
                if self.bo.hand_bishop[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.bishop
                    moves.append(m)
                if self.bo.hand_rook[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.rook
                    moves.append(m)
                ubb_target0b = self.bi.xor(ito, ubb_target0b)
        else:
            while bb_target != 0:
                ito = self.bi.last_one(bb_target)
                if self.bo.hand_lance[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.lance
                    moves.append(m)
                if self.bo.hand_knight[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.knight
                    moves.append(m)
                if self.bo.hand_silver[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.silver
                    moves.append(m)
                if self.bo.hand_gold[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.gold
                    moves.append(m)
                if self.bo.hand_bishop[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.bishop
                    moves.append(m)
                if self.bo.hand_rook[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.rook
                    moves.append(m)
                bb_target = self.bi.xor(ito, bb_target)
            while ubb_target0b != 0:
                ito = self.bi.last_one(ubb_target0b)
                if self.bo.hand_lance[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.lance
                    moves.append(m)
                if self.bo.hand_silver[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.silver
                    moves.append(m)
                if self.bo.hand_gold[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.gold
                    moves.append(m)
                if self.bo.hand_bishop[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.bishop
                    moves.append(m)
                if self.bo.hand_rook[0] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.rook
                    moves.append(m)
                ubb_target0b = self.bi.xor(ito, ubb_target0b)
        while ubb_target0a != 0:
            ito = self.bi.last_one(ubb_target0a)
            if self.bo.hand_silver[0] > 0:
                m = move.move(self.bo, self.pc)
                m.ito = ito
                m.piece_to_move = self.pc.silver
                moves.append(m)
            if self.bo.hand_gold[0] > 0:
                m = move.move(self.bo, self.pc)
                m.ito = ito
                m.piece_to_move = self.pc.gold
                moves.append(m)
            if self.bo.hand_bishop[0] > 0:
                m = move.move(self.bo, self.pc)
                m.ito = ito
                m.piece_to_move = self.pc.bishop
                moves.append(m)
            if self.bo.hand_rook[0] > 0:
                m = move.move(self.bo, self.pc)
                m.ito = ito
                m.piece_to_move = self.pc.rook
                moves.append(m)
            ubb_target0a = self.bi.xor(ito, ubb_target0a)

    #後手が王手を避ける手を生成する
    def w_gen_evasion(self, moves):
        sq_wk = self.bo.sq_king[1]
        bb_woccupy = self.bi.xor(sq_wk, self.bo.bb_occ_color[1])
        bb_desti = self.bi.bb_not_and(self.at.abb_king_attacks[sq_wk], bb_woccupy)

        self.bo.bb_occupied = self.bi.xor(sq_wk, self.bo.bb_occupied)

        #Kingを動かす手
        while bb_desti != 0:
            ito = self.bi.first_one(bb_desti)
            if self.at.is_attacked(self.bo, ito, self.co.black) == 0:
                m = move.move(self.bo, self.pc)
                m.ifrom = sq_wk
                m.piece_to_move = self.pc.king
                m.ito = ito
                m.cap_to_move = abs(self.bo.board[ito])
                moves.append(m)
            bb_desti = self.bi.xor(ito, bb_desti)

        self.bo.bb_occupied = self.bi.xor(sq_wk, self.bo.bb_occupied)

        #両王手の場合は駒を動かすしかないので、return
        bb_checker = self.at.b_attacks_to_piece(self.bo, sq_wk)
        nchecker = self.bi.popu_count(bb_checker)
        if nchecker == 2:
            return

        sq_check = self.bi.first_one(bb_checker)
        bb_inter = self.at.abb_obstacle[sq_wk][sq_check]

        #他の駒を動かす（王手している駒を取る、王手している龍・飛・馬・角の間に駒を動かす）
        bb_target = self.bi.bb_or(bb_inter, bb_checker)

        #歩を動かす手
        bb_desti = self.bi.bb_and(bb_target, self.bo.bb_pawn_attacks[1])
        while bb_desti != 0:
            ito = self.bi.first_one(bb_desti)
            bb_desti = self.bi.xor(ito, bb_desti)
            ifrom = ito - 9
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:
                m = move.move(self.bo, self.pc)
                if ito > 53:
                    self.set_move(m, ifrom, ito, self.pc.pawn, 1)
                else:
                    self.set_move(m, ifrom, ito, self.pc.pawn, 0)
                moves.append(m)

        #香を動かす手
        bb_piece = self.bo.bb_lance[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_file_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_and(bb_desti, self.at.abb_plus_rays[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_target)
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:
                ito = self.bi.first_one(bb_desti)
                if ito > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 1)
                    moves.append(m)
                if ito <= 62:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.lance, 0)
                    moves.append(m)

        #桂を動かす手
        bb_piece = self.bo.bb_knight[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_target, self.at.abb_knight_attacks[1][ifrom])
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ito > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 1)
                    moves.append(m)
                if ito <= 62:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.knight, 0)
                    moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.first_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    if ito > 53:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.knight, 1)
                        moves.append(m)
                    if ito <= 62:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.knight, 0)
                        moves.append(m)

        #銀を動かす手
        bb_piece = self.bo.bb_silver[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_target, self.at.abb_silver_attacks[1][ifrom])
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.silver, 0)
                moves.append(m)
                if ifrom > 53 or ito > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.silver, 1)
                    moves.append(m)
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

        #金・成金を動かす手
        bb_piece = self.bo.bb_total_gold[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.bi.bb_and(bb_target, self.at.abb_gold_attacks[1][ifrom])
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, abs(self.bo.board[ifrom]), 0)
                moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.first_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, abs(self.bo.board[ifrom]), 0)
                    moves.append(m)

        #角を動かす手
        bb_piece = self.bo.bb_bishop[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_and(bb_desti, bb_target)            
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ifrom > 53 or ito > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.bishop, 1)
                    moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.bishop, 0)
                    moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.last_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    if ifrom > 53 or ito > 53:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.bishop, 1)
                        moves.append(m)
                    else:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.bishop, 0)
                        moves.append(m)

        #飛を動かす手
        bb_piece = self.bo.bb_rook[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_and(bb_desti, bb_target)            
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                if ifrom > 53 or ito > 53:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.rook, 1)
                    moves.append(m)
                else:
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.rook, 0)
                    moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.first_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    if ifrom > 53 or ito > 53:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.rook, 1)
                        moves.append(m)
                    else:
                        m = move.move(self.bo, self.pc)
                        self.set_move(m, ifrom, ito, self.pc.rook, 0)
                        moves.append(m)

        #馬を動かす手
        bb_piece = self.bo.bb_horse[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_bishop_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_or(bb_desti, self.at.abb_king_attacks[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_target)            
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.horse, 0)
                moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.first_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.horse, 0)
                    moves.append(m)

        #龍を動かす手
        bb_piece = self.bo.bb_dragon[1]
        while bb_piece != 0:
            ifrom = self.bi.first_one(bb_piece)
            bb_piece = self.bi.xor(ifrom, bb_piece)
            bb_desti = self.at.get_rook_attacks(self.bo.bb_occupied, ifrom)
            bb_desti = self.bi.bb_or(bb_desti, self.at.abb_king_attacks[ifrom])
            bb_desti = self.bi.bb_and(bb_desti, bb_target)            
            if bb_desti == 0:
                continue
            idirec = self.at.adirec[sq_wk][ifrom]
            if idirec == 0 or self.at.is_pinned_on_king(self.bo, ifrom, idirec, self.co.white) == 0:#Pythonはdo~whileがないのでこの形にしている
                ito = self.bi.first_one(bb_desti)
                bb_desti = self.bi.xor(ito, bb_desti)
                m = move.move(self.bo, self.pc)
                self.set_move(m, ifrom, ito, self.pc.dragon, 0)
                moves.append(m)
                while bb_desti != 0:
                    ito = self.bi.first_one(bb_desti)
                    bb_desti = self.bi.xor(ito, bb_desti)
                    m = move.move(self.bo, self.pc)
                    self.set_move(m, ifrom, ito, self.pc.dragon, 0)
                    moves.append(m)

        #持ち駒がない場合はここで終了
        if self.bo.hand_pawn[1] == 0 and self.bo.hand_lance[1] == 0 and self.bo.hand_knight[1] == 0 and self.bo.hand_silver[1] == 0 and self.bo.hand_gold[1] == 0 and self.bo.hand_bishop[1] == 0 and self.bo.hand_rook[1] == 0:
            return

        #合駒できない場合はここで終了
        if bb_inter == 0:
            return

        #駒打ちの手（合駒）
        bb_target = bb_inter
        ubb_target2a = bb_target & (0x1ff)
        ubb_target2b = bb_target & (0x003fe00)
        bb_target &= (0x7FFFFFFFFFFFFFFF << 18)

        #歩は二歩と打ち歩詰めにならないようにする
        if self.bo.hand_pawn[1] > 0:
            ais_pawn = []
            for i in range(self.bo.nfile):
                temp = self.bo.bb_pawn_attacks[1] & (self.bi.mask_file1 >> i)
                ais_pawn.append(temp)
                
            while bb_target != 0:
                ito = self.bi.first_one(bb_target)
                if ais_pawn[self.bo.file_table[ito]] == 0 and self.at.is_mate_pawn_drop(self.bo, ito, self.co.white) != 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.pawn
                    moves.append(m)
                if self.bo.hand_lance[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.lance
                    moves.append(m)
                if self.bo.hand_knight[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.knight
                    moves.append(m)
                if self.bo.hand_silver[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.silver
                    moves.append(m)
                if self.bo.hand_gold[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.gold
                    moves.append(m)
                if self.bo.hand_bishop[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.bishop
                    moves.append(m)
                if self.bo.hand_rook[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.rook
                    moves.append(m)
                bb_target = self.bi.xor(ito, bb_target)
            while ubb_target2b != 0:
                ito = self.bi.first_one(ubb_target2b)
                if ais_pawn[self.bo.file_table[ito]] == 0 and self.at.is_mate_pawn_drop(self.bo, ito, self.co.white) != 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.pawn
                    moves.append(m)
                if self.bo.hand_lance[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.lance
                    moves.append(m)
                if self.bo.hand_silver[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.silver
                    moves.append(m)
                if self.bo.hand_gold[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.gold
                    moves.append(m)
                if self.bo.hand_bishop[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.bishop
                    moves.append(m)
                if self.bo.hand_rook[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.rook
                    moves.append(m)
                ubb_target2b = self.bi.xor(ito, ubb_target2b)
        else:
            while bb_target != 0:
                ito = self.bi.first_one(bb_target)
                if self.bo.hand_lance[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.lance
                    moves.append(m)
                if self.bo.hand_knight[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.knight
                    moves.append(m)
                if self.bo.hand_silver[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.silver
                    moves.append(m)
                if self.bo.hand_gold[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.gold
                    moves.append(m)
                if self.bo.hand_bishop[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.bishop
                    moves.append(m)
                if self.bo.hand_rook[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.rook
                    moves.append(m)
                bb_target = self.bi.xor(ito, bb_target)
            while ubb_target2b != 0:
                ito = self.bi.first_one(ubb_target2b)
                if self.bo.hand_lance[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.lance
                    moves.append(m)
                if self.bo.hand_silver[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.silver
                    moves.append(m)
                if self.bo.hand_gold[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.gold
                    moves.append(m)
                if self.bo.hand_bishop[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.bishop
                    moves.append(m)
                if self.bo.hand_rook[1] > 0:
                    m = move.move(self.bo, self.pc)
                    m.ito = ito
                    m.piece_to_move = self.pc.rook
                    moves.append(m)
                ubb_target2b = self.bi.xor(ito, ubb_target2b)
        while ubb_target2a != 0:
            ito = self.bi.first_one(ubb_target2a)
            if self.bo.hand_silver[1] > 0:
                m = move.move(self.bo, self.pc)
                m.ito = ito
                m.piece_to_move = self.pc.silver
                moves.append(m)
            if self.bo.hand_gold[1] > 0:
                m = move.move(self.bo, self.pc)
                m.ito = ito
                m.piece_to_move = self.pc.gold
                moves.append(m)
            if self.bo.hand_bishop[1] > 0:
                m = move.move(self.bo, self.pc)
                m.ito = ito
                m.piece_to_move = self.pc.bishop
                moves.append(m)
            if self.bo.hand_rook[1] > 0:
                m = move.move(self.bo, self.pc)
                m.ito = ito
                m.piece_to_move = self.pc.rook
                moves.append(m)
            ubb_target2a = self.bi.xor(ito, ubb_target2a)

    def set_move(self, m, ifrom, ito, pc, promo):
        m.ifrom = ifrom
        m.ito = ito
        m.piece_to_move = pc
        m.cap_to_move = abs(self.bo.board[ito])
        m.flag_promo = promo