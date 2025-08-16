import move

class gendrop:
    def __init__(self, board, bitop, pc, attack, color):
        self.bo = board
        self.bi = bitop
        self.pc = pc
        self.at = attack
        self.at.cls_board = self.bo
        self.co = color

    #先手が駒を打つ手を生成する
    def b_gen_drop(self, moves):

        #持ち駒がない場合はここで終了
        if self.bo.hand_pawn[0] == 0 and self.bo.hand_lance[0] == 0 and self.bo.hand_knight[0] == 0 and self.bo.hand_silver[0] == 0 and self.bo.hand_gold[0] == 0 and self.bo.hand_bishop[0] == 0 and self.bo.hand_rook[0] == 0:
            return

        bb_target = self.bi.bb_not_and(self.bi.bb_full, self.bo.bb_occupied)
        ibb_target0a = self.bi.bb_full & ((0x7fc0000) << 54) & bb_target
        ibb_target0b = self.bi.bb_full & ((0x003fe00) << 54) & bb_target
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
            while ibb_target0b != 0:
                ito = self.bi.last_one(ibb_target0b)
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
                ibb_target0b = self.bi.xor(ito, ibb_target0b)
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
            while ibb_target0b != 0:
                ito = self.bi.last_one(ibb_target0b)
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
                ibb_target0b = self.bi.xor(ito, ibb_target0b)
        while ibb_target0a != 0:
            ito = self.bi.last_one(ibb_target0a)
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
            ibb_target0a = self.bi.xor(ito, ibb_target0a)

    #後手が駒を打つ手を生成する
    def w_gen_drop(self, moves):

        #持ち駒がない場合はここで終了
        if self.bo.hand_pawn[1] == 0 and self.bo.hand_lance[1] == 0 and self.bo.hand_knight[1] == 0 and self.bo.hand_silver[1] == 0 and self.bo.hand_gold[1] == 0 and self.bo.hand_bishop[1] == 0 and self.bo.hand_rook[1] == 0:
            return

        bb_target = self.bi.bb_not_and(self.bi.bb_full, self.bo.bb_occupied)
        ibb_target2a = self.bi.bb_full & (0x1ff) & bb_target
        ibb_target2b = self.bi.bb_full & (0x003fe00) & bb_target
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
            while ibb_target2b != 0:
                ito = self.bi.first_one(ibb_target2b)
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
                ibb_target2b = self.bi.xor(ito, ibb_target2b)
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
            while ibb_target2b != 0:
                ito = self.bi.first_one(ibb_target2b)
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
                ibb_target2b = self.bi.xor(ito, ibb_target2b)
        while ibb_target2a != 0:
            ito = self.bi.first_one(ibb_target2a)
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
            ibb_target2a = self.bi.xor(ito, ibb_target2a)