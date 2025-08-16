import file
import rank
import bitop
import board
import direction
import piece

class attack:
    def __init__(self):
        self.cls_file = file.file()
        self.cls_rank = rank.rank()
        self.cls_board = board.board()
        self.cls_bitop = bitop.bitop(self.cls_board)
        self.cls_direc = direction.direction()
        self.cls_piece = piece.piece()
        self.abb_king_attacks = [0] * self.cls_board.square_nb
        self.abb_gold_attacks = [[0 for j in range(self.cls_board.square_nb)] for i in range(2)]
        self.abb_silver_attacks = [[0 for j in range(self.cls_board.square_nb)] for i in range(2)]
        self.abb_knight_attacks = [[0 for j in range(self.cls_board.square_nb)] for i in range(2)]
        self.abb_stomach_attacks = [0 for j in range(self.cls_board.square_nb)]
        self.abb_2up3sq_attacks = [[0 for j in range(self.cls_board.square_nb)] for i in range(2)]
        self.abb_2up_attacks = [[0 for j in range(self.cls_board.square_nb)] for i in range(2)]
        self.abb_3up_attacks = [[0 for j in range(self.cls_board.square_nb)] for i in range(2)]
        self.abb_diag_back_attacks = [[0 for j in range(self.cls_board.square_nb)] for i in range(2)]
        self.init_attack_tables()
        self.abb_diag1_mask_ex = [0] * self.cls_board.square_nb
        self.abb_diag2_mask_ex = [0] * self.cls_board.square_nb
        self.abb_rank_mask_ex = [0] * self.cls_board.square_nb
        self.abb_file_mask_ex = [0] * self.cls_board.square_nb
        self.init_diag1_mask_table()
        self.init_diag2_mask_table()
        self.init_rank_mask_table()
        self.init_file_mask_table()
        self.abb_diag1_attacks = [[0 for j in range(128)] for i in range(self.cls_board.square_nb)]
        self.init_diag1_attack_tables()
        self.abb_diag2_attacks = [[0 for j in range(128)] for i in range(self.cls_board.square_nb)]
        self.init_diag2_attack_tables()
        self.abb_rank_attacks = [[0 for j in range(128)] for i in range(self.cls_board.square_nb)]
        self.init_rank_attack_tables()
        self.abb_file_attacks = [[0 for j in range(128)] for i in range(self.cls_board.square_nb)]
        self.init_file_attack_tables()
        self.abb_plus_rays = [0] * self.cls_board.square_nb
        self.abb_minus_rays = [0] * self.cls_board.square_nb
        self.adirec = [[0 for j in range(self.cls_board.square_nb)] for i in range(self.cls_board.square_nb)]
        self.abb_obstacle = [[0 for j in range(self.cls_board.square_nb)] for i in range(self.cls_board.square_nb)]
        self.init_other_attack_tables()

    def init_attack_tables(self):
        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 1, ifile - 1, bb)
                bb = self.set_attacks(irank + 1, ifile + 1, bb)
                bb = self.set_attacks(irank + 1, ifile, bb)
                bb = self.set_attacks(irank - 1, ifile - 1, bb)
                bb = self.set_attacks(irank - 1, ifile + 1, bb)
                bb = self.set_attacks(irank - 1, ifile, bb)
                bb = self.set_attacks(irank, ifile - 1, bb)
                bb = self.set_attacks(irank, ifile + 1, bb)
                self.abb_king_attacks[irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 1, ifile, bb)
                bb = self.set_attacks(irank - 1, ifile - 1, bb)
                bb = self.set_attacks(irank - 1, ifile + 1, bb)
                bb = self.set_attacks(irank - 1, ifile, bb)
                bb = self.set_attacks(irank, ifile - 1, bb)
                bb = self.set_attacks(irank, ifile + 1, bb)
                self.abb_gold_attacks[0][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 1, ifile - 1, bb)
                bb = self.set_attacks(irank + 1, ifile + 1, bb)
                bb = self.set_attacks(irank + 1, ifile, bb)
                bb = self.set_attacks(irank - 1, ifile, bb)
                bb = self.set_attacks(irank, ifile - 1, bb)
                bb = self.set_attacks(irank, ifile + 1, bb)
                self.abb_gold_attacks[1][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 1, ifile - 1, bb)
                bb = self.set_attacks(irank + 1, ifile + 1, bb)
                bb = self.set_attacks(irank - 1, ifile, bb)
                bb = self.set_attacks(irank - 1, ifile - 1, bb)
                bb = self.set_attacks(irank - 1, ifile + 1, bb)
                self.abb_silver_attacks[0][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 1, ifile - 1, bb)
                bb = self.set_attacks(irank + 1, ifile + 1, bb)
                bb = self.set_attacks(irank + 1, ifile, bb)
                bb = self.set_attacks(irank - 1, ifile - 1, bb)
                bb = self.set_attacks(irank - 1, ifile + 1, bb)
                self.abb_silver_attacks[1][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank - 2, ifile - 1, bb)
                bb = self.set_attacks(irank - 2, ifile + 1, bb)
                self.abb_knight_attacks[0][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 2, ifile - 1, bb)
                bb = self.set_attacks(irank + 2, ifile + 1, bb)
                self.abb_knight_attacks[1][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank, ifile - 1, bb)
                bb = self.set_attacks(irank, ifile + 1, bb)
                self.abb_stomach_attacks[irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank - 2, ifile - 1, bb)
                bb = self.set_attacks(irank - 2, ifile, bb)
                bb = self.set_attacks(irank - 2, ifile + 1, bb)
                self.abb_2up3sq_attacks[0][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 2, ifile - 1, bb)
                bb = self.set_attacks(irank + 2, ifile, bb)
                bb = self.set_attacks(irank + 2, ifile + 1, bb)
                self.abb_2up3sq_attacks[1][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank - 2, ifile, bb)
                self.abb_2up_attacks[0][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 2, ifile, bb)
                self.abb_2up_attacks[1][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank - 3, ifile, bb)
                self.abb_3up_attacks[0][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 3, ifile, bb)
                self.abb_3up_attacks[1][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank + 1, ifile - 1, bb)
                bb = self.set_attacks(irank + 1, ifile + 1, bb)
                self.abb_diag_back_attacks[0][irank * self.cls_board.nfile + ifile] = bb

                bb = self.cls_bitop.bb_ini()
                bb = self.set_attacks(irank - 1, ifile - 1, bb)
                bb = self.set_attacks(irank - 1, ifile + 1, bb)
                self.abb_diag_back_attacks[1][irank * self.cls_board.nfile + ifile] = bb

    def init_diag1_mask_table(self):
        abb_plus8dir = [0] * self.cls_board.square_nb
        abb_minus8dir = [0] * self.cls_board.square_nb

        bb2 = self.cls_bitop.bb_ini()
        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                if irank != self.cls_rank.rank1 and irank != self.cls_rank.rank9 and ifile != self.cls_file.file1 and ifile != self.cls_file.file9:
                    bb2 = self.cls_bitop.xor(isquare, bb2)

        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                abb_plus8dir[isquare] = self.cls_bitop.bb_ini()
                abb_minus8dir[isquare] = self.cls_bitop.bb_ini()
                for i in range(1, self.cls_board.nfile):
                    abb_plus8dir[isquare] = self.set_attacks(irank + i, ifile - i, abb_plus8dir[isquare])
                    abb_minus8dir[isquare] = self.set_attacks(irank - i, ifile + i, abb_minus8dir[isquare])

        for isquare in range(self.cls_board.square_nb):
            bb = self.cls_bitop.bb_ini()
            bb = self.cls_bitop.bb_or(abb_plus8dir[isquare], abb_minus8dir[isquare])
            bb = self.cls_bitop.bb_and(bb, bb2)
            self.abb_diag1_mask_ex[isquare] = bb

    def init_diag2_mask_table(self):
        abb_plus10dir = [0] * self.cls_board.square_nb
        abb_minus10dir = [0] * self.cls_board.square_nb

        bb2 = self.cls_bitop.bb_ini()
        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                if irank != self.cls_rank.rank1 and irank != self.cls_rank.rank9 and ifile != self.cls_file.file1 and ifile != self.cls_file.file9:
                    bb2 = self.cls_bitop.xor(isquare, bb2)

        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                abb_plus10dir[isquare] = self.cls_bitop.bb_ini()
                abb_minus10dir[isquare] = self.cls_bitop.bb_ini()
                for i in range(1, self.cls_board.nfile):
                    abb_plus10dir[isquare] = self.set_attacks(irank + i, ifile + i, abb_plus10dir[isquare])
                    abb_minus10dir[isquare] = self.set_attacks(irank - i, ifile - i, abb_minus10dir[isquare])

        for isquare in range(self.cls_board.square_nb):
            bb = self.cls_bitop.bb_ini()
            bb = self.cls_bitop.bb_or(abb_plus10dir[isquare], abb_minus10dir[isquare])
            bb = self.cls_bitop.bb_and(bb, bb2)
            self.abb_diag2_mask_ex[isquare] = bb

    def init_rank_mask_table(self):
        abb_plus1dir = [0] * self.cls_board.square_nb
        abb_minus1dir = [0] * self.cls_board.square_nb

        bb2 = self.cls_bitop.bb_ini()
        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                if ifile != self.cls_file.file1 and ifile != self.cls_file.file9:
                    bb2 = self.cls_bitop.xor(isquare, bb2)

        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                abb_plus1dir[isquare] = self.cls_bitop.bb_ini()
                abb_minus1dir[isquare] = self.cls_bitop.bb_ini()
                for i in range(1, self.cls_board.nfile):
                    abb_plus1dir[isquare] = self.set_attacks(irank, ifile + i, abb_plus1dir[isquare])
                    abb_minus1dir[isquare] = self.set_attacks(irank, ifile - i, abb_minus1dir[isquare])

        for isquare in range(self.cls_board.square_nb):
            bb = self.cls_bitop.bb_ini()
            bb = self.cls_bitop.bb_or(abb_plus1dir[isquare], abb_minus1dir[isquare])
            bb = self.cls_bitop.bb_and(bb, bb2)
            self.abb_rank_mask_ex[isquare] = bb

    def init_file_mask_table(self):
        abb_plus9dir = [0] * self.cls_board.square_nb
        abb_minus9dir = [0] * self.cls_board.square_nb

        bb2 = self.cls_bitop.bb_ini()
        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                if irank != self.cls_rank.rank1 and irank != self.cls_rank.rank9:
                    bb2 = self.cls_bitop.xor(isquare, bb2)

        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                abb_plus9dir[isquare] = self.cls_bitop.bb_ini()
                abb_minus9dir[isquare] = self.cls_bitop.bb_ini()
                for i in range(1, self.cls_board.nfile):
                    abb_plus9dir[isquare] = self.set_attacks(irank + i, ifile, abb_plus9dir[isquare])
                    abb_minus9dir[isquare] = self.set_attacks(irank - i, ifile, abb_minus9dir[isquare])

        for isquare in range(self.cls_board.square_nb):
            bb = self.cls_bitop.bb_ini()
            bb = self.cls_bitop.bb_or(abb_plus9dir[isquare], abb_minus9dir[isquare])
            bb = self.cls_bitop.bb_and(bb, bb2)
            self.abb_file_mask_ex[isquare] = bb

    def init_diag1_attack_tables(self):
        abb_effect = []

        for i in range(self.cls_board.square_nb):
            bb = self.cls_bitop.bb_ini()
            abb_effect.append(bb)
        
        for pos in range(self.cls_board.square_nb):
            if pos > 8:
                sq = pos - 8
                if self.cls_board.rank_table[pos] != self.cls_board.rank_table[sq]:
                    while sq >= 0:
                        abb_effect[pos] = self.cls_bitop.xor(sq, abb_effect[pos])
                        if self.cls_board.file_table[sq] == self.cls_file.file9:
                            break
                        sq -= 8
            if pos < 72:
                sq = pos + 8
                if self.cls_board.rank_table[pos] != self.cls_board.rank_table[sq]:
                    while sq <= 80:
                        abb_effect[pos] = self.cls_bitop.xor(sq, abb_effect[pos])
                        if self.cls_board.file_table[sq] == self.cls_file.file1:
                            break
                        sq += 8
        
        for pos in range(self.cls_board.square_nb):
            if pos == 80:
                break
            array_mask = [self.cls_board.square_nb] * 7
            mask = self.abb_diag1_mask_ex[pos]
            i = 0
            length = 0
            while mask != 0:
                sq = self.cls_bitop.last_one(mask)
                array_mask[i] = sq
                mask = self.cls_bitop.xor(sq, mask)
                length += 1
                i += 1
            array_effect = [self.cls_board.square_nb] * 8
            bb_effect = abb_effect[pos]
            i = 0
            while bb_effect != 0:
                sq = self.cls_bitop.last_one(bb_effect)
                array_effect[i] = sq
                bb_effect = self.cls_bitop.xor(sq, bb_effect)
                i += 1
            for hash in range(128):
                bb_hash = self.cls_bitop.bb_ini()
                bb_hash = hash
                bb_temp_mask = self.cls_bitop.bb_ini()
                bb_temp_effect = self.cls_bitop.bb_ini()
                if hash == 0 or hash == 1:
                    hash_length = 1
                elif hash == 2 or hash == 3:
                    hash_length = 2
                elif hash >= 4 and hash < 8:
                    hash_length = 3
                elif hash >= 8 and hash < 16:
                    hash_length = 4
                elif hash >= 16 and hash < 32:
                    hash_length = 5
                elif hash >= 32 and hash < 64:
                    hash_length = 6
                elif hash >= 64 and hash < 128:
                    hash_length = 7
                while bb_hash != 0:
                    sq = self.cls_bitop.last_one(bb_hash)
                    if array_mask[80 - sq] != self.cls_board.square_nb:
                        bb_temp_mask = self.cls_bitop.bb_or(bb_temp_mask, self.cls_bitop.bb_mask[array_mask[80 - sq]])
                    bb_hash = self.cls_bitop.xor(sq, bb_hash)

                if self.abb_diag1_mask_ex[pos] == 0 and hash != 0:
                    break
                x = hash & self.cls_bitop.bb_mask[80 - length]
                if x != 0 and hash_length > 0:
                    break
                
                start = pos + 8
                end1 = 0
                bb_temp_effect = self.cls_bitop.bb_ini()
                bb = self.cls_bitop.bb_ini()
                if array_effect[0] != self.cls_board.square_nb:
                    end1 = array_effect[0]
                for i in range(start, end1 + 1, 8):
                    bb = self.cls_bitop.bb_and(bb_temp_mask, self.cls_bitop.bb_mask[i])
                    bb_temp_effect = self.cls_bitop.bb_or(bb_temp_effect, self.cls_bitop.bb_mask[i])
                    if bb != 0:
                        break
                end2 = 0
                for i in range(7, -1, -1):
                    if array_effect[i] != self.cls_board.square_nb and end2 < start:
                        end2 = array_effect[i]
                        break
                start = pos - 8
                for i in range(start, end2 - 1, -8):
                    bb = self.cls_bitop.bb_and(bb_temp_mask, self.cls_bitop.bb_mask[i])
                    bb_temp_effect = self.cls_bitop.bb_or(bb_temp_effect, self.cls_bitop.bb_mask[i])
                    if bb != 0:
                        break
                self.abb_diag1_attacks[pos][hash] = bb_temp_effect

    def init_diag2_attack_tables(self):
        abb_effect = []

        for i in range(self.cls_board.square_nb):
            bb = self.cls_bitop.bb_ini()
            abb_effect.append(bb)
        
        for pos in range(self.cls_board.square_nb):
            if pos > 9:
                sq = pos - 10
                x = self.cls_board.rank_table[pos] - self.cls_board.rank_table[sq]
                if abs(x) != 2:
                    while sq >= 0:
                        abb_effect[pos] = self.cls_bitop.xor(sq, abb_effect[pos])
                        if self.cls_board.file_table[sq] == self.cls_file.file1:
                            break
                        sq -= 10
            if pos < 71:
                sq = pos + 10
                x = self.cls_board.rank_table[pos] - self.cls_board.rank_table[sq]
                if abs(x) != 2:
                    while sq <= 80:
                        abb_effect[pos] = self.cls_bitop.xor(sq, abb_effect[pos])
                        if self.cls_board.file_table[sq] == self.cls_file.file9:
                            break
                        sq += 10
        
        for pos in range(self.cls_board.square_nb):
            if pos == 72:
                continue
            array_mask = [self.cls_board.square_nb] * 7
            mask = self.abb_diag2_mask_ex[pos]
            i = 0
            length = 0
            while mask != 0:
                sq = self.cls_bitop.last_one(mask)
                array_mask[i] = sq
                mask = self.cls_bitop.xor(sq, mask)
                length += 1
                i += 1
            array_effect = [self.cls_board.square_nb] * 8
            bb_effect = abb_effect[pos]
            i = 0
            while bb_effect != 0:
                sq = self.cls_bitop.last_one(bb_effect)
                array_effect[i] = sq
                bb_effect = self.cls_bitop.xor(sq, bb_effect)
                i += 1
            for hash in range(128):
                bb_hash = self.cls_bitop.bb_ini()
                bb_hash = hash
                bb_temp_mask = self.cls_bitop.bb_ini()
                bb_temp_effect = self.cls_bitop.bb_ini()
                if hash == 0 or hash == 1:
                    hash_length = 1
                elif hash == 2 or hash == 3:
                    hash_length = 2
                elif hash >= 4 and hash < 8:
                    hash_length = 3
                elif hash >= 8 and hash < 16:
                    hash_length = 4
                elif hash >= 16 and hash < 32:
                    hash_length = 5
                elif hash >= 32 and hash < 64:
                    hash_length = 6
                elif hash >= 64 and hash < 128:
                    hash_length = 7
                while bb_hash != 0:
                    sq = self.cls_bitop.last_one(bb_hash)
                    if array_mask[80 - sq] != self.cls_board.square_nb:
                        bb_temp_mask = self.cls_bitop.bb_or(bb_temp_mask, self.cls_bitop.bb_mask[array_mask[80 - sq]])
                    bb_hash = self.cls_bitop.xor(sq, bb_hash)

                if self.abb_diag2_mask_ex[pos] == 0 and hash != 0:
                    break
                x = hash & self.cls_bitop.bb_mask[80 - length]
                if x != 0 and hash_length > 0:
                    break
                
                start = pos + 10
                end1 = 0
                bb_temp_effect = self.cls_bitop.bb_ini()
                bb = self.cls_bitop.bb_ini()
                if array_effect[0] != self.cls_board.square_nb:
                    end1 = array_effect[0]
                for i in range(start, end1 + 1, 10):
                    bb = self.cls_bitop.bb_and(bb_temp_mask, self.cls_bitop.bb_mask[i])
                    bb_temp_effect = self.cls_bitop.bb_or(bb_temp_effect, self.cls_bitop.bb_mask[i])
                    if bb != 0:
                        break
                end2 = 0
                for i in range(7, -1, -1):
                    if array_effect[i] != self.cls_board.square_nb and end2 < start:
                        end2 = array_effect[i]
                        break
                start = pos - 10
                for i in range(start, end2 - 1, -10):
                    bb = self.cls_bitop.bb_and(bb_temp_mask, self.cls_bitop.bb_mask[i])
                    bb_temp_effect = self.cls_bitop.bb_or(bb_temp_effect, self.cls_bitop.bb_mask[i])
                    if bb != 0:
                        break
                self.abb_diag2_attacks[pos][hash] = bb_temp_effect

    def init_rank_attack_tables(self):
        abb_effect = []

        for i in range(self.cls_board.square_nb):
            bb = self.cls_bitop.bb_ini()
            abb_effect.append(bb)
        
        for pos in range(self.cls_board.square_nb):
            if self.cls_board.file_table[pos] != self.cls_file.file1:
                sq = pos - 1
                while sq >= 0:
                    abb_effect[pos] = self.cls_bitop.xor(sq, abb_effect[pos])
                    if self.cls_board.file_table[sq] == self.cls_file.file1:
                        break
                    sq -= 1
            if self.cls_board.file_table[pos] != self.cls_file.file9:
                sq = pos + 1
                while sq <= 80:
                    abb_effect[pos] = self.cls_bitop.xor(sq, abb_effect[pos])
                    if self.cls_board.file_table[sq] == self.cls_file.file9:
                        break
                    sq += 1
        
        for pos in range(self.cls_board.square_nb):
            array_mask = [self.cls_board.square_nb] * 7
            mask = self.abb_rank_mask_ex[pos]
            i = 0
            length = 0
            while mask != 0:
                sq = self.cls_bitop.last_one(mask)
                array_mask[i] = sq
                mask = self.cls_bitop.xor(sq, mask)
                length += 1
                i += 1
            array_effect = [self.cls_board.square_nb] * 8
            bb_effect = abb_effect[pos]
            i = 0
            while bb_effect != 0:
                sq = self.cls_bitop.last_one(bb_effect)
                array_effect[i] = sq
                bb_effect = self.cls_bitop.xor(sq, bb_effect)
                i += 1
            for hash in range(128):
                bb_hash = self.cls_bitop.bb_ini()
                bb_hash = hash
                bb_temp_mask = self.cls_bitop.bb_ini()
                bb_temp_effect = self.cls_bitop.bb_ini()
                if hash == 0 or hash == 1:
                    hash_length = 1
                elif hash == 2 or hash == 3:
                    hash_length = 2
                elif hash >= 4 and hash < 8:
                    hash_length = 3
                elif hash >= 8 and hash < 16:
                    hash_length = 4
                elif hash >= 16 and hash < 32:
                    hash_length = 5
                elif hash >= 32 and hash < 64:
                    hash_length = 6
                elif hash >= 64 and hash < 128:
                    hash_length = 7
                while bb_hash != 0:
                    sq = self.cls_bitop.last_one(bb_hash)
                    if array_mask[80 - sq] != self.cls_board.square_nb:
                        bb_temp_mask = self.cls_bitop.bb_or(bb_temp_mask, self.cls_bitop.bb_mask[array_mask[80 - sq]])
                    bb_hash = self.cls_bitop.xor(sq, bb_hash)

                x = hash & self.cls_bitop.bb_mask[80 - length]
                if x != 0 and hash_length > 0:
                    break
                
                start = pos + 1
                end1 = 0
                bb_temp_effect = self.cls_bitop.bb_ini()
                bb = self.cls_bitop.bb_ini()
                if array_effect[0] != self.cls_board.square_nb:
                    end1 = array_effect[0]
                for i in range(start, end1 + 1):
                    bb = self.cls_bitop.bb_and(bb_temp_mask, self.cls_bitop.bb_mask[i])
                    bb_temp_effect = self.cls_bitop.bb_or(bb_temp_effect, self.cls_bitop.bb_mask[i])
                    if bb != 0:
                        break
                end2 = 0
                for i in range(7, -1, -1):
                    if array_effect[i] != self.cls_board.square_nb and end2 < start:
                        end2 = array_effect[i]
                        break
                start = pos - 1
                for i in range(start, end2 - 1, -1):
                    bb = self.cls_bitop.bb_and(bb_temp_mask, self.cls_bitop.bb_mask[i])
                    bb_temp_effect = self.cls_bitop.bb_or(bb_temp_effect, self.cls_bitop.bb_mask[i])
                    if bb != 0:
                        break
                self.abb_rank_attacks[pos][hash] = bb_temp_effect

    def init_file_attack_tables(self):
        abb_effect = []

        for i in range(self.cls_board.square_nb):
            bb = self.cls_bitop.bb_ini()
            abb_effect.append(bb)
        
        for pos in range(self.cls_board.square_nb):
            if self.cls_board.rank_table[pos] != self.cls_rank.rank1:
                sq = pos - 9
                while sq >= 0:
                    abb_effect[pos] = self.cls_bitop.xor(sq, abb_effect[pos])
                    if self.cls_board.rank_table[sq] == self.cls_rank.rank1:
                        break
                    sq -= 9
            if self.cls_board.rank_table[pos] != self.cls_rank.rank9:
                sq = pos + 9
                while sq <= 80:
                    abb_effect[pos] = self.cls_bitop.xor(sq, abb_effect[pos])
                    if self.cls_board.rank_table[sq] == self.cls_rank.rank9:
                        break
                    sq += 9
        
        for pos in range(self.cls_board.square_nb):
            array_mask = [self.cls_board.square_nb] * 7
            mask = self.abb_file_mask_ex[pos]
            i = 0
            length = 0
            while mask != 0:
                sq = self.cls_bitop.last_one(mask)
                array_mask[i] = sq
                mask = self.cls_bitop.xor(sq, mask)
                length += 1
                i += 1
            array_effect = [self.cls_board.square_nb] * 8
            bb_effect = abb_effect[pos]
            i = 0
            while bb_effect != 0:
                sq = self.cls_bitop.last_one(bb_effect)
                array_effect[i] = sq
                bb_effect = self.cls_bitop.xor(sq, bb_effect)
                i += 1
            for hash in range(128):
                bb_hash = self.cls_bitop.bb_ini()
                bb_hash = hash
                bb_temp_mask = self.cls_bitop.bb_ini()
                bb_temp_effect = self.cls_bitop.bb_ini()
                if hash == 0 or hash == 1:
                    hash_length = 1
                elif hash == 2 or hash == 3:
                    hash_length = 2
                elif hash >= 4 and hash < 8:
                    hash_length = 3
                elif hash >= 8 and hash < 16:
                    hash_length = 4
                elif hash >= 16 and hash < 32:
                    hash_length = 5
                elif hash >= 32 and hash < 64:
                    hash_length = 6
                elif hash >= 64 and hash < 128:
                    hash_length = 7
                while bb_hash != 0:
                    sq = self.cls_bitop.last_one(bb_hash)
                    if array_mask[80 - sq] != self.cls_board.square_nb:
                        bb_temp_mask = self.cls_bitop.bb_or(bb_temp_mask, self.cls_bitop.bb_mask[array_mask[80 - sq]])
                    bb_hash = self.cls_bitop.xor(sq, bb_hash)

                x = hash & self.cls_bitop.bb_mask[80 - length]
                if x != 0 and hash_length > 0:
                    break
                
                start = pos + 9
                end1 = 0
                bb_temp_effect = self.cls_bitop.bb_ini()
                bb = self.cls_bitop.bb_ini()
                if array_effect[0] != self.cls_board.square_nb:
                    end1 = array_effect[0]
                for i in range(start, end1 + 1, 9):
                    bb = self.cls_bitop.bb_and(bb_temp_mask, self.cls_bitop.bb_mask[i])
                    bb_temp_effect = self.cls_bitop.bb_or(bb_temp_effect, self.cls_bitop.bb_mask[i])
                    if bb != 0:
                        break
                end2 = 0
                for i in range(7, -1, -1):
                    if array_effect[i] != self.cls_board.square_nb and end2 < start:
                        end2 = array_effect[i]
                        break
                start = pos - 9
                for i in range(start, end2 - 1, -9):
                    bb = self.cls_bitop.bb_and(bb_temp_mask, self.cls_bitop.bb_mask[i])
                    bb_temp_effect = self.cls_bitop.bb_or(bb_temp_effect, self.cls_bitop.bb_mask[i])
                    if bb != 0:
                        break
                self.abb_file_attacks[pos][hash] = bb_temp_effect

    def init_other_attack_tables(self):
        abb_plus1dir = [0] * self.cls_board.square_nb
        abb_plus8dir = [0] * self.cls_board.square_nb
        abb_plus9dir = [0] * self.cls_board.square_nb
        abb_plus10dir = [0] * self.cls_board.square_nb
        abb_minus1dir = [0] * self.cls_board.square_nb
        abb_minus8dir = [0] * self.cls_board.square_nb
        abb_minus9dir = [0] * self.cls_board.square_nb
        abb_minus10dir = [0] * self.cls_board.square_nb

        for irank in range(self.cls_board.nrank):
            for ifile in range(self.cls_board.nfile):
                isquare = irank * self.cls_board.nfile + ifile
                abb_plus1dir[isquare] = self.cls_bitop.bb_ini()
                abb_plus8dir[isquare] = self.cls_bitop.bb_ini()
                abb_plus9dir[isquare] = self.cls_bitop.bb_ini()
                abb_plus10dir[isquare] = self.cls_bitop.bb_ini()
                abb_minus1dir[isquare] = self.cls_bitop.bb_ini()
                abb_minus8dir[isquare] = self.cls_bitop.bb_ini()
                abb_minus9dir[isquare] = self.cls_bitop.bb_ini()
                abb_minus10dir[isquare] = self.cls_bitop.bb_ini()
                for i in range(1, self.cls_board.nfile):
                    abb_plus1dir[isquare] = self.set_attacks(irank, ifile + i, abb_plus1dir[isquare])
                    abb_plus8dir[isquare] = self.set_attacks(irank + i, ifile - i, abb_plus8dir[isquare])
                    abb_plus9dir[isquare] = self.set_attacks(irank + i, ifile, abb_plus9dir[isquare])
                    abb_plus10dir[isquare] = self.set_attacks(irank + i, ifile + i, abb_plus10dir[isquare])
                    abb_minus1dir[isquare] = self.set_attacks(irank, ifile - i, abb_minus1dir[isquare])
                    abb_minus8dir[isquare] = self.set_attacks(irank - i, ifile + i, abb_minus8dir[isquare])
                    abb_minus9dir[isquare] = self.set_attacks(irank - i, ifile, abb_minus9dir[isquare])
                    abb_minus10dir[isquare] = self.set_attacks(irank - i, ifile - i, abb_minus10dir[isquare])

        for isquare in range(self.cls_board.square_nb):
            self.abb_plus_rays[isquare] = self.cls_bitop.bb_or(abb_plus1dir[isquare], abb_plus8dir[isquare])
            self.abb_plus_rays[isquare] = self.cls_bitop.bb_or(self.abb_plus_rays[isquare], abb_plus9dir[isquare])
            self.abb_plus_rays[isquare] = self.cls_bitop.bb_or(self.abb_plus_rays[isquare], abb_plus10dir[isquare])
            self.abb_minus_rays[isquare] = self.cls_bitop.bb_or(abb_minus1dir[isquare], abb_minus8dir[isquare])
            self.abb_minus_rays[isquare] = self.cls_bitop.bb_or(self.abb_minus_rays[isquare], abb_minus9dir[isquare])
            self.abb_minus_rays[isquare] = self.cls_bitop.bb_or(self.abb_minus_rays[isquare], abb_minus10dir[isquare])

        for ifrom in range(self.cls_board.square_nb):
            for ito in range(self.cls_board.square_nb):
                self.adirec[ifrom][ito] = self.cls_direc.direc_misc
            bb = self.cls_bitop.bb_or(abb_plus1dir[ifrom], abb_minus1dir[ifrom])
            while bb != 0:
                ito = self.cls_bitop.first_one(bb)
                self.adirec[ifrom][ito] = self.cls_direc.direc_rank
                bb = self.cls_bitop.xor(ito, bb)
            bb = self.cls_bitop.bb_or(abb_plus8dir[ifrom], abb_minus8dir[ifrom])
            while bb != 0:
                ito = self.cls_bitop.first_one(bb)
                self.adirec[ifrom][ito] = self.cls_direc.direc_diag1
                bb = self.cls_bitop.xor(ito, bb)
            bb = self.cls_bitop.bb_or(abb_plus9dir[ifrom], abb_minus9dir[ifrom])
            while bb != 0:
                ito = self.cls_bitop.first_one(bb)
                self.adirec[ifrom][ito] = self.cls_direc.direc_file
                bb = self.cls_bitop.xor(ito, bb)
            bb = self.cls_bitop.bb_or(abb_plus10dir[ifrom], abb_minus10dir[ifrom])
            while bb != 0:
                ito = self.cls_bitop.first_one(bb)
                self.adirec[ifrom][ito] = self.cls_direc.direc_diag2
                bb = self.cls_bitop.xor(ito, bb)

        for ifrom in range(self.cls_board.square_nb):
            for ito in range(self.cls_board.square_nb):
                self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_ini()
                if (ifrom - ito) > 0:
                    if self.adirec[ifrom][ito] == self.cls_direc.direc_rank:
                        self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_xor(abb_minus1dir[ito + 1], abb_minus1dir[ifrom])
                    elif self.adirec[ifrom][ito] == self.cls_direc.direc_file:
                        self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_xor(abb_minus9dir[ito + 9], abb_minus9dir[ifrom])
                    elif self.adirec[ifrom][ito] == self.cls_direc.direc_diag1:
                        self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_xor(abb_minus8dir[ito + 8], abb_minus8dir[ifrom])
                    elif self.adirec[ifrom][ito] == self.cls_direc.direc_diag2:
                        self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_xor(abb_minus10dir[ito + 10], abb_minus10dir[ifrom])
                else:
                    if self.adirec[ifrom][ito] == self.cls_direc.direc_rank:
                        self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_xor(abb_plus1dir[ito - 1], abb_plus1dir[ifrom])
                    elif self.adirec[ifrom][ito] == self.cls_direc.direc_file:
                        self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_xor(abb_plus9dir[ito - 9], abb_plus9dir[ifrom])
                    elif self.adirec[ifrom][ito] == self.cls_direc.direc_diag1:
                        self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_xor(abb_plus8dir[ito - 8], abb_plus8dir[ifrom])
                    elif self.adirec[ifrom][ito] == self.cls_direc.direc_diag2:
                        self.abb_obstacle[ifrom][ito] = self.cls_bitop.bb_xor(abb_plus10dir[ito - 10], abb_plus10dir[ifrom])

    def set_attacks(self, irank, ifile, bb):
        if irank >= self.cls_rank.rank1 and irank <= self.cls_rank.rank9 and ifile >= self.cls_file.file1 and ifile <= self.cls_file.file9:
            bb = self.cls_bitop.xor(irank * self.cls_board.nfile + ifile, bb)
        return bb

    def get_diag1_attacks(self, bb_occupied, sq):
        hash = self.cls_bitop.pext(bb_occupied, self.abb_diag1_mask_ex[sq])
        return self.abb_diag1_attacks[sq][hash]

    def get_diag2_attacks(self, bb_occupied, sq):
        hash = self.cls_bitop.pext(bb_occupied, self.abb_diag2_mask_ex[sq])
        return self.abb_diag2_attacks[sq][hash]

    def get_rank_attacks(self, bb_occupied, sq):
        hash = self.cls_bitop.pext(bb_occupied, self.abb_rank_mask_ex[sq])
        return self.abb_rank_attacks[sq][hash]

    def get_file_attacks(self, bb_occupied, sq):
        hash = self.cls_bitop.pext(bb_occupied, self.abb_file_mask_ex[sq])
        return self.abb_file_attacks[sq][hash]

    def get_bishop_attacks(self, bb_occupied, sq):
        bb0 = self.get_diag1_attacks(bb_occupied, sq)
        bb1 = self.get_diag2_attacks(bb_occupied, sq)
        return (bb0 | bb1)

    def get_rook_attacks(self, bb_occupied, sq):
        bb0 = self.get_rank_attacks(bb_occupied, sq)
        bb1 = self.get_file_attacks(bb_occupied, sq)
        return (bb0 | bb1)

    #sqの位置に当たりをかけている先手の駒を取得する
    def b_attacks_to_piece(self, board, sq):
        bb_ret = self.cls_bitop.bb_ini()
        if sq < (self.cls_rank.rank9 * self.cls_board.nfile) and self.cls_board.board[sq + self.cls_board.nfile] == self.cls_piece.pawn:
            bb_ret = self.cls_bitop.bb_mask[sq + self.cls_board.nfile]
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_knight[0], self.abb_knight_attacks[1][sq])
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_silver[0], self.abb_silver_attacks[1][sq])
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_total_gold[0], self.abb_gold_attacks[1][sq])
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_hdk[0], self.abb_king_attacks[sq])
        bb_attacks = self.get_bishop_attacks(board.bb_occupied, sq)
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_bh[0], bb_attacks)
        bb = board.bb_rd[0]
        bb_attacks = self.get_rank_attacks(board.bb_occupied, sq)
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, bb, bb_attacks)
        bb = self.cls_bitop.bb_and_or(bb, board.bb_lance[0], self.abb_plus_rays[sq])
        bb_attacks = self.get_file_attacks(board.bb_occupied, sq)
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, bb, bb_attacks)
        return bb_ret

    #sqの位置に当たりをかけている後手の駒を取得する
    def w_attacks_to_piece(self, board, sq):
        bb_ret = self.cls_bitop.bb_ini()
        if sq >= self.cls_board.nfile and self.cls_board.board[sq - self.cls_board.nfile] == -self.cls_piece.pawn:
            bb_ret = self.cls_bitop.bb_mask[sq - self.cls_board.nfile]
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_knight[1], self.abb_knight_attacks[0][sq])
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_silver[1], self.abb_silver_attacks[0][sq])
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_total_gold[1], self.abb_gold_attacks[0][sq])
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_hdk[1], self.abb_king_attacks[sq])
        bb_attacks = self.get_bishop_attacks(board.bb_occupied, sq)
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, board.bb_bh[1], bb_attacks)
        bb = board.bb_rd[1]
        bb_attacks = self.get_rank_attacks(board.bb_occupied, sq)
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, bb, bb_attacks)
        bb = self.cls_bitop.bb_and_or(bb, board.bb_lance[1], self.abb_minus_rays[sq])
        bb_attacks = self.get_file_attacks(board.bb_occupied, sq)
        bb_ret = self.cls_bitop.bb_and_or(bb_ret, bb, bb_attacks)
        return bb_ret

    #sqの位置に相手からの当たりがかかっているか？
    def is_attacked(self, board, sq, color):
        if color == 0:
            bb_rays = self.abb_plus_rays[sq]
        else:
            bb_rays = self.abb_minus_rays[sq]
        bb = self.cls_bitop.bb_ini()
        bb = self.cls_bitop.bb_and(board.bb_pawn_attacks[color], self.cls_bitop.bb_mask[sq])
        bb = self.cls_bitop.bb_and_or(bb, board.bb_knight[color], self.abb_knight_attacks[color ^ 1][sq])
        bb = self.cls_bitop.bb_and_or(bb, board.bb_silver[color], self.abb_silver_attacks[color ^ 1][sq])
        bb = self.cls_bitop.bb_and_or(bb, board.bb_total_gold[color], self.abb_gold_attacks[color ^ 1][sq])
        bb = self.cls_bitop.bb_and_or(bb, board.bb_hdk[color], self.abb_king_attacks[sq])
        bb_atk = self.get_bishop_attacks(board.bb_occupied, sq)
        bb = self.cls_bitop.bb_and_or(bb, board.bb_bh[color], bb_atk)
        bb1 = board.bb_rd[color]
        bb1 = self.cls_bitop.bb_and_or(bb1, board.bb_lance[color], bb_rays)
        bb_atk = self.get_file_attacks(board.bb_occupied, sq)
        bb = self.cls_bitop.bb_and_or(bb, bb1, bb_atk)
        bb_atk = self.get_rank_attacks(board.bb_occupied, sq)
        bb = self.cls_bitop.bb_and_or(bb, board.bb_rd[color], bb_atk)
        return bb

    #玉について、isquareの位置の駒がPinされているかどうかのチェック
    def is_pinned_on_king(self, board, isquare, idirec, color):
        bb_king = self.cls_bitop.bb_mask[board.sq_king[color]]
        if idirec == self.cls_direc.direc_rank:
            bb_attacks = self.get_rank_attacks(board.bb_occupied, isquare)
            bb = bb_attacks & bb_king
            if bb != 0:
                bb = bb_attacks & board.bb_rd[color ^ 1]
                return bb
        elif idirec == self.cls_direc.direc_file:
            bb_attacks = self.get_file_attacks(board.bb_occupied, isquare)
            bb = bb_attacks & bb_king
            if bb != 0:
                bb = bb_attacks & board.bb_rd[color ^ 1]
                if bb == 0:
                    bb = bb_attacks & bb_king
                    bb = bb_attacks & board.bb_lance[color ^ 1]
                return bb
        elif idirec == self.cls_direc.direc_diag1:
            bb_attacks = self.get_diag1_attacks(board.bb_occupied, isquare)
            bb = bb_attacks & bb_king
            if bb != 0:
                bb = bb_attacks & board.bb_bh[color ^ 1]
                return bb
        elif idirec == self.cls_direc.direc_diag2:
            bb_attacks = self.get_diag2_attacks(board.bb_occupied, isquare)
            bb = bb_attacks & bb_king
            if bb != 0:
                bb = bb_attacks & board.bb_bh[color ^ 1]
                return bb
        return 0

    def is_discover_king(self, board, ifrom, ito, color):
        idirec = self.adirec[board.sq_king[color]][ifrom]
        if idirec != 0 and idirec != self.adirec[board.sq_king[color]][ito] and self.is_pinned_on_king(board, ifrom, idirec, color) != 0:
            return 1
        else:
            return 0

    #打ち歩詰めチェック
    def is_mate_pawn_drop(self, board, sq_drop, color):
        if color == 0:
            if board.board[sq_drop - 9] != -self.cls_piece.king:
                return 1
        else:
            if board.board[sq_drop + 9] != self.cls_piece.king:
                return 1
        bb_sum = self.cls_bitop.bb_and(board.bb_knight[color], self.abb_knight_attacks[color ^ 1][sq_drop])
        bb_sum = self.cls_bitop.bb_and_or(bb_sum, board.bb_silver[color], self.abb_silver_attacks[color ^ 1][sq_drop])
        bb_sum = self.cls_bitop.bb_and_or(bb_sum, board.bb_total_gold[color], self.abb_gold_attacks[color ^ 1][sq_drop])
        bb = self.get_bishop_attacks(board.bb_occupied, sq_drop)
        bb_sum = self.cls_bitop.bb_and_or(bb_sum, board.bb_bh[color], bb)
        bb = self.get_rook_attacks(board.bb_occupied, sq_drop)
        bb_sum = self.cls_bitop.bb_and_or(bb_sum, board.bb_rd[color], bb)
        bb = self.cls_bitop.bb_or(board.bb_horse[color], board.bb_dragon[color])
        bb_sum = self.cls_bitop.bb_and_or(bb_sum, bb, self.abb_king_attacks[sq_drop])
        while bb_sum != 0:
            ifrom = self.cls_bitop.first_one(bb_sum)
            bb_sum = self.cls_bitop.xor(ifrom, bb_sum)
            if self.is_discover_king(board, ifrom, sq_drop, color) != 0:
                continue
            return 0
        iking = board.sq_king[color]
        iret = 1
        board.bb_occ_color[color ^ 1] = self.cls_bitop.xor(sq_drop, board.bb_occ_color[color ^ 1])
        bb_move = self.cls_bitop.bb_not_and(self.abb_king_attacks[iking], board.bb_occ_color[color])
        while bb_move != 0:
            ito = self.cls_bitop.first_one(bb_move)
            if self.is_attacked(board, ito, color) == 0:
                iret = 0
                break
            bb_move = self.cls_bitop.xor(ito, bb_move)
        board.bb_occ_color[color ^ 1] = self.cls_bitop.xor(sq_drop, board.bb_occ_color[color ^ 1])
        return iret