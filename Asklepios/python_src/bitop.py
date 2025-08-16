class bitop:
    def __init__(self, b):
        self.bb_mask = [0] * b.square_nb
        self.bb_full = self.bb_ini()
        for i in range(b.square_nb):
            self.bb_mask[i] = 1 << (b.square_nb - i - 1)
            self.bb_full = self.bb_or(self.bb_full, self.bb_mask[i])
        self.mask_file1 = (self.bb_mask[0] | self.bb_mask[9] | self.bb_mask[18] | self.bb_mask[27] | self.bb_mask[36] | self.bb_mask[45] | self.bb_mask[54] | self.bb_mask[63] | self.bb_mask[72])
        self.aifirst_one = [0] * 512
        for i in range(512):
            self.aifirst_one[i] = self.first_one00(b.nrank, i)

        self.ailast_one = [0] * 512
        for i in range(512):
            self.ailast_one[i] = self.last_one00(b.nrank, i)

    def bb_or(self, a, b):
        return (a | b)

    def bb_and(self, a, b):
        return (a & b)

    def bb_not_and(self, a, b):
        return (a & ~b)

    def bb_and_or(self, a, b, c):
        return (a | (b & c))

    def bb_xor(self, a, b):
        return (a ^ b)

    def bb_ini(self):
        return 0

    def popu_count(self, bb):
        i = 0
        while (bb):
            i += 1
            bb &= bb - 1

        return i

    def xor(self, sq, bb):
        bb ^= self.bb_mask[sq]
        return bb

    def first_one00(self, nrank, pcs):
        for i in range(nrank):
            if pcs & (1 << (nrank - i - 1)):
                break

        return i

    def last_one00(self, nrank, pcs):
        for i in range(8, -1, -1):
            if pcs & (1 << (nrank - i - 1)):
                break
        return i

    def first_one(self, bb):
        if bb & ((0x1ff) << 72):
            return self.aifirst_one[bb >> 72]
        if bb & ((0x1ff) << 63):
            return self.aifirst_one[bb >> 63] + 9
        if bb & ((0x1ff) << 54):
            return self.aifirst_one[bb >> 54] + 18
        if bb & ((0x1ff) << 45):
            return self.aifirst_one[bb >> 45] + 27
        if bb & ((0x1ff) << 36):
            return self.aifirst_one[bb >> 36] + 36
        if bb & ((0x1ff) << 27):
            return self.aifirst_one[bb >> 27] + 45
        if bb & ((0x1ff) << 18):
            return self.aifirst_one[bb >> 18] + 54
        if bb & ((0x1ff) << 9):
            return self.aifirst_one[bb >> 9] + 63
        return self.aifirst_one[bb] + 72

    def last_one(self, bb):
        if bb & (0x1ff):
            x = bb & (0x1ff)
            return self.ailast_one[x] + 72
        if bb & ((0x1ff) << 9):
            x = bb & ((0x1ff) << 9)
            return self.ailast_one[x >> 9] + 63
        if bb & ((0x1ff) << 18):
            x = bb & ((0x1ff) << 18)
            return self.ailast_one[x >> 18] + 54
        if bb & ((0x1ff) << 27):
            x = bb & ((0x1ff) << 27)
            return self.ailast_one[x >> 27] + 45
        if bb & ((0x1ff) << 36):
            x = bb & ((0x1ff) << 36)
            return self.ailast_one[x >> 36] + 36
        if bb & ((0x1ff) << 45):
            x = bb & ((0x1ff) << 45)
            return self.ailast_one[x >> 45] + 27
        if bb & ((0x1ff) << 54):
            x = bb & ((0x1ff) << 54)
            return self.ailast_one[x >> 54] + 18
        if bb & ((0x1ff) << 63):
            x = bb & ((0x1ff) << 63)
            return self.ailast_one[x >> 63] + 9
        return self.ailast_one[bb >> 72]

    def pext(self, bb_occupied, bb_mask):
        res = 0
        bb = 1
        while bb_mask != 0:
            x = bb_occupied & bb_mask & (~bb_mask + 1)
            if x != 0:
                res |= bb
            bb_mask &= (bb_mask - 1)
            bb += bb
        return res