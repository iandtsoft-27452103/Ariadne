import move

class csa:
    def __init__(self):
        self.conv_table = [0] * 100
        self.conv_table[11] = 8
        self.conv_table[12] = 17
        self.conv_table[13] = 26
        self.conv_table[14] = 35
        self.conv_table[15] = 44
        self.conv_table[16] = 53
        self.conv_table[17] = 62
        self.conv_table[18] = 71
        self.conv_table[19] = 80
        
        self.conv_table[21] = 7
        self.conv_table[22] = 16
        self.conv_table[23] = 25
        self.conv_table[24] = 34
        self.conv_table[25] = 43
        self.conv_table[26] = 52
        self.conv_table[27] = 61
        self.conv_table[28] = 70
        self.conv_table[29] = 79

        self.conv_table[31] = 6
        self.conv_table[32] = 15
        self.conv_table[33] = 24
        self.conv_table[34] = 33
        self.conv_table[35] = 42
        self.conv_table[36] = 51
        self.conv_table[37] = 60
        self.conv_table[38] = 69
        self.conv_table[39] = 78

        self.conv_table[41] = 5
        self.conv_table[42] = 14
        self.conv_table[43] = 23
        self.conv_table[44] = 32
        self.conv_table[45] = 41
        self.conv_table[46] = 50
        self.conv_table[47] = 59
        self.conv_table[48] = 68
        self.conv_table[49] = 77

        self.conv_table[51] = 4
        self.conv_table[52] = 13
        self.conv_table[53] = 22
        self.conv_table[54] = 31
        self.conv_table[55] = 40
        self.conv_table[56] = 49
        self.conv_table[57] = 58
        self.conv_table[58] = 67
        self.conv_table[59] = 76

        self.conv_table[61] = 3
        self.conv_table[62] = 12
        self.conv_table[63] = 21
        self.conv_table[64] = 30
        self.conv_table[65] = 39
        self.conv_table[66] = 48
        self.conv_table[67] = 57
        self.conv_table[68] = 66
        self.conv_table[69] = 75
        
        self.conv_table[71] = 2
        self.conv_table[72] = 11
        self.conv_table[73] = 20
        self.conv_table[74] = 29
        self.conv_table[75] = 38
        self.conv_table[76] = 47
        self.conv_table[77] = 56
        self.conv_table[78] = 65
        self.conv_table[79] = 74
        
        self.conv_table[81] = 1
        self.conv_table[82] = 10
        self.conv_table[83] = 19
        self.conv_table[84] = 28
        self.conv_table[85] = 37
        self.conv_table[86] = 46
        self.conv_table[87] = 55
        self.conv_table[88] = 64
        self.conv_table[89] = 73

        self.conv_table[91] = 0
        self.conv_table[92] = 9
        self.conv_table[93] = 18
        self.conv_table[94] = 27
        self.conv_table[95] = 36
        self.conv_table[96] = 45
        self.conv_table[97] = 54
        self.conv_table[98] = 63
        self.conv_table[99] = 72

        self.rev_table = [0] * 81
        self.rev_table[0] = 91
        self.rev_table[1] = 81
        self.rev_table[2] = 71
        self.rev_table[3] = 61
        self.rev_table[4] = 51
        self.rev_table[5] = 41
        self.rev_table[6] = 31
        self.rev_table[7] = 21
        self.rev_table[8] = 11

        self.rev_table[9] = 92
        self.rev_table[10] = 82
        self.rev_table[11] = 72
        self.rev_table[12] = 62
        self.rev_table[13] = 52
        self.rev_table[14] = 42
        self.rev_table[15] = 32
        self.rev_table[16] = 22
        self.rev_table[17] = 12

        self.rev_table[18] = 93
        self.rev_table[19] = 83
        self.rev_table[20] = 73
        self.rev_table[21] = 63
        self.rev_table[22] = 53
        self.rev_table[23] = 43
        self.rev_table[24] = 33
        self.rev_table[25] = 23
        self.rev_table[26] = 13

        self.rev_table[27] = 94
        self.rev_table[28] = 84
        self.rev_table[29] = 74
        self.rev_table[30] = 64
        self.rev_table[31] = 54
        self.rev_table[32] = 44
        self.rev_table[33] = 34
        self.rev_table[34] = 24
        self.rev_table[35] = 14

        self.rev_table[36] = 95
        self.rev_table[37] = 85
        self.rev_table[38] = 75
        self.rev_table[39] = 65
        self.rev_table[40] = 55
        self.rev_table[41] = 45
        self.rev_table[42] = 35
        self.rev_table[43] = 25
        self.rev_table[44] = 15

        self.rev_table[45] = 96
        self.rev_table[46] = 86
        self.rev_table[47] = 76
        self.rev_table[48] = 66
        self.rev_table[49] = 56
        self.rev_table[50] = 46
        self.rev_table[51] = 36
        self.rev_table[52] = 26
        self.rev_table[53] = 16

        self.rev_table[54] = 97
        self.rev_table[55] = 87
        self.rev_table[56] = 77
        self.rev_table[57] = 67
        self.rev_table[58] = 57
        self.rev_table[59] = 47
        self.rev_table[60] = 37
        self.rev_table[61] = 27
        self.rev_table[62] = 17

        self.rev_table[63] = 98
        self.rev_table[64] = 88
        self.rev_table[65] = 78
        self.rev_table[66] = 68
        self.rev_table[67] = 58
        self.rev_table[68] = 48
        self.rev_table[69] = 38
        self.rev_table[70] = 28
        self.rev_table[71] = 18

        self.rev_table[72] = 99
        self.rev_table[73] = 89
        self.rev_table[74] = 79
        self.rev_table[75] = 69
        self.rev_table[76] = 59
        self.rev_table[77] = 49
        self.rev_table[78] = 39
        self.rev_table[79] = 29
        self.rev_table[80] = 19

    def csa_to_board(self, bo, pc, str_move):
        ifrom = (int)(str_move[0] + str_move[1])
        ito = (int)(str_move[2] + str_move[3])
        str_piece = str_move[4] + str_move[5]

        m = move.move(bo, pc)
        if ifrom != 0:
            m.ifrom = self.conv_table[ifrom]
        m.ito = self.conv_table[ito]
        m.cap_to_move = abs(bo.board[m.ito])

        if str_piece == 'FU':
            m.piece_to_move = pc.pawn
        elif str_piece == 'KY':
            m.piece_to_move = pc.lance
        elif str_piece == 'KE':
            m.piece_to_move = pc.knight
        elif str_piece == 'GI':
            m.piece_to_move = pc.silver
        elif str_piece == 'KI':
            m.piece_to_move = pc.gold
        elif str_piece == 'KA':
            m.piece_to_move = pc.bishop
        elif str_piece == 'HI':
            m.piece_to_move = pc.rook
        elif str_piece == 'OU':
            m.piece_to_move = pc.king
        elif str_piece == 'TO':
            if bo.board[m.ifrom] == pc.pawn or bo.board[m.ifrom] == -pc.pawn:
                m.piece_to_move = pc.pawn
                m.flag_promo = 1
            else:
                m.piece_to_move = pc.pro_pawn
        elif str_piece == 'NY':
            if bo.board[m.ifrom] == pc.lance or bo.board[m.ifrom] == -pc.lance:
                m.piece_to_move = pc.lance
                m.flag_promo = 1
            else:
                m.piece_to_move = pc.pro_lance
        elif str_piece == 'NK':
            if bo.board[m.ifrom] == pc.knight or bo.board[m.ifrom] == -pc.knight:
                m.piece_to_move = pc.knight
                m.flag_promo = 1
            else:
                m.piece_to_move = pc.pro_knight
        elif str_piece == 'NG':
            if bo.board[m.ifrom] == pc.silver or bo.board[m.ifrom] == -pc.silver:
                m.piece_to_move = pc.silver
                m.flag_promo = 1
            else:
                m.piece_to_move = pc.pro_silver
        elif str_piece == 'UM':
            if bo.board[m.ifrom] == pc.bishop or bo.board[m.ifrom] == -pc.bishop:
                m.piece_to_move = pc.bishop
                m.flag_promo = 1
            else:
                m.piece_to_move = pc.horse
        elif str_piece == 'RY':
            if bo.board[m.ifrom] == pc.rook or bo.board[m.ifrom] == -pc.rook:
                m.piece_to_move = pc.rook
                m.flag_promo = 1
            else:
                m.piece_to_move = pc.dragon

        return m

    def board_to_csa(self, bo, pc, m):
        str_piece = ''
        if m.ifrom == bo.square_nb:
            str_ifrom = '00'
        else:
            str_ifrom = str(self.rev_table[m.ifrom])
        str_ito = str(self.rev_table[m.ito])
        if m.flag_promo == 1:
            if m.piece_to_move == pc.pawn:
                str_piece = 'TO'
            elif m.piece_to_move == pc.lance:
                str_piece = 'NY'
            elif m.piece_to_move == pc.knight:
                str_piece = 'NK'
            elif m.piece_to_move == pc.silver:
                str_piece = 'NG'
            elif m.piece_to_move == pc.bishop:
                str_piece = 'UM'
            elif m.piece_to_move == pc.rook:
                str_piece = 'RY'
        else:
            if m.piece_to_move == pc.pawn:
                str_piece = 'FU'
            elif m.piece_to_move == pc.lance:
                str_piece = 'KY'
            elif m.piece_to_move == pc.knight:
                str_piece = 'KE'
            elif m.piece_to_move == pc.silver:
                str_piece = 'GI'
            elif m.piece_to_move == pc.gold:
                str_piece = 'KI'
            elif m.piece_to_move == pc.bishop:
                str_piece = 'KA'
            elif m.piece_to_move == pc.rook:
                str_piece = 'HI'
            elif m.piece_to_move == pc.king:
                str_piece = 'OU'
            elif m.piece_to_move == pc.pro_pawn:
                str_piece = 'TO'
            elif m.piece_to_move == pc.pro_lance:
                str_piece = 'NY'
            elif m.piece_to_move == pc.pro_knight:
                str_piece = 'NK'
            elif m.piece_to_move == pc.pro_silver:
                str_piece = 'NG'
            elif m.piece_to_move == pc.horse:
                str_piece = 'UM'
            elif m.piece_to_move == pc.dragon:
                str_piece = 'RY'
        str_move = str_ifrom + str_ito + str_piece
        return str_move