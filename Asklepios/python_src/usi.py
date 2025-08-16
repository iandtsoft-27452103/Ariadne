import rank

class usi:
    def __init__(self):
        self.cls_rank = rank.rank()
        self.sq_table = ['9a', '8a', '7a', '6a', '5a', '4a', '3a', '2a', '1a',
                         '9b', '8b', '7b', '6b', '5b', '4b', '3b', '2b', '1b',
                         '9c', '8c', '7c', '6c', '5c', '4c', '3c', '2c', '1c',
                         '9d', '8d', '7d', '6d', '5d', '4d', '3d', '2d', '1d',
                         '9e', '8e', '7e', '6e', '5e', '4e', '3e', '2e', '1e',
                         '9f', '8f', '7f', '6f', '5f', '4f', '3f', '2f', '1f',
                         '9g', '8g', '7g', '6g', '5g', '4g', '3g', '2g', '1g',
                         '9h', '8h', '7h', '6h', '5h', '4h', '3h', '2h', '1h',
                         '9i', '8i', '7i', '6i', '5i', '4i', '3i', '2i', '1i']

    def usi_to_move(self, bo, pc, str_move, move):
        if str_move[0] == 'P' or str_move[0] == 'p':
            move.piece_to_move = pc.pawn
        elif str_move[0] == 'L' or str_move[0] == 'l':
            move.piece_to_move = pc.lance
        elif str_move[0] == 'N' or str_move[0] == 'n':
            move.piece_to_move = pc.knight
        elif str_move[0] == 'S' or str_move[0] == 's':
            move.piece_to_move = pc.silver
        elif str_move[0] == 'G' or str_move[0] == 'g':
            move.piece_to_move = pc.gold
        elif str_move[0] == 'B' or str_move[0] == 'b':
            move.piece_to_move = pc.bishop
        elif str_move[0] == 'R' or str_move[0] == 'r':
            move.piece_to_move = pc.rook
        else:
            move.ifrom = self.usi_to_square(str_move[0], str_move[1], bo)
            move.piece_to_move = abs(bo.board[move.ifrom])
        
        move.ito = self.usi_to_square(str_move[2], str_move[3], bo)
        move.cap_to_move = abs(bo.board[move.ito])

        if len(str_move) == 5:
            move.flag_promo = 1
            
    def usi_to_square(self, str_file, str_rank, bo):
        f = bo.nfile - (int)(str_file)
        if str_rank == 'a':
            r = self.cls_rank.rank1
        elif str_rank == 'b':
            r = self.cls_rank.rank2
        elif str_rank == 'c':
            r = self.cls_rank.rank3
        elif str_rank == 'd':
            r = self.cls_rank.rank4
        elif str_rank == 'e':
            r = self.cls_rank.rank5
        elif str_rank == 'f':
            r = self.cls_rank.rank6
        elif str_rank == 'g':
            r = self.cls_rank.rank7
        elif str_rank == 'h':
            r = self.cls_rank.rank8
        else:
            r = self.cls_rank.rank9
        sq = r * bo.nrank + f
        return sq

    def board_to_usi(self, bo, color, pc, move):

        if move.ifrom == bo.square_nb:
            if move.piece_to_move == pc.pawn:
                str_usi = 'P*' + self.sq_table[move.ito]
            elif move.piece_to_move == pc.lance:
                str_usi = 'L*' + self.sq_table[move.ito]
            elif move.piece_to_move == pc.knight:
                str_usi = 'N*' + self.sq_table[move.ito]
            elif move.piece_to_move == pc.silver:
                str_usi = 'S*' + self.sq_table[move.ito]
            elif move.piece_to_move == pc.gold:
                str_usi = 'G*' + self.sq_table[move.ito]
            elif move.piece_to_move == pc.bishop:
                str_usi = 'B*' + self.sq_table[move.ito]
            elif move.piece_to_move == pc.rook:
                str_usi = 'R*' + self.sq_table[move.ito]
            return str_usi

        str_usi = self.sq_table[move.ifrom] + self.sq_table[move.ito]

        if move.flag_promo == 1:
            str_usi += '+'

        return str_usi