import record
import board
import makemove
import piece
import bitop
import color
import csa
import sfen
import bitop
import teacher
import csa

class inout:
    def __init__(self):
        self.rec = []
        self.teacher = []
        self.ma = makemove.makemove()
        self.pc = piece.piece()
        self.bo = board.board()
        self.bitop = bitop.bitop(self.bo)
        self.c = color.color()
        self.csa = csa.csa()
        #bi = bitop.bitop(self.bo)
        #self.sfen = sfen.sfen(self.bo, bi)

    #学習用の棋譜を読み込む
    def read_records(self, file_name):
        file = open(file_name, 'r', 1, 'UTF-8')
        i = 0
        for line in file:
            line = line.replace('\n', '')
            line = line.replace(' ', '')
            line = line.replace('%TORYO,', '')
            s = line.split(',')
            temp = record.record()
            temp.result = s[0]
            c = 0
            ply = 1
            self.bo.init_board(self.bo.board_default, self.bo.hand_default, self.bitop, self.c.black)
            for j in range(2, len(s)):
                if s[j] == '/':
                    break
                x = s[j].replace('+', '')
                x = x.replace('-', '')
                temp.str_moves.append(x)
                move = self.csa.csa_to_board(self.bo, self.pc, x)
                temp.moves.append(move)

                self.ma.makemove(self.bo, move, ply, self.bitop, self.pc, c)

                c = c ^ 1
                ply += 1
            temp.ply = len(temp.moves)
            self.rec.append(temp)
            #print(i)
            i += 1

        file.close()

    #Aperyで生成したSFEN局面を読み込む（Policy Network用）
    def read_sfen(self, file_name):
        file = open(file_name, 'r', 1, 'UTF-8')
        i = 0
        for line in file:
            line = line.replace('\n', '')
            line = line.replace('\ufeff', '')
            s = line.split(',')
            bestmove = s[1]
            s2 = s[0].split(' ')
            s2 = s2[1] + ' ' + s2[2] + ' ' +  s2[3] + ' ' + s2[4]
            t = teacher.teacher(bestmove, s2)
            self.teacher.append(t)
            i += 1

        file.close()

    #Aperyで生成したSFEN局面を読み込む（Value Network用）
    def read_sfen_value(self, file_name):
        file = open(file_name, 'r', 1, 'UTF-8')
        i = 0
        for line in file:
            line = line.replace('\n', '')
            line = line.replace('\ufeff', '')
            s = line.split(',')
            result = s[2]
            s2 = s[0].split(' ')
            s2 = s2[1] + ' ' + s2[2] + ' ' +  s2[3] + ' ' + s2[4]
            t = teacher.teacher_value(result, s2)
            self.teacher.append(t)
            i += 1

        file.close()