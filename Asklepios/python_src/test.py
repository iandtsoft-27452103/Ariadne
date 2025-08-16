import record
import board
import makemove
import unmakemove
import piece
import bitop
import color
import csa

class test:
    def __init__(self):
        self.rec = []
        self.ma = makemove.makemove()
        self.um = unmakemove.unmakemove()
        self.pc = piece.piece()
        self.bo = board.board()
        self.bitop = bitop.bitop(self.bo)
        self.c = color.color()
        self.csa = csa.csa()

    #局面更新のテスト
    def test1(self):
        file = open('test_records.txt', 'r', 1, 'UTF-8')
        i = 0
        for line in file:
            line = line.replace('\n', '')
            line = line.replace(' ', '')
            line = line.replace('%TORYO,', '')
            s = line.split(',')
            temp = record.record()
            c = 0
            ply = 1
            self.bo.init_board(self.bo.board_default, self.bo.hand_default, self.bitop, self.c.black)
            for j in range(2, len(s) - 1):
                if s[j] == '/':
                    break
                x = s[j].replace('+', '')
                x = x.replace('-', '')
                temp.str_moves.append(x)
                move = self.csa.csa_to_board(self.bo, self.pc, x)
                temp.moves.append(move)

                self.ma.makemove(self.bo, move, ply, self.bitop, self.pc, c)

                #if i == 0 and ply == 99:
                    #self.out_board(self.bo)

                c = c ^ 1
                ply += 1
            temp.ply = len(temp.moves)
            self.rec.append(temp)
            self.out_board(self.bo)
            print(i)
            i += 1

        file.close()
    
    #局面戻しのテスト
    def test2(self):
        file = open('test_records.txt', 'r', 1, 'UTF-8')
        i = 0
        for line in file:
            line = line.replace('\n', '')
            line = line.replace(' ', '')
            line = line.replace('%TORYO,', '')
            s = line.split(',')
            temp = record.record()
            c = 0
            ply = 1
            self.bo.init_board(self.bo.board_default, self.bo.hand_default, self.bitop, self.c.black)
            for j in range(2, len(s) - 1):
                if s[j] == '/':
                    break
                x = s[j].replace('+', '')
                x = x.replace('-', '')
                temp.str_moves.append(x)
                move = self.csa.csa_to_board(self.bo, self.pc, x)
                temp.moves.append(move)

                self.ma.makemove(self.bo, move, ply, self.bitop, self.pc, c)

                if i == 0 and ply == 125:
                    print(x)
                    self.out_board(self.bo)
                    self.um.unmakemove(self.bo, move, ply, self.bitop, self.pc, c)
                    self.out_board(self.bo)
                    file.close()
                    return

                c = c ^ 1
                ply += 1
            temp.ply = len(temp.moves)
            self.rec.append(temp)
            #self.out_board(self.bo)
            #print(i)
            i += 1

        file.close()

    #盤面の出力
    def out_board(self, bo):
        print('   9   8   7   6   5   4   3   2   1')
        for irank in range(bo.nrank):
            s = '|'
            for ifile in range(bo.nfile):
                pc = bo.board[(irank * bo.nrank) + ifile]
                if pc == self.pc.empty:
                    s += '   '
                elif pc == self.pc.pawn:
                    s += ' 歩'
                elif pc == self.pc.lance:
                    s += ' 香'
                elif pc == self.pc.knight:
                    s += ' 桂'
                elif pc == self.pc.silver:
                    s += ' 銀'
                elif pc == self.pc.gold:
                    s += ' 金'
                elif pc == self.pc.bishop:
                    s += ' 角'
                elif pc == self.pc.rook:
                    s += ' 飛'
                elif pc == self.pc.king:
                    s += ' 玉'
                elif pc == self.pc.pro_pawn:
                    s += ' と'
                elif pc == self.pc.pro_lance:
                    s += ' 杏'
                elif pc == self.pc.pro_knight:
                    s += ' 圭'
                elif pc == self.pc.pro_silver:
                    s += ' 全'
                elif pc == self.pc.horse:
                    s += ' 馬'
                elif pc == self.pc.dragon:
                    s += ' 龍'
                elif pc == -self.pc.pawn:
                    s += 'v歩'
                elif pc == -self.pc.lance:
                    s += 'v香'
                elif pc == -self.pc.knight:
                    s += 'v桂'
                elif pc == -self.pc.silver:
                    s += 'v銀'
                elif pc == -self.pc.gold:
                    s += 'v金'
                elif pc == -self.pc.bishop:
                    s += 'v角'
                elif pc == -self.pc.rook:
                    s += 'v飛'
                elif pc == -self.pc.king:
                    s += 'v玉'
                elif pc == -self.pc.pro_pawn:
                    s += 'vと'
                elif pc == -self.pc.pro_lance:
                    s += 'v杏'
                elif pc == -self.pc.pro_knight:
                    s += 'v圭'
                elif pc == -self.pc.pro_silver:
                    s += 'v全'
                elif pc == -self.pc.horse:
                    s += 'v馬'
                elif pc == -self.pc.dragon:
                    s += 'v龍'
                s += '|'
            s += (str)(irank + 1)
            print('____________________________________')
            print(s)
        print('____________________________________')
        s = '先持：'
        s += '歩' + (str)(bo.hand_pawn[0]) + ','
        s += '香' + (str)(bo.hand_lance[0]) + ','
        s += '桂' + (str)(bo.hand_knight[0]) + ','
        s += '銀' + (str)(bo.hand_silver[0]) + ','
        s += '金' + (str)(bo.hand_gold[0]) + ','
        s += '角' + (str)(bo.hand_bishop[0]) + ','
        s += '飛' + (str)(bo.hand_rook[0])
        print(s)
        s = '後持：'
        s += '歩' + (str)(bo.hand_pawn[1]) + ','
        s += '香' + (str)(bo.hand_lance[1]) + ','
        s += '桂' + (str)(bo.hand_knight[1]) + ','
        s += '銀' + (str)(bo.hand_silver[1]) + ','
        s += '金' + (str)(bo.hand_gold[1]) + ','
        s += '角' + (str)(bo.hand_bishop[1]) + ','
        s += '飛' + (str)(bo.hand_rook[1])
        print(s)
        s = '先-歩：'
        bb = bo.bb_pawn[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-香：'
        bb = bo.bb_lance[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-桂：'
        bb = bo.bb_knight[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-銀：'
        bb = bo.bb_silver[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-金：'
        bb = bo.bb_gold[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-角：'
        bb = bo.bb_bishop[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-飛：'
        bb = bo.bb_rook[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-玉：'
        s += (str)(bo.sq_king[0])
        print(s)
        s = '先-と：'
        bb = bo.bb_pro_pawn[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-杏：'
        bb = bo.bb_pro_lance[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-圭：'
        bb = bo.bb_pro_knight[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-全：'
        bb = bo.bb_pro_silver[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-馬：'
        bb = bo.bb_horse[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '先-龍：'
        bb = bo.bb_dragon[0]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-歩：'
        bb = bo.bb_pawn[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-香：'
        bb = bo.bb_lance[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-桂：'
        bb = bo.bb_knight[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-銀：'
        bb = bo.bb_silver[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-金：'
        bb = bo.bb_gold[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-角：'
        bb = bo.bb_bishop[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-飛：'
        bb = bo.bb_rook[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-玉：'
        s += (str)(bo.sq_king[1])
        print(s)
        s = '後-と：'
        bb = bo.bb_pro_pawn[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-杏：'
        bb = bo.bb_pro_lance[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-圭：'
        bb = bo.bb_pro_knight[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-全：'
        bb = bo.bb_pro_silver[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-馬：'
        bb = bo.bb_horse[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        s = '後-龍：'
        bb = bo.bb_dragon[1]
        while bb != 0:
            sq = self.bitop.first_one(bb)
            bb = self.bitop.xor(sq, bb)
            s += (str)(sq) + ' '
        print(s)
        print('')