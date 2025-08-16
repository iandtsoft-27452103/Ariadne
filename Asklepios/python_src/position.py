import file
import rank
import board
import bitop
import makemove
import move
import usi
class position:
    def __init__(self):
        self.posi_no = 0
        self.record_no = 0
        self.ply = 0

    def number_position(self, rec):
        positions = []

        posi_cnt = 0
        for i in range(len(rec)):
            bo = board.board()
            pos = position()
            pos.posi_no = posi_cnt
            pos.record_no = i
            pos.ply = 0
            positions.append(pos)
            posi_cnt += 1
            for j in range(1, len(rec[i].moves)):
                pos = position()
                pos.posi_no = posi_cnt
                pos.record_no = i
                pos.ply = j
                positions.append(pos)
                posi_cnt += 1

        return positions

class set_position:
    def set_position(self, bo, bi, pc, c, cmd, start_index):
        ma = makemove.makemove()
        cls_usi = usi.usi()

        color = c
        ply = 1
        for i in range(start_index, len(cmd)):
            m = move.move(bo, pc)
            cls_usi.usi_to_move(bo, pc, cmd[i], m)
            ma.makemove(bo, m, ply, bi, pc, color)
            color ^= 1
            ply += 1

        return color