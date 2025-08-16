#Policy Network用
class teacher:
    def __init__(self, bm, sfen):
        self.str_sfen = sfen
        self.bestmove = bm

#Value Network用
class teacher_value:
    def __init__(self, r, sfen):
        self.str_sfen = sfen
        self.result = r