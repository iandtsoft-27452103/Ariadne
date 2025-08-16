class move:
    def __init__(self, bo, pc):
        self.ito = bo.square_nb
        self.ifrom = bo.square_nb
        self.flag_promo = 0
        self.piece_to_move = pc.empty
        self.cap_to_move = pc.empty