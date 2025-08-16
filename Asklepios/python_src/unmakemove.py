class unmakemove:
    def unmakemove(self, board, move, ply, bitop, pc, color):
        if color == 0:
            sign = 1
        else:
            sign = -1

        if move.ifrom == board.square_nb:
            #駒打ちの場合
            board.board[move.ito] = pc.empty
            board.bb_occupied = bitop.xor(move.ito, board.bb_occupied)
            board.bb_occ_color[color] = bitop.xor(move.ito, board.bb_occ_color[color])
            if move.piece_to_move == pc.pawn:
                board.bb_pawn[color] = bitop.xor(move.ito, board.bb_pawn[color])
                board.bb_pawn_attacks[color] = bitop.xor(move.ito - (sign * 9), board.bb_pawn_attacks[color])
                board.hand_pawn[color] += 1
            elif move.piece_to_move == pc.lance:
                board.bb_lance[color] = bitop.xor(move.ito, board.bb_lance[color])
                board.hand_lance[color] += 1
            elif move.piece_to_move == pc.knight:
                board.bb_knight[color] = bitop.xor(move.ito, board.bb_knight[color])
                board.hand_knight[color] += 1
            elif move.piece_to_move == pc.silver:
                board.bb_silver[color] = bitop.xor(move.ito, board.bb_silver[color])
                board.hand_silver[color] += 1
            elif move.piece_to_move == pc.gold:
                board.bb_gold[color] = bitop.xor(move.ito, board.bb_gold[color])
                board.bb_total_gold[color] = bitop.xor(move.ito, board.bb_total_gold[color])
                board.hand_gold[color] += 1
            elif move.piece_to_move == pc.bishop:
                board.bb_bishop[color] = bitop.xor(move.ito, board.bb_bishop[color])
                board.bb_bh[color] = bitop.xor(move.ito, board.bb_bh[color])
                board.hand_bishop[color] += 1
            elif move.piece_to_move == pc.rook:
                board.bb_rook[color] = bitop.xor(move.ito, board.bb_rook[color])
                board.bb_rd[color] = bitop.xor(move.ito, board.bb_rd[color])
                board.hand_rook[color] += 1
        else:
            #駒を動かす場合
            bb_set_clear = bitop.bb_or(bitop.bb_mask[move.ifrom], bitop.bb_mask[move.ito])
            board.bb_occupied = bitop.bb_xor(board.bb_occupied, bb_set_clear)
            board.bb_occ_color[color] = bitop.bb_xor(board.bb_occ_color[color], bb_set_clear)
            board.board[move.ifrom] = sign * move.piece_to_move
            board.board[move.ito] = pc.empty
            if move.flag_promo == 1:
                #成る手の場合
                if move.piece_to_move == pc.pawn:
                    board.bb_pawn[color] = bitop.xor(move.ifrom, board.bb_pawn[color])
                    board.bb_pawn_attacks[color] = bitop.xor(move.ito, board.bb_pawn_attacks[color])
                    board.bb_pro_pawn[color] = bitop.xor(move.ito, board.bb_pro_pawn[color])
                    board.bb_total_gold[color] = bitop.xor(move.ito, board.bb_total_gold[color])
                elif move.piece_to_move == pc.lance:
                    board.bb_lance[color] = bitop.xor(move.ifrom, board.bb_lance[color])
                    board.bb_pro_lance[color] = bitop.xor(move.ito, board.bb_pro_lance[color])
                    board.bb_total_gold[color] = bitop.xor(move.ito, board.bb_total_gold[color])
                elif move.piece_to_move == pc.knight:
                    board.bb_knight[color] = bitop.xor(move.ifrom, board.bb_knight[color])
                    board.bb_pro_knight[color] = bitop.xor(move.ito, board.bb_pro_knight[color])
                    board.bb_total_gold[color] = bitop.xor(move.ito, board.bb_total_gold[color])
                elif move.piece_to_move == pc.silver:
                    board.bb_silver[color] = bitop.xor(move.ifrom, board.bb_silver[color])
                    board.bb_pro_silver[color] = bitop.xor(move.ito, board.bb_pro_silver[color])
                    board.bb_total_gold[color] = bitop.xor(move.ito, board.bb_total_gold[color])
                elif move.piece_to_move == pc.bishop:
                    board.bb_bishop[color] = bitop.xor(move.ifrom, board.bb_bishop[color])
                    board.bb_horse[color] = bitop.xor(move.ito, board.bb_horse[color])
                    #board.bb_bh[color] = bitop.xor(move.ito, board.bb_bh[color])
                    board.bb_bh[color] = bitop.bb_xor(board.bb_bh[color], bb_set_clear)
                    board.bb_hdk[color] = bitop.xor(move.ito, board.bb_hdk[color])
                elif move.piece_to_move == pc.rook:
                    board.bb_rook[color] = bitop.xor(move.ifrom, board.bb_rook[color])
                    board.bb_dragon[color] = bitop.xor(move.ito, board.bb_dragon[color])
                    #board.bb_rd[color] = bitop.xor(move.ito, board.bb_rd[color])
                    board.bb_rd[color] = bitop.bb_xor(board.bb_rd[color], bb_set_clear)
                    board.bb_hdk[color] = bitop.xor(move.ito, board.bb_hdk[color])
            else:
                #成らない手の場合
                if move.piece_to_move == pc.pawn:
                    board.bb_pawn[color] = bitop.bb_xor(board.bb_pawn[color], bb_set_clear)
                    board.bb_pawn_attacks[color] = bitop.xor(move.ito, board.bb_pawn_attacks[color])
                    board.bb_pawn_attacks[color] = bitop.xor(move.ito - (sign * 9), board.bb_pawn_attacks[color])
                elif move.piece_to_move == pc.lance:
                    board.bb_lance[color] = bitop.bb_xor(board.bb_lance[color], bb_set_clear)
                elif move.piece_to_move == pc.knight:
                    board.bb_knight[color] = bitop.bb_xor(board.bb_knight[color], bb_set_clear)
                elif move.piece_to_move == pc.silver:
                    board.bb_silver[color] = bitop.bb_xor(board.bb_silver[color], bb_set_clear)
                elif move.piece_to_move == pc.gold:
                    board.bb_gold[color] = bitop.bb_xor(board.bb_gold[color], bb_set_clear)
                    board.bb_total_gold[color] = bitop.bb_xor(board.bb_total_gold[color], bb_set_clear)
                elif move.piece_to_move == pc.bishop:
                    board.bb_bishop[color] = bitop.bb_xor(board.bb_bishop[color], bb_set_clear)
                    board.bb_bh[color] = bitop.bb_xor(board.bb_bh[color], bb_set_clear)
                elif move.piece_to_move == pc.rook:
                    board.bb_rook[color] = bitop.bb_xor(board.bb_rook[color], bb_set_clear)
                    board.bb_rd[color] = bitop.bb_xor(board.bb_rd[color], bb_set_clear)
                elif move.piece_to_move == pc.king:
                    board.bb_hdk[color] = bitop.bb_xor(board.bb_hdk[color], bb_set_clear)
                    board.sq_king[color] = move.ifrom
                elif move.piece_to_move == pc.pro_pawn:
                    board.bb_pro_pawn[color] = bitop.bb_xor(board.bb_pro_pawn[color], bb_set_clear)
                    board.bb_total_gold[color] = bitop.bb_xor(board.bb_total_gold[color], bb_set_clear)
                elif move.piece_to_move == pc.pro_lance:
                    board.bb_pro_lance[color] = bitop.bb_xor(board.bb_pro_lance[color], bb_set_clear)
                    board.bb_total_gold[color] = bitop.bb_xor(board.bb_total_gold[color], bb_set_clear)
                elif move.piece_to_move == pc.pro_knight:
                    board.bb_pro_knight[color] = bitop.bb_xor(board.bb_pro_knight[color], bb_set_clear)
                    board.bb_total_gold[color] = bitop.bb_xor(board.bb_total_gold[color], bb_set_clear)
                elif move.piece_to_move == pc.pro_silver:
                    board.bb_pro_silver[color] = bitop.bb_xor(board.bb_pro_silver[color], bb_set_clear)
                    board.bb_total_gold[color] = bitop.bb_xor(board.bb_total_gold[color], bb_set_clear)
                elif move.piece_to_move == pc.horse:
                    board.bb_horse[color] = bitop.bb_xor(board.bb_horse[color], bb_set_clear)
                    board.bb_bh[color] = bitop.bb_xor(board.bb_bh[color], bb_set_clear)
                    board.bb_hdk[color] = bitop.bb_xor(board.bb_hdk[color], bb_set_clear)
                elif move.piece_to_move == pc.dragon:
                    board.bb_dragon[color] = bitop.bb_xor(board.bb_dragon[color], bb_set_clear)
                    board.bb_rd[color] = bitop.bb_xor(board.bb_rd[color], bb_set_clear)
                    board.bb_hdk[color] = bitop.bb_xor(board.bb_hdk[color], bb_set_clear)

            if move.cap_to_move != pc.empty:
                board.board[move.ito] = -sign * move.cap_to_move
                board.bb_occupied = bitop.xor(move.ito, board.bb_occupied)
                board.bb_occ_color[color ^ 1] = bitop.xor(move.ito, board.bb_occ_color[color ^ 1])
                #駒取りの場合
                if move.cap_to_move == pc.pawn:
                    board.bb_pawn[color ^ 1] = bitop.xor(move.ito, board.bb_pawn[color ^ 1])
                    board.bb_pawn_attacks[color ^ 1] = bitop.xor(move.ito + (sign * 9), board.bb_pawn_attacks[color ^ 1])
                    board.hand_pawn[color] -= 1
                elif move.cap_to_move == pc.lance:
                    board.bb_lance[color ^ 1] = bitop.xor(move.ito, board.bb_lance[color ^ 1])
                    board.hand_lance[color] -= 1
                elif move.cap_to_move == pc.knight:
                    board.bb_knight[color ^ 1] = bitop.xor(move.ito, board.bb_knight[color ^ 1])
                    board.hand_knight[color] -= 1
                elif move.cap_to_move == pc.silver:
                    board.bb_silver[color ^ 1] = bitop.xor(move.ito, board.bb_silver[color ^ 1])
                    board.hand_silver[color] -= 1
                elif move.cap_to_move == pc.gold:
                    board.bb_gold[color ^ 1] = bitop.xor(move.ito, board.bb_gold[color ^ 1])
                    board.bb_total_gold[color ^ 1] = bitop.xor(move.ito, board.bb_total_gold[color ^ 1])
                    board.hand_gold[color] -= 1
                elif move.cap_to_move == pc.bishop:
                    board.bb_bishop[color ^ 1] = bitop.xor(move.ito, board.bb_bishop[color ^ 1])
                    board.bb_bh[color ^ 1] = bitop.xor(move.ito, board.bb_bh[color ^ 1])
                    board.hand_bishop[color] -= 1
                elif move.cap_to_move == pc.rook:
                    board.bb_rook[color ^ 1] = bitop.xor(move.ito, board.bb_rook[color ^ 1])
                    board.bb_rd[color ^ 1] = bitop.xor(move.ito, board.bb_rd[color ^ 1])
                    board.hand_rook[color] -= 1
                elif move.cap_to_move == pc.pro_pawn:
                    board.bb_pro_pawn[color ^ 1] = bitop.xor(move.ito, board.bb_pro_pawn[color ^ 1])
                    board.bb_total_gold[color ^ 1] = bitop.xor(move.ito, board.bb_total_gold[color ^ 1])
                    board.hand_pawn[color] -= 1
                elif move.cap_to_move == pc.pro_lance:
                    board.bb_pro_lance[color ^ 1] = bitop.xor(move.ito, board.bb_pro_lance[color ^ 1])
                    board.bb_total_gold[color ^ 1] = bitop.xor(move.ito, board.bb_total_gold[color ^ 1])
                    board.hand_lance[color] -= 1
                elif move.cap_to_move == pc.pro_knight:
                    board.bb_pro_knight[color ^ 1] = bitop.xor(move.ito, board.bb_pro_knight[color ^ 1])
                    board.bb_total_gold[color ^ 1] = bitop.xor(move.ito, board.bb_total_gold[color ^ 1])
                    board.hand_knight[color] -= 1
                elif move.cap_to_move == pc.pro_silver:
                    board.bb_pro_silver[color ^ 1] = bitop.xor(move.ito, board.bb_pro_silver[color ^ 1])
                    board.bb_total_gold[color ^ 1] = bitop.xor(move.ito, board.bb_total_gold[color ^ 1])
                    board.hand_silver[color] -= 1
                elif move.cap_to_move == pc.horse:
                    board.bb_horse[color ^ 1] = bitop.xor(move.ito, board.bb_horse[color ^ 1])
                    board.bb_bh[color ^ 1] = bitop.xor(move.ito, board.bb_bh[color ^ 1])
                    board.bb_hdk[color ^ 1] = bitop.xor(move.ito, board.bb_hdk[color ^ 1])
                    board.hand_bishop[color] -= 1
                elif move.cap_to_move == pc.dragon:
                    board.bb_dragon[color ^ 1] = bitop.xor(move.ito, board.bb_dragon[color ^ 1])
                    board.bb_rd[color ^ 1] = bitop.xor(move.ito, board.bb_rd[color ^ 1])
                    board.bb_hdk[color ^ 1] = bitop.xor(move.ito, board.bb_hdk[color ^ 1])
                    board.hand_rook[color] -= 1