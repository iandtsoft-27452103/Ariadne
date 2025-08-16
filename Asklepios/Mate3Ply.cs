using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Mate1Ply;
using static Asklepios.Board;
using static Asklepios.AttackBitBoard;
using static Asklepios.MakeMove;
using static Asklepios.ShogiCommon;
using static Asklepios.ShogiCommon.Direction;
using static Asklepios.ShogiCommon.Square;
using static Asklepios.ShogiCommon.File;
using static Asklepios.BitOperation;
using System.Runtime.CompilerServices;
using System.Numerics;
using System.Reflection.Metadata;

namespace Asklepios
{
    internal class Mate3Ply
    {
        enum MateType {
            mate_king_cap_checker = 0,
            mate_cap_checker_gen,
            mate_cap_checker,
            mate_king_cap_gen,
            mate_king_cap,
            mate_king_move_gen,
            mate_king_move,
            mate_intercept_move_gen,
            mate_intercept_move,
            mate_intercept_weak_move,
            mate_intercept_drop_sup
        };

        public Mate3Ply()
        {
            Mate3Dict.Clear();
        }

        public Dictionary<ulong, Move> Mate3Dict = new Dictionary<ulong, Move>();


        // ※詰みがあった場合に3手のpvを返す
        public bool IsMateIn3Ply(ref BoardTree BTree, int color, bool use_hash, ref Move[] three_moves)
        {
            Move move = new Move();
            List<NextMove> next_move_list = new List<NextMove>();
            NextMove next_move = new NextMove();
            if (use_hash && Mate3Dict.TryGetValue(BTree.CurrentHash, out move))
            {
                uint v = move.Value;
                if (v == 0U) { return false; }
                else { return true; }
            }

            bool flag_skip = false;

            List<CheckMove> move_list = new List<CheckMove>();
            if (color == 0) { GenCheck.BGenChecks(ref BTree, ref move_list); }
            else { GenCheck.WGenChecks(ref BTree, ref move_list); }

            for (int i = 0; i < move_list.Count; i++)
            {
                next_move_list.Clear();

                CheckMove current_check_move = move_list[i];
                if (current_check_move.check_type == 1) { flag_skip = false; }
                if (flag_skip ) { continue; }

                Do(ref BTree, color, current_check_move);
                if (IsAttacked(BTree, BTree.SQ_King[color], color ^ 1) != 0)
                {
                    UnDo(ref BTree, color, current_check_move);
                    continue;
                }

                int value = Mate3And(ref BTree, color ^ 1, 0, current_check_move, ref next_move, ref next_move_list, ref three_moves);

                UnDo(ref BTree, color, current_check_move);

                if (value != 0)
                {
                    if (use_hash)
                    {
                        Mate3Dict.TryAdd(BTree.CurrentHash, current_check_move);
                    }                       
                    three_moves[0] = current_check_move;
                    return true;
                }

                if (current_check_move.check_type == 2 && current_check_move.To != next_move.To)
                {
                    flag_skip = true;
                }
            }

            Move null_move = new Move();
            if (use_hash)
            {
                Mate3Dict.TryAdd(BTree.CurrentHash, null_move);
            }

            return false;
        }

        int Mate3And(ref BoardTree BTree, int color, int flag, CheckMove move_last, ref NextMove next_move, ref List<NextMove> next_move_list, ref Move[] three_moves)
        {
            int move;
            int gen_move_count = 0;
            int index = 0;
            int[] sq = new int[2];
            NextMove evasion_move = new NextMove();
            Move mate_move = new Move();
            next_move.next_phase = (int)MateType.mate_king_cap_checker;
            Checker(BTree, ref sq, color);

            while (GenNextEvasionMate(ref BTree, sq, color, flag, ref next_move, move_last, ref next_move_list, ref index, ref gen_move_count))
            {
                /*if (next_move_list.Count > 30)
                {
                    int x = 0;
                }*/

                if (next_move.next_phase == (int)MateType.mate_intercept_drop_sup)
                {
                    return 0;
                }

                Do(ref BTree, color, next_move);

                if (IsAttacked(BTree, BTree.SQ_King[color ^ 1], color ^ 1) != 0) { move = 0; }
                else 
                { 
                    move = IsMateIn1Ply(ref BTree, ref mate_move, color ^ 1);
                    evasion_move = next_move_list[next_move_list.Count - 1];
                }

                /*if (move == 0 && next_move.next_phase == (int)MateType.mate_intercept_weak_move)
                {
                    move = MateWeakOr(ref BTree, color ^ 1, sq[0], next_move.To, move_last, ref next_mover_list, ref three_moves);
                }*/

                UnDo(ref BTree, color, next_move);

                if (move == 0) { return 0; }
            }

            three_moves[1] = evasion_move;
            three_moves[2] = mate_move;

            return 1;
        }

        void Checker(BoardTree BTree, ref int[] sq, int color)
        {
            int n, sq0, sq1;
            int sq_king = BTree.SQ_King[color];
            BitBoard bb = BitBoard.Copy(AttacksToPiece(BTree, sq_king, color ^ 1));
            sq0 = LastOne012(bb.p[0], bb.p[1], bb.p[2]);// 本来あり得ないが、王手している駒を返さないことがある
            sq1 = NSquare;
            Xor(sq0, ref bb);
            if (BBTest(bb) != 0)
            {
                sq1 = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                if (BBContract(AbbPieceAttacks[color, (int)Piece.Type.King, sq_king], AbbMask[sq1]) != 0)
                {
                    n = sq0;
                    sq0 = sq1;
                    sq1 = n;
                }
            }
            sq[0] = sq0;
            sq[1] = sq1;
        }

        void GenKingMove(BoardTree BTree, int[] sq, int color, bool is_capture, ref List<NextMove> move_list)
        {
            int ifrom = BTree.SQ_King[color];
            BitBoard bb = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.King, ifrom]);
            if (is_capture) 
            { 
                BBAnd(ref bb, bb, BTree.BB_Occupied[color ^ 1]);
                BBNotAnd(ref bb, bb, AbbMask[sq[0]]);
            }
            else { BBNotAnd(ref bb, bb, BTree.BB_Occupied[color ^ 1]); }
            BBNotAnd(ref bb, bb, BTree.BB_Occupied[color]);

            if (BBTest(bb) == 0 && sq[0] < NSquare)
            {
                bb = BitBoard.Copy(AbbMask[sq[0]]);
            }

            while (BBTest(bb) != 0)
            {
                int ito = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(ito, ref bb);
                if (sq[1] != NSquare && (ADirec[ifrom, sq[1]] == ADirec[ifrom, ito])) { continue; }
                if (sq[0] != ito && ADirec[ifrom, sq[0]] == ADirec[ifrom, ito])
                {
                    if ((ADirec[ifrom, sq[0]] & (uint)flag_cross) != 0)
                    {
                        if (Math.Abs(BTree.ShogiBoard[sq[0]]) == (int)Piece.Type.Lance
                            || Math.Abs(BTree.ShogiBoard[sq[0]]) == (int)Piece.Type.Rook
                            || Math.Abs(BTree.ShogiBoard[sq[0]]) == (int)Piece.Type.Dragon)
                        {
                            continue;
                        }
                    }
                    else if ((ADirec[ifrom, sq[0]] & (uint)flag_diag) != 0
                        && (Math.Abs(BTree.ShogiBoard[sq[0]]) == (int)Piece.Type.Bishop
                            || Math.Abs(BTree.ShogiBoard[sq[0]]) == (int)Piece.Type.Horse))
                    {
                        continue;
                    }
                }

                if (IsAttacked(BTree, ito, color) != 0) { continue; }
                NextMove move = new NextMove();
                move.Push(ifrom, ito, Piece.Type.King, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                move_list.Add(move);
            }
        }

        NextMove GenKingCapChecker(BoardTree BTree, int ito, int color)
        {
            int ifrom = BTree.SQ_King[color];
            NextMove move = new NextMove();
            if (BBContract(AbbPieceAttacks[color, (int)Piece.Type.King, ifrom], AbbMask[ito]) == 0) { return move; }
            if (IsAttacked(BTree, ito, color ^ 1) != 0) { return move; }
            move.Push(ifrom, ito, Piece.Type.King, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
            return move;
        }

        /*int MateWeakOr(ref BoardTree BTree, int color, int ifrom, int ito, CheckMove move_last, ref List<NextMove> next_move_list, ref Move[] three_moves)
        {
            if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { return 0; }
            Piece.Type pc = (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]);
            int pc_cap = (int)Math.Abs(BTree.ShogiBoard[ito]);
            Move move = new Move();
            if ((pc == Piece.Type.Bishop || pc == Piece.Type.Rook) && (CompareFormula0(color, ifrom) || CompareFormula0(color, ito)))
            {
                move.Push(ifrom, ito, pc, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
            }
            else
            {
                move.Push(ifrom, ito, pc, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
            }
            Do(ref BTree, color, move);

            int flag;

            if (move_last.From < NSquare)
            {
                if (IsAttacked(BTree, BTree.SQ_King[color], color) != 0)
                {
                    UnDo(ref BTree, color, move);
                    return 0;
                }
                flag = 1;
            }
            else { flag = 2; }

            int value = Mate3And(ref BTree, color ^ 1, flag, move_last, ref move, ref next_move_list, ref three_moves);

            UnDo(ref BTree, color, move);

            return value;
        }*/

        bool GenNextEvasionMate(ref BoardTree BTree, int[] sq, int color, int flag, ref NextMove next_move, CheckMove move_last, ref List<NextMove> next_move_list, ref int index, ref int gen_move_count)
        {
            int before, after, start;

            if (gen_move_count == 1)
            {
                return false;
            }
            else if (gen_move_count > 1)
            {
                gen_move_count--;
                next_move = next_move_list[++index];
                next_move.next_phase = next_move_list[index].next_phase;
                return true;
            }

            before = after = 0;

            start = next_move_list.Count;

            switch ((MateType)next_move.next_phase)
            {
                case MateType.mate_king_cap_checker:
                    goto a;
                case MateType.mate_cap_checker_gen:
                    goto b;
                case MateType.mate_cap_checker:
                    goto c;
                case MateType.mate_king_cap_gen:
                    goto d;
                case MateType.mate_king_cap:
                    goto e;
                case MateType.mate_king_move_gen:
                    goto f;
                case MateType.mate_king_move:
                    goto g;
                case MateType.mate_intercept_move_gen:
                    goto h;
                case MateType.mate_intercept_move:
                    goto i;
                case MateType.mate_intercept_weak_move:
                    goto j;
                default:// MateType.mate_intercept_drop_sup
                    goto k;
            }

        a:
            NextMove m = GenKingCapChecker(BTree, sq[0], color);
            next_move = m;
            next_move.next_phase = (int)MateType.mate_cap_checker_gen;
            if (next_move.Value != 0)
            {
                gen_move_count = 0;// 実際は1手生成している
                next_move_list.Add(m);
                return true;
            }
        b:
            next_move.next_phase = (int)MateType.mate_cap_checker;
            before = next_move_list.Count;
            if (sq[1] == NSquare)
            {
                GenMoveTo(BTree, sq[0], color, ref next_move_list);
            }
            after = next_move_list.Count;
        c:
            if (before != after)
            {
                gen_move_count = after - before;
                next_move = next_move_list[before];
                index = before;
                return true;
            }
        d:
            next_move.next_phase = (int)MateType.mate_king_cap;
            before = next_move_list.Count;
            GenKingMove(BTree, sq, color, true, ref next_move_list);
            after = next_move_list.Count;
        e:
            if (before != after)
            {
                gen_move_count = after - before;
                next_move = next_move_list[before];
                next_move.next_phase = (int)MateType.mate_king_move_gen;
                index = before;
                return true;
            }
        f:
            next_move.next_phase = (int)MateType.mate_king_move;
            before = next_move_list.Count;
            GenKingMove(BTree, sq, color, false, ref next_move_list);
            after = next_move_list.Count;
        g:
            if (before != after)// 複数の手を生成していたら、次回は次のインデックスに移るだけにする
            {
                if (before == 0)
                {
                    gen_move_count = after - before;
                    next_move = next_move_list[before];
                    next_move_list[next_move_list.Count - 1].next_phase = (int)MateType.mate_intercept_move_gen;
                    index = before;
                    //next_move.next_phase = (int)MateType.mate_king_move;
                    return true;
                }
                else
                {
                    if (next_move_list[before].Value != next_move_list[before - 1].Value)
                    {
                        gen_move_count = after - before;
                        next_move = next_move_list[before];
                        next_move_list[next_move_list.Count - 1].next_phase = (int)MateType.mate_intercept_move_gen;
                        index = before;
                        //next_move.next_phase = (int)MateType.mate_king_move;
                        return true;
                    }
                }
            }
        h:
            next_move.next_phase = (int)MateType.mate_intercept_move;
            before = next_move_list.Count;
            if (sq[1] == NSquare && Math.Abs(BTree.ShogiBoard[sq[0]]) != (int)Piece.Type.Knight)
            {
                int n = 0;// これで大丈夫か？
                GenIntercept(BTree, sq[0], color, ref n, move_last, ref next_move_list, flag);
                if (n < 0)
                {
                    gen_move_count = 0;// 実際は1手生成している
                    next_move.next_phase = (int)MateType.mate_intercept_drop_sup;
                    next_move = next_move_list[before];
                    index = before;
                    return true;
                }
                gen_move_count = n;
            }
        i:
            next_move.next_phase = (int)MateType.mate_intercept_weak_move;
        j:
            after = next_move_list.Count;
            if (start < after)// 複数の手を生成していたら、次回は次のインデックスに移るだけにする
            {
                if (next_move_list[after - 1].Value != next_move_list[after - 1].Value)
                {
                    gen_move_count = 0;// 実際は1手生成している
                    next_move = next_move_list[after - 1];
                    index = after - 1;
                    return true;
                }
            }
            next_move.next_phase = (int)MateType.mate_intercept_drop_sup;
        k:
            return false;
        }

        void GenIntercept(BoardTree BTree, int sq_checker, int color, ref int remaining, CheckMove move_last, ref List<NextMove> next_move_list, int flag)
        {
            int n0, n1, sq_k, min_chuai, inc, itemp, dist, ifrom, ito, direc, nsup, i;
            bool flag_promo, flag_unpromo;
            BitBoard bb_defender;
            Piece.Type pc;
            //List<NextMove> amove = new List<NextMove>();
            NextMove[] amove = new NextMove[16];
            int nmove = 0;
            n0 = n1 = 0;
            sq_k = BTree.SQ_King[color];
            bb_defender = BitBoard.Copy(BTree.BB_Occupied[color]);
            BBNotAnd(ref bb_defender, bb_defender, AbbMask[sq_k]);
            switch (ADirec[sq_k, sq_checker])
            {
                case (uint)direc_rank:
                    min_chuai = (sq_k < (int)A8 || (int)I2 < sq_k) ? 2 : 4;
                    inc = 1;
                    break;
                case (uint)direc_diag1:
                    min_chuai = 3;
                    inc = 8;
                    break;
                case (uint)direc_file:
                    itemp = (int)AiFile[sq_k];
                    min_chuai = (itemp == (int)file1 || itemp == (int)file9) ? 2 : 4;
                    inc = 9;
                    break;
                default: //direc_diag2
                    min_chuai = 3;
                    inc = 10;
                    break;
            }
            if (sq_k > sq_checker) { inc = -inc; }
            for (dist = 1, ito = sq_k + inc; ito != sq_checker; dist += 1, ito += inc)
            {
                BitBoard bb_temp0, bb_temp1, bb_atk, bb;
                bb_temp0 = BitBoard.Copy(AttacksToPiece(BTree, ito, color));
                bb_temp1 = BitBoard.Copy(AttacksToPiece(BTree, ito, color ^ 1));
                bb_atk = new BitBoard();
                bb = new BitBoard();
                BBOr(ref bb_atk, bb_temp0, bb_temp1);
                BBAnd(ref bb, bb_defender, bb_atk);
                while (BBTest(bb) != 0)
                {
                    ifrom = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                    Xor(ifrom, ref bb);
                    direc = (int)ADirec[sq_k, ifrom];
                    flag_promo = false;
                    flag_unpromo = true;
                    if (direc != 0 && IsPinnedOnKing(BTree, ifrom, direc, color) != 0)
                        continue;
                    pc = (Piece.Type)(Math.Abs(BTree.ShogiBoard[ifrom]));
                    switch (pc)
                    {
                        case Piece.Type.Pawn:
                            if (CompareFormula0(color, ito)) { flag_promo = true; flag_unpromo = false; }
                            break;
                        case Piece.Type.Lance:
                        case Piece.Type.Knight:
                            if (CompareFormula1(color, ito)) { flag_promo = true; flag_unpromo = false; }
                            else if (CompareFormula0(color, ito)) { flag_promo = true; }
                            break;
                        case Piece.Type.Silver:
                            if (CompareFormula0(color, ito) || CompareFormula0(color, ifrom)) { flag_promo = true; }
                            break;
                        case Piece.Type.Bishop:
                        case Piece.Type.Rook:
                            if (CompareFormula0(color, ito) || CompareFormula0(color, ifrom))
                            {
                                flag_promo = true; flag_unpromo = false;
                            }
                            break;
                        default:
                            break;
                    }
                    NextMove nm;
                    if (flag_promo)
                    {
                        nm = new NextMove();
                        nm.Push(ifrom, ito, pc, (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]), 1);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                    if (flag_unpromo)
                    {
                        nm = new NextMove();
                        nm.Push(ifrom, ito, pc, (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]), 1);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                }

                nsup = (ito == sq_k + inc) ? nmove + 1 : nmove;
                //if (nsup > 1 && next_move_list.Count > 1)// ToDo: next_move_list.Count > 1を付けたが、PlayOut以外ではエラーが起きていないので、元に戻すかもしれない
                if (nsup > 1)
                {
                    for (i = n0 + n1 - 1; i >= n0; i--) { next_move_list[i + nmove] = next_move_list[i]; }
                    for (i = 0; i < nmove; i++) 
                    {
                        //next_move_list[n0++] = amove[i];
                        next_move_list.Add(amove[i]);
                        n0++;
                    }
                }
                else if (nmove > 0) 
                {
                    //next_move_list[n0 + n1++] = amove[0];
                    next_move_list.Add(amove[0]);
                }

                if (nsup == 0)
                {
                    /* - tentative assumption - */
                    /* no recursive drops at non-supported square. */
                    if (flag == 2) { continue; }

                    /* -tentative assumption- */
                    /* no intercept-drop at non-supported square. */
                    if (move_last.To == sq_checker
                        && dist > min_chuai)
                    {
                        continue;
                    }
                }
                nmove = 0;

                if (nsup > 0)
                {

                    if (BTree.Hand[color, (int)Piece.Type.Rook] > 0) 
                    { 
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                    else if (BTree.Hand[color, (int)Piece.Type.Lance] > 0 && CompareFormula2(color, ito))
                    {
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Lance, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                    else if (BTree.Hand[color, (int)Piece.Type.Pawn] > 0
                        && CompareFormula2(color, ito)
                        && (BBTest(BTree.BB_Piece[color, (int)Piece.Type.Pawn]) & MaskFile[AiFile[ito]]) == 0
                        && IsMatePawnDrop(ref BTree, ito, color) == 0)
                    {
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Pawn, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }

                }
                else
                {
                    if (BTree.Hand[color, (int)Piece.Type.Pawn] > 0
                        && CompareFormula2(color, ito)
                        && (BBTest(BTree.BB_Piece[color, (int)Piece.Type.Pawn]) & MaskFile[AiFile[ito]]) == 0
                        && IsMatePawnDrop(ref BTree, ito, color) == 0)
                    {
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Pawn, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                    if (BTree.Hand[color, (int)Piece.Type.Lance] > 0 && CompareFormula2(color, ito)) 
                    {
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Lance, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                    if (BTree.Hand[color, (int)Piece.Type.Rook] > 0) 
                    {
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                }

                if (BTree.Hand[color, (int)Piece.Type.Knight] > 0 && CompareFormula3(color, ito)) 
                {
                    NextMove nm = new NextMove();
                    nm.Push(NSquare, ito, Piece.Type.Knight, 0, 0);
                    //amove.Add(nm);
                    amove[nmove++] = nm;
                }

                if (BTree.Hand[color, (int)Piece.Type.Silver] > 0)
                {
                    {
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Silver, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                }
                if (BTree.Hand[color, (int)Piece.Type.Gold] > 0)
                {
                    {
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Gold, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                }
                if (BTree.Hand[color, (int)Piece.Type.Bishop] > 0)
                {
                    {
                        NextMove nm = new NextMove();
                        nm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                        //amove.Add(nm);
                        amove[nmove++] = nm;
                    }
                }

                if (nsup > 0)
                {
                    /* - tentative assumption - */
                    /* a supported intercepter saves the king for two plies at least. */
                    if (nmove > 0 && flag == 0 && dist > min_chuai
                        && move_last.From == NSquare)
                    {
                        remaining = -1;
                        next_move_list[0] = amove[0];
                    }

                    for (i = n0 + n1 - 1; i >= n0; i--) { next_move_list[i + nmove] = next_move_list[i]; }
                    for (i = 0; i < nmove; i++)
                    {
                        //next_move_list[n0++] = amove[i];
                        next_move_list.Add(amove[i]);
                        n0++;
                    }
                }
                else
                {
                    for (i = 0; i < nmove; i++)
                    {
                        next_move_list.Add(amove[i]);
                        n1++;
                    }
                }
            }
            remaining = n0;
        }

        void GenMoveTo(BoardTree BTree, int ito, int color, ref List<NextMove> next_move_list)
        {
            BitBoard bb = BitBoard.Copy(AttacksToPiece(BTree, ito, color));
            BBNotAnd(ref bb, bb, AbbMask[BTree.SQ_King[color]]);
            while (BBTest(bb) != 0)
            {
                int ifrom = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(ifrom, ref bb);
                int direc = (int)ADirec[BTree.SQ_King[color], ifrom];
                if (direc != 0 && IsPinnedOnKing(BTree, ifrom, direc, color) != 0) { continue; }
                bool flag_promo = false;
                bool flag_unpromo = true;
                Piece.Type pc = (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]);
                switch (pc)
                {
                    case Piece.Type.Pawn:
                        if (CompareFormula0(color, ito)) { flag_promo = true; flag_unpromo = false; }
                        break;
                    case Piece.Type.Lance:
                    case Piece.Type.Knight:
                        if (CompareFormula1(color, ito)) { flag_promo = true; flag_unpromo = false; }
                        else if (CompareFormula0(color, ito)) { flag_promo = true; }
                        break;
                    case Piece.Type.Silver:
                        if (CompareFormula0(color, ito) || CompareFormula0(color, ifrom)) { flag_promo = true; }
                        break;
                    case Piece.Type.Bishop:
                    case Piece.Type.Rook:
                        if (CompareFormula0(color, ito) || CompareFormula0(color, ifrom)){ flag_promo = true; flag_unpromo = false; }
                        break;
                }
                NextMove m = new NextMove();
                if (flag_promo)
                {
                    m.Push(ifrom, ito, pc, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                    next_move_list.Add(m);
                }
                if (flag_unpromo)
                {
                    m.Push(ifrom, ito, pc, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    next_move_list.Add(m);
                }      
            }
        }

        bool CompareFormula0(int color, int sq)
        {
            if (color == 0)
            {
                if (sq < (int)A6)
                {
                    return true;
                }
                else
                {
                    return false;
                }

            }
            else
            {
                if (sq > (int)I4)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }

        bool CompareFormula1(int color, int sq)
        {
            if (color == 0)
            {
                if (sq < (int)A7)
                {
                    return true;
                }
                else
                {
                    return false;
                }

            }
            else
            {
                if (sq > (int)I3)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }

        bool CompareFormula2(int color, int sq)
        {
            if (color == 0)
            {
                if (sq < (int)A1)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                if (sq > (int)I9)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }

        bool CompareFormula3(int color, int sq)
        {
            if (color == 0)
            {
                if (sq < (int)A2)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                if (sq > (int)I8)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }
    }
}
