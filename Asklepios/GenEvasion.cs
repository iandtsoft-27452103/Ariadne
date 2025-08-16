using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.AttackBitBoard;
using static Asklepios.BitOperation;
using static Asklepios.ShogiCommon;
using static Asklepios.ShogiCommon.Square;

namespace Asklepios
{
    internal class GenEvasion
    {
        public static void Generate(ref BoardTree BTree, ref List<Move> move_list, int color)
        {
            int ifrom, ito, idirec, flag_promo;
            int sign = 1;
            if (color == 1)
                sign = -1;
            int sq_king = BTree.SQ_King[color];
            BitBoard bb_desti = new BitBoard();

            Xor(sq_king, ref BTree.BB_Occupied[color]);

            BBNotAnd(ref bb_desti, AbbPieceAttacks[color, (int)Piece.Type.King, sq_king], BTree.BB_Occupied[color]);
            while (BBTest(bb_desti) != 0)
            {
                ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                if (IsAttacked(BTree, ito, color) == 0)
                {
                    Move m = new Move();
                    m.Push(sq_king, ito, Piece.Type.King, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(m);
                }
                Xor(ito, ref bb_desti);
            }

            Xor(sq_king, ref BTree.BB_Occupied[color]);

            //両王手の場合は玉を動かすしかない
            BitBoard bb_checker = BitBoard.Copy(AttacksToPiece(BTree, sq_king, color ^ 1));
            int nchecker = PopuCount(bb_checker.p[0]) + PopuCount(bb_checker.p[1]) + PopuCount(bb_checker.p[2]);
            if (nchecker == 2) { return; }//2023.11.27 デバッグ時に誤ってコメントアウトされていたので戻した

            int sq_check = LastOne012(bb_checker.p[0], bb_checker.p[1], bb_checker.p[2]);
            BitBoard bb_inter = BitBoard.Copy(AbbObstacle[sq_king, sq_check]);

            /* move other pieces */
            //他の駒を動かす（王手をかけている駒を取る、王手をかけている長距離利き駒の間に駒を動かす）
            BitBoard bb_target = new BitBoard();
            BitBoard bb_occupied = new BitBoard();
            BBOr(ref bb_target, bb_inter, bb_checker);
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);

            BBAnd(ref bb_desti, bb_target, BTree.BB_PawnAttacks[color]);
            while (BBTest(bb_desti) != 0)
            {
                ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                Xor(ito, ref bb_desti);
                ifrom = ito + (sign * 9);
                idirec = (int)ADirec[sq_king, ifrom];
                flag_promo = 0;
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    if (CompareFormula0(color, ito)) 
                        flag_promo = 1;
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), flag_promo);
                    move_list.Add(m);
                }
            }

            BitBoard bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Lance]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetFileAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, AbbPlusMinusRays[color, ifrom]);
                BBAnd(ref bb_desti, bb_desti, bb_target);
                if (BBTest(bb_desti) == 0) { continue; }
                idirec = (int)ADirec[sq_king, ifrom];
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    if (CompareFormula0(color, ito))
                    {
                        Move m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                        move_list.Add(m);
                    }
                    if (CompareFormula2(color, ito))
                    {
                        Move m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                        move_list.Add(m);
                    }
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Knight]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                BBAnd(ref bb_desti, bb_target, AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom]);
                if (BBTest(bb_desti) == 0) { continue; }
                idirec = (int)ADirec[sq_king, ifrom];
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    do
                    {
                        ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                        Xor(ito, ref bb_desti);
                        if (CompareFormula0(color, ito))
                        {
                            Move m = new Move();
                            m.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                            move_list.Add(m);
                        }
                        if (CompareFormula2(color, ito))
                        {
                            Move m = new Move();
                            m.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                            move_list.Add(m);
                        }
                    } while (BBTest(bb_desti) != 0);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Silver]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                BBAnd(ref bb_desti, bb_target, AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom]);
                if (BBTest(bb_desti) == 0) { continue; }
                idirec = (int)ADirec[sq_king, ifrom];
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    do
                    {
                        ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                        Xor(ito, ref bb_desti);
                        Move m;
                        if (CompareFormula0(color, ifrom) || CompareFormula0(color, ito))
                        {
                            m = new Move();
                            m.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                            move_list.Add(m);
                        }
                        m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                        move_list.Add(m);
                    } while (BBTest(bb_desti) != 0);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Total_Gold[color]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                BBAnd(ref bb_desti, bb_target, AbbPieceAttacks[color, (int)Piece.Type.Gold, ifrom]);
                if (BBTest(bb_desti) == 0) { continue; }
                idirec = (int)ADirec[sq_king, ifrom];
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    do
                    {
                        ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                        Xor(ito, ref bb_desti);
                        Move m = new Move();
                        m.Push(ifrom, ito, (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]), (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                        move_list.Add(m);
                    } while (BBTest(bb_desti) != 0);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Bishop]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, bb_target);
                if (BBTest(bb_desti) == 0) { continue; }
                idirec = (int)ADirec[sq_king, ifrom];
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    do
                    {
                        ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                        Xor(ito, ref bb_desti);
                        Move m;
                        if (CompareFormula0(color, ifrom) || CompareFormula0(color, ito))
                        {
                            m = new Move();
                            m.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                            move_list.Add(m);
                        }
                        m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                        move_list.Add(m);
                    } while (BBTest(bb_desti) != 0);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Rook]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, bb_target);
                if (BBTest(bb_desti) == 0) { continue; }
                idirec = (int)ADirec[sq_king, ifrom];
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    do
                    {
                        ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                        Xor(ito, ref bb_desti);
                        Move m;
                        if (CompareFormula0(color, ifrom) || CompareFormula0(color, ito))
                        {
                            m = new Move();
                            m.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                            move_list.Add(m);
                        }
                        m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                        move_list.Add(m);
                    } while (BBTest(bb_desti) != 0);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Horse]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetHorseAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, bb_target);
                if (BBTest(bb_desti) == 0) { continue; }
                idirec = (int)ADirec[sq_king, ifrom];
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    do
                    {
                        ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                        Xor(ito, ref bb_desti);
                        Move m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Horse, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                        move_list.Add(m);
                    } while(BBTest(bb_desti) != 0);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetDragonAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, bb_target);
                if (BBTest(bb_desti) == 0) { continue; }
                idirec = (int)ADirec[sq_king, ifrom];
                if (idirec == 0 || IsPinnedOnKing(BTree, ifrom, idirec, color) == 0)
                {
                    do
                    {
                        ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                        Xor(ito, ref bb_desti);
                        Move m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Dragon, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                        move_list.Add(m);
                    } while (BBTest(bb_desti) != 0);
                }
            }

            bool hand_flag = false;
            for (int i = (int)Piece.Type.Pawn; i < NHand; i++)
            {
                if (BTree.Hand[color, i] != 0)
                {
                    hand_flag = true;
                    break;
                }
            }

            if (!hand_flag)
                return;

            ulong lbb_target_a, lbb_target_b;
            int bit_index, num;
            if (color == 0)
            {
                bb_target = BitBoard.Copy(bb_inter);
                lbb_target_a = bb_target.p[0] & 0x7fc0000U;
                lbb_target_b = bb_target.p[0] & 0x003fe00U;
                bb_target.p[0] &= 0x00001ffU;
                bb_target.p[1] &= 0x7ffffffU;
                bb_target.p[2] &= 0x7ffffffU;
                bit_index = 0;
                num = 26;
            }
            else
            {
                bb_target = BitBoard.Copy(bb_inter);
                lbb_target_a = bb_target.p[2] & 0x00001ffU;
                lbb_target_b = bb_target.p[2] & 0x003fe00U;
                bb_target.p[0] &= 0x7ffffffU;
                bb_target.p[1] &= 0x7ffffffU;
                bb_target.p[2] &= 0x7fc0000U;
                bit_index = 2;
                num = 80;
            }

            if (BTree.Hand[color, (int)Piece.Type.Pawn] != 0)
            {
                // 歩は二歩と打ち歩詰めにならないようにする
                ulong[] ais_pawn = new ulong[9];
                ulong ubb_pawn_cmp = BBTest(BTree.BB_PawnAttacks[color]);
                ais_pawn[0] = ubb_pawn_cmp & MaskFile[0];
                ais_pawn[1] = ubb_pawn_cmp & MaskFile[1];
                ais_pawn[2] = ubb_pawn_cmp & MaskFile[2];
                ais_pawn[3] = ubb_pawn_cmp & MaskFile[3];
                ais_pawn[4] = ubb_pawn_cmp & MaskFile[4];
                ais_pawn[5] = ubb_pawn_cmp & MaskFile[5];
                ais_pawn[6] = ubb_pawn_cmp & MaskFile[6];
                ais_pawn[7] = ubb_pawn_cmp & MaskFile[7];
                ais_pawn[8] = ubb_pawn_cmp & MaskFile[8];

                while (BBTest(bb_target) != 0)
                {
                    ito = LastOne012(bb_target.p[0], bb_target.p[1], bb_target.p[2]);

                    //二歩ではなく、打ち歩詰めでもなかったら歩を打つ
                    if (ais_pawn[AiFile[ito]] == 0 && IsMatePawnDrop(ref BTree, ito, color) == 0)// IsMatePawnDropはバグがあるかもしれないので注意！
                    {
                        Move m = new Move();
                        m.Push(NSquare, ito, Piece.Type.Pawn, 0, 0);
                        move_list.Add(m);
                    }

                    //その他の駒を打つ
                    for (int i = (int)Piece.Type.Lance; i < NHand; i++)
                    {
                        if (BTree.Hand[color, i] > 0)
                        {
                            Move m = new Move();
                            m.Push(NSquare, ito, (Piece.Type)i, 0, 0);
                            move_list.Add(m);
                        }
                    }
                    Xor(ito, ref bb_target);
                }

                while (lbb_target_b != 0)
                {
                    ito = LastOneN(lbb_target_b, num);

                    //二歩ではなく、打ち歩詰めでもなかったら歩を打つ
                    if (ais_pawn[AiFile[ito]] != 0 && IsMatePawnDrop(ref BTree, ito, color) == 0)// IsMatePawnDropはバグがあるかもしれないので注意！
                    {
                        Move m = new Move();
                        m.Push(NSquare, ito, Piece.Type.Pawn, 0, 0);
                        move_list.Add(m);
                    }

                    //桂以外の駒を打つ
                    for (int i = (int)Piece.Type.Lance; i < NHand; i++)
                    {
                        if (i == (int)Piece.Type.Knight)
                            continue;

                        if (BTree.Hand[color, i] > 0)
                        {
                            Move m = new Move();
                            m.Push(NSquare, ito, (Piece.Type)i, 0, 0);
                            move_list.Add(m);
                        }
                    }

                    lbb_target_b ^= AbbMask[ito].p[bit_index];
                }
            }
            else
            {
                //歩が持ち駒にない場合
                while (BBTest(bb_target) != 0)
                {
                    ito = LastOne012(bb_target.p[0], bb_target.p[1], bb_target.p[2]);

                    //その他の駒を打つ
                    for (int i = (int)Piece.Type.Lance; i < NHand; i++)
                    {
                        if (BTree.Hand[color, i] > 0)
                        {
                            Move m = new Move();
                            m.Push(NSquare, ito, (Piece.Type)i, 0, 0);
                            move_list.Add(m);
                        }
                    }
                    Xor(ito, ref bb_target);
                }

                while (lbb_target_b != 0)
                {
                    ito = LastOneN(lbb_target_b, num);

                    //桂以外の駒を打つ
                    for (int i = (int)Piece.Type.Lance; i < NHand; i++)
                    {
                        if (i == (int)Piece.Type.Knight)
                            continue;

                        if (BTree.Hand[color, i] > 0)
                        {
                            Move m = new Move();
                            m.Push(NSquare, ito, (Piece.Type)i, 0, 0);
                            move_list.Add(m);
                        }
                    }

                    lbb_target_b ^= AbbMask[ito].p[bit_index];
                }
            }

            //桂香以外の駒を打つ
            while (lbb_target_a != 0)
            {
                ito = LastOneN(lbb_target_a, num);

                //桂香以外の駒を打つ
                for (int i = (int)Piece.Type.Silver; i < NHand; i++)
                {
                    if (BTree.Hand[color, i] > 0)
                    {
                        Move m = new Move();
                        m.Push(NSquare, ito, (Piece.Type)i, 0, 0);
                        move_list.Add(m);
                    }
                }

                lbb_target_a ^= AbbMask[ito].p[bit_index];
            }
        }
        static bool CompareFormula0(int color, int sq)
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

        static bool CompareFormula1(int color, int sq)
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

        static bool CompareFormula2(int color, int sq)
        {
            if (color == 0)
            {
                if (sq >= (int)A7)
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
                if (sq <= (int)I3)
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
