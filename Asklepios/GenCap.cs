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
using System.ComponentModel;

namespace Asklepios
{
    internal class GenCap
    {
        public static void Generate(BoardTree BTree, ref List<Move> move_list, int color)
        {
            int ifrom, ito, flag_promo;
            int sign = 1;
            if (color == 1)
                sign = -1;

            BitBoard bb_piece;
            BitBoard bb_occupied = new BitBoard();
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            BitBoard bb_capture = BitBoard.Copy(BTree.BB_Occupied[color ^ 1]);
            BitBoard bb_movable = new BitBoard();
            //BitBoard bb_mask = BitBoard.SetFull();
            BBNot(ref bb_movable, BTree.BB_Occupied[color]);
            BitBoard bb_desti = new BitBoard();
            int[] index = new int[3];
            if (color == 0)
            {
                index[0] = 0;
                index[1] = 1;
                index[2] = 2;
            }
            else
            {
                index[0] = 2;
                index[1] = 1;
                index[2] = 0;
            }
            bb_desti.p[index[0]] = BTree.BB_PawnAttacks[color].p[index[0]] & bb_movable.p[index[0]];
            bb_desti.p[index[1]] = BTree.BB_PawnAttacks[color].p[index[1]] & bb_capture.p[index[1]];
            bb_desti.p[index[2]] = BTree.BB_PawnAttacks[color].p[index[2]] & bb_capture.p[index[2]];
            while (BBTest(bb_desti) != 0)
            {
                ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                Xor(ito, ref bb_desti);
                Move m = new Move();
                ifrom = ito + (sign * 9);
                flag_promo = 0;
                if (CompareFormula0(color, ito))
                    flag_promo = 1;
                m.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), flag_promo);
                move_list.Add(m);
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Silver]);
            while(BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                BBAnd(ref bb_desti, bb_capture, AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom]);
                while(BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m;
                    if (CompareFormula0(color, ito) || CompareFormula0(color, ifrom))
                    {
                        m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                        move_list.Add(m);
                    }
                    m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Total_Gold[color]);
            while(BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                BBAnd(ref bb_desti, bb_capture, AbbPieceAttacks[color, (int)Piece.Type.Gold, ifrom]);
                while(BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    m.Push(ifrom, ito, (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]), (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(m);
                }
            }

            ifrom = BTree.SQ_King[color];
            BBAnd(ref bb_desti, bb_capture, AbbPieceAttacks[color, (int)Piece.Type.King, ifrom]);
            while(BBTest(bb_desti) != 0)
            {
                ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                Xor(ito, ref bb_desti);
                Move m = new Move();
                m.Push(ifrom, ito, Piece.Type.King, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                move_list.Add(m);
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Bishop]);
            while(BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));

                bb_desti.p[index[0]] &= bb_movable.p[index[0]];
                if (CompareFormula0(color, ifrom))
                {
                    bb_desti.p[index[1]] &= bb_movable.p[index[1]];
                    bb_desti.p[index[2]] &= bb_movable.p[index[2]];
                }
                else
                {
                    bb_desti.p[index[1]] &= bb_capture.p[index[1]];
                    bb_desti.p[index[2]] &= bb_capture.p[index[2]];
                }

                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    flag_promo = 0;
                    if (CompareFormula0(color, ito) || CompareFormula0(color, ifrom))
                        flag_promo = 1;
                    m.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), flag_promo);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Rook]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));

                bb_desti.p[0] &= bb_movable.p[index[0]];
                if (CompareFormula0(color, ifrom))
                {
                    bb_desti.p[index[1]] &= bb_movable.p[index[1]];
                    bb_desti.p[index[2]] &= bb_movable.p[index[2]];
                }
                else
                {
                    bb_desti.p[index[1]] &= bb_capture.p[index[1]];
                    bb_desti.p[index[2]] &= bb_capture.p[index[2]];
                }

                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    flag_promo = 0;
                    if (CompareFormula0(color, ito) || CompareFormula0(color, ifrom))
                        flag_promo = 1;
                    m.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), flag_promo);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Horse]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetHorseAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, bb_capture);
                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Horse, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetDragonAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, bb_capture);
                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Dragon, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Lance]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetFileAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, AbbPlusMinusRays[color, ifrom]);
                bb_desti.p[index[0]] &= bb_movable.p[index[0]];
                bb_desti.p[index[1]] &= bb_capture.p[index[1]];
                bb_desti.p[index[2]] &= bb_capture.p[index[2]];
                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    if (CompareFormula1(color, ito))
                    {
                        m.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                        move_list.Add(m);
                    }
                    else if(CompareFormula2(color, ito) && CompareFormula0(color, ito))
                    {
                        m.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                        move_list.Add(m);
                        m = new Move();
                        if ((Piece.Type)Math.Abs(BTree.ShogiBoard[ito]) != 0)
                        {
                            m.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                            move_list.Add(m);
                        }
                    }
                    else
                    {
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
                bb_desti = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom]);
                bb_desti.p[index[0]] &= bb_movable.p[index[0]];
                bb_desti.p[index[1]] &= bb_capture.p[index[1]];
                bb_desti.p[index[2]] &= bb_capture.p[index[2]];
                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    if (CompareFormula1(color, ito))
                    {
                        m.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                        move_list.Add(m);
                    }
                    else if (CompareFormula2(color, ito) && CompareFormula0(color, ito))
                    {
                        m.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                        move_list.Add(m);
                        m = new Move();
                        if ((Piece.Type)Math.Abs(BTree.ShogiBoard[ito]) != 0)
                        {
                            m.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                            move_list.Add(m);
                        }
                    }
                    else
                    {
                        m.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                        move_list.Add(m);
                    }
                }
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
