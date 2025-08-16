using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.AttackBitBoard;
using static Asklepios.BitOperation;
using static Asklepios.ShogiCommon;
using static Asklepios.ShogiCommon.Square;

namespace Asklepios
{
    internal class GenNoCap
    {
        public static void Generate(BoardTree BTree, ref List<Move> move_list, int color)
        {
            int ifrom, ito, flag_promo;
            int sign = 1;
            if (color == 1)
                sign = -1;
            BitBoard bb_occupied = new BitBoard();
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            BitBoard bb_empty = BitBoard.Copy(bb_occupied);
            BitBoard bb_full = BitBoard.SetFull();
            BBNot(ref bb_empty, bb_empty);
            BBAnd(ref bb_empty, bb_empty, bb_full);
            BitBoard bb_piece = new BitBoard();
            BitBoard bb_desti = new BitBoard();
            int[] index = new int[2];
            int[] num = new int[2];
            int lance_index = new int();
            ulong lance_mask = new ulong();
            if (color == 0)
            {
                index[0] = 1;
                index[1] = 2;
                num[0] = 53;
                num[1] = 80;
                lance_index = 0;
                lance_mask = 0x1ffU;
            }
            else
            {
                index[0] = 0;
                index[1] = 1;
                num[0] = 26;
                num[1] = 53;
                lance_index = 2;
                lance_mask = 0x7fc0000U;
            }


            bb_piece.p[index[0]] = BTree.BB_PawnAttacks[color].p[index[0]] & bb_empty.p[index[0]];
            bb_piece.p[index[1]] = BTree.BB_PawnAttacks[color].p[index[1]] & bb_empty.p[index[1]];
            while ((bb_piece.p[index[0]] | bb_piece.p[index[1]]) != 0)
            {
                ito = LastOneAB(bb_piece.p[index[0]], num[0], bb_piece.p[index[1]], num[1]);
                bb_piece.p[index[0]] ^= AbbMask[ito].p[index[0]];
                bb_piece.p[index[1]] ^= AbbMask[ito].p[index[1]];
                Move m = new Move();
                ifrom = ito + (sign * 9);
                m.Push(ifrom, ito, Piece.Type.Pawn, 0, 0);
                move_list.Add(m);
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Silver]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                BBAnd(ref bb_desti, bb_empty, AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom]);
                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m;
                    flag_promo = 0;
                    if (CompareFormula(color, ito) || CompareFormula(color, ifrom))
                    {
                        m = new Move();
                        m.Push(ifrom, ito, Piece.Type.Silver, 0, 1);
                    }
                    m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Silver, 0, flag_promo);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Total_Gold[color]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                BBAnd(ref bb_desti, bb_empty, AbbPieceAttacks[color, (int)Piece.Type.Gold, ifrom]);
                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    m.Push(ifrom, ito, (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]), 0, 0);
                    move_list.Add(m);
                }
            }

            ifrom = BTree.SQ_King[color];
            BBAnd(ref bb_desti, bb_empty, AbbPieceAttacks[color, (int)Piece.Type.King, ifrom]);
            while (BBTest(bb_desti) != 0)
            {
                ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                Xor(ito, ref bb_desti);
                Move m = new Move();
                m.Push(ifrom, ito, Piece.Type.King, 0, 0);
                move_list.Add(m);
            }

            bb_piece.p[index[0]] = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[index[0]];
            bb_piece.p[index[1]] = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[index[1]];
            while ((bb_piece.p[index[0]] | bb_piece.p[index[1]]) != 0)
            {
                ifrom = LastOneAB(bb_piece.p[index[0]], num[0], bb_piece.p[index[1]], num[1]);
                bb_piece.p[index[0]] ^= AbbMask[ifrom].p[index[0]];
                bb_piece.p[index[1]] ^= AbbMask[ifrom].p[index[1]];
                bb_desti = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));
                bb_desti.p[index[0]] &= bb_empty.p[index[0]];
                bb_desti.p[index[1]] &= bb_empty.p[index[1]];
                while ((bb_desti.p[index[0]] | bb_desti.p[index[1]]) != 0)
                {
                    ito = LastOneAB(bb_desti.p[index[0]], num[0], bb_desti.p[index[1]], num[1]);
                    bb_desti.p[index[0]] ^= AbbMask[ito].p[index[0]];
                    bb_desti.p[index[1]] ^= AbbMask[ito].p[index[1]];
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(m);
                }
            }


            bb_piece.p[index[0]] = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[index[0]];
            bb_piece.p[index[1]] = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[index[1]];
            while ((bb_piece.p[index[0]] | bb_piece.p[index[1]]) != 0)
            {
                ifrom = LastOneAB(bb_piece.p[index[0]], num[0], bb_piece.p[index[1]], num[1]);
                bb_piece.p[index[0]] ^= AbbMask[ifrom].p[index[0]];
                bb_piece.p[index[1]] ^= AbbMask[ifrom].p[index[1]];
                bb_desti = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));
                bb_desti.p[index[0]] &= bb_empty.p[index[0]];
                bb_desti.p[index[1]] &= bb_empty.p[index[1]];
                while ((bb_desti.p[index[0]] | bb_desti.p[index[1]]) != 0)
                {
                    ito = LastOneAB(bb_desti.p[index[0]], num[0], bb_desti.p[index[1]], num[1]);
                    bb_desti.p[index[0]] ^= AbbMask[ito].p[index[0]];
                    bb_desti.p[index[1]] ^= AbbMask[ito].p[index[1]];
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Horse]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetHorseAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, bb_empty);
                while(BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Horse, 0, 0);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetDragonAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, bb_empty);
                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Dragon, 0, 0);
                    move_list.Add(m);
                }
            }

            bb_empty.p[lance_index] &= lance_mask;
            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Lance]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                bb_desti = BitBoard.Copy(GetFileAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_desti, bb_desti, AbbPlusMinusRays[color, ifrom]);
                BBAnd(ref bb_desti, bb_desti, bb_empty);
                while(BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Lance, 0, 0);
                    move_list.Add(m);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Knight]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);
                BBAnd(ref bb_desti, bb_empty, AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom]);
                while (BBTest(bb_desti) != 0)
                {
                    ito = LastOne012(bb_desti.p[0], bb_desti.p[1], bb_desti.p[2]);
                    Xor(ito, ref bb_desti);
                    Move m = new Move();
                    m.Push(ifrom, ito, Piece.Type.Knight, 0, 0);
                    move_list.Add(m);
                }
            }
        }

        static bool CompareFormula(int color, int sq)
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
    }
}
