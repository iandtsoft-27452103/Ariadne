using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.ShogiCommon.Direction;
using static Asklepios.ShogiCommon.Square;
using static Asklepios.ShogiCommon.File;
using static Asklepios.ShogiCommon.Rank;
using static Asklepios.ShogiCommon;
using static Asklepios.AttackBitBoard;
using static Asklepios.BitOperation;
using static Asklepios.Board;
using static Asklepios.Check;
using System.Numerics;

namespace Asklepios
{
    internal class Mate1Ply
    {
        // 1手詰め
        public static int IsMateIn1Ply(ref BoardTree BTree, ref Move mate_move, int color)
        {
            //if (IsAttacked(BTree, BTree.SQ_King[color], color) != 0)
                //return 0;

            BitBoard bb_occupied = new BitBoard();
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            int value_nfile = NFile;
            if (color == 1) { value_nfile = -value_nfile; }

            /*  Drops  */
            int ito;
            BitBoard bb_drop = BitBoard.Copy(bb_occupied);
            BitBoard bb = new BitBoard();
            BitBoard bb_attacks = new BitBoard();
            BitBoard bb_check_pro = new BitBoard();
            BBNot(ref bb_drop, bb_drop);

            if (BTree.Hand[color, (int)Piece.Type.Rook] > 0)
            { 
                BBAnd(ref bb, AbbPieceAttacks[color, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]]);
                BBAnd(ref bb, bb, bb_drop);
                while (BBTest(bb) != 0)
                {
                    ito = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                    Xor(ito, ref bb);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    BBOr(ref bb_attacks, AbbFileAttacks[ito, 0], AbbRankAttacks[ito, 0]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    mate_move.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    return 1;
                }

            }
            else if (BTree.Hand[color, (int)Piece.Type.Lance] > 0 && CompareFormula0(color, BTree.SQ_King[color ^ 1]))
            {

                ito = BTree.SQ_King[color ^ 1] + value_nfile;
                if (BTree.ShogiBoard[ito] == 0 && IsAttacked(BTree, ito, color ^ 1) != 0)
                {
                    bb_attacks = BitBoard.Copy(AbbFileAttacks[ito, 0]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) == 0
                        && CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) == 0)
                    {
                        mate_move.Push(NSquare, ito, Piece.Type.Lance, 0, 0);
                        return 1;
                    }
                }
            }

            if (BTree.Hand[color, (int)Piece.Type.Bishop] > 0)
            {

                BBAnd(ref bb, AbbPieceAttacks[color, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]], 
                    AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]]);
                BBAnd(ref bb, bb, bb_drop);
                while (BBTest(bb) != 0)
                {
                    ito = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                    Xor(ito, ref bb);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    BBOr(ref bb_attacks, AbbDiag1Attacks[ito, 0], AbbDiag2Attacks[ito, 0]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    mate_move.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    return 1;
                }
            }

            if (BTree.Hand[color, (int)Piece.Type.Gold] > 0)
            {

                if (BTree.Hand[color, (int)Piece.Type.Rook] > 0)
                {
                    BBAnd(ref bb, AbbPieceAttacks[color, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]],
                        AbbPieceAttacks[color, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]]);
                    BBNotAnd(ref bb, bb_drop, bb);
                    BBAnd(ref bb, bb, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]]);
                }
                else { BBAnd(ref bb, bb_drop, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]]); }

                while (BBTest(bb) != 0)
                {
                    ito = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                    Xor(ito, ref bb);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color , (int)Piece.Type.Gold, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    mate_move.Push(NSquare, ito, Piece.Type.Gold, 0, 0);
                    return 1;
                }
            }

            if (BTree.Hand[color, (int)Piece.Type.Silver] > 0)
            {

                if (BTree.Hand[color, (int)Piece.Type.Gold] > 0)
                {
                    if (BTree.Hand[color, (int)Piece.Type.Bishop] > 0) { goto b_silver_drop_end; }
                    BBNotAnd(ref bb,
                        AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]],
                        AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]]);
                    BBAnd(ref bb, bb, bb_drop);
                }
                else
                {
                    BBAnd(ref bb, bb_drop, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]]);
                    if (BTree.Hand[color , (int)Piece.Type.Bishop] > 0)
                    {
                        BBAnd(ref bb, bb, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]]);
                    }
                }

                while (BBTest(bb) != 0)
                {
                    ito = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                    Xor(ito, ref bb);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Silver, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    mate_move.Push(NSquare, ito, Piece.Type.Silver, 0, 0);
                    return 1;
                }
            }

        b_silver_drop_end:

            if (BTree.Hand[color, (int)Piece.Type.Knight] > 0)
            {
                BBAnd(ref bb, bb_drop, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, BTree.SQ_King[color ^ 1]]);
                while (BBTest(bb) != 0)
                {
                    ito = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                    Xor(ito, ref bb);

                    bb_attacks = new BitBoard();
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    mate_move.Push(NSquare, ito, Piece.Type.Knight, 0, 0);
                    return 1;
                }
            }

            /*  Moves  */
            int ifrom, flag_promo;
            int[] bb_index = new int[3];
            int[] sq_index = new int[3];

            if (color == 0)
            {
                bb_index[0] = 0;
                bb_index[1] = 1;
                bb_index[2] = 2;
                sq_index[0] = 26;
                sq_index[1] = 53;
                sq_index[2] = 80;
            }
            else
            {
                bb_index[0] = 2;
                bb_index[1] = 1;
                bb_index[2] = 0;
                sq_index[0] = 80;
                sq_index[1] = 53;
                sq_index[2] = 26;
            }

            BitBoard bb_move = new BitBoard();
            BitBoard bb_check = new BitBoard();
            BBNot(ref bb_move, BTree.BB_Occupied[color]);

            bb = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            while (BBTest(bb) != 0)
            {
                ifrom = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(ifrom, ref bb);

                bb_attacks = BitBoard.Copy(GetDragonAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_check, bb_move, bb_attacks);
                BBAnd(ref bb_check, bb_check, AbbPieceAttacks[color, (int)Piece.Type.King, BTree.SQ_King[color ^ 1]]);
                if (BBTest(bb_check) == 0) { continue; }

                Xor(ifrom, ref BTree.BB_HDK[color]);
                Xor(ifrom, ref BTree.BB_RD[color]);
                Xor(ifrom, ref BTree.BB_Occupied[color]);

                do
                {
                    ito = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    Xor(ito, ref bb_check);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    if (((int)ADirec[BTree.SQ_King[color ^ 1], ito] & (int)flag_cross) != 0)
                    {
                        BBOr(ref bb_attacks, AbbFileAttacks[ito, 0], AbbRankAttacks[ito, 0]);
                        BBOr(ref bb_attacks, bb_attacks, AbbPieceAttacks[color, (int)Piece.Type.King, ito]);
                    }
                    else { bb_attacks = BitBoard.Copy(GetDragonAttacks(bb_occupied, ito)); }

                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    Xor(ifrom, ref BTree.BB_Occupied[color]);
                    Xor(ifrom, ref BTree.BB_RD[color]);
                    Xor(ifrom, ref BTree.BB_HDK[color]);
                    mate_move.Push(ifrom, ito, Piece.Type.Dragon, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    return 1;
                } while (BBTest(bb_check) != 0);

                Xor(ifrom, ref BTree.BB_Occupied[color]);
                Xor(ifrom, ref BTree.BB_RD[color]);
                Xor(ifrom, ref BTree.BB_HDK[color]);
            }

            bb.p[bb_index[0]] = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[bb_index[0]];
            while (bb.p[bb_index[0]] != 0)
            {
                ifrom = LastOneN(bb.p[bb_index[0]], sq_index[0]);
                bb.p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];

                bb_attacks = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_check, bb_move, bb_attacks);
                BBAnd(ref bb_check, bb_check, AbbPieceAttacks[color, (int)Piece.Type.King, BTree.SQ_King[color ^ 1]]);
                if (BBTest(bb_check) == 0) { continue; }

                BTree.BB_RD[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];

                do
                {
                    ito = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    Xor(ito, ref bb_check);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    if (((int)ADirec[BTree.SQ_King[color ^ 1], ito] & (int)flag_cross) != 0)
                    {
                        BBOr(ref bb_attacks, AbbFileAttacks[ito, 0], AbbRankAttacks[ito, 0]);
                        BBOr(ref bb_attacks, bb_attacks, AbbPieceAttacks[color, (int)Piece.Type.King, ito]);
                    }
                    else { bb_attacks = BitBoard.Copy(GetDragonAttacks(bb_occupied, ito)); }

                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    BTree.BB_RD[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    mate_move.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                    return 1;
                } while (BBTest(bb_check) != 0);

                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_RD[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
            }

            bb.p[bb_index[1]] = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[bb_index[1]];
            bb.p[bb_index[2]] = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[bb_index[2]];
            while ((bb.p[bb_index[1]] | bb.p[bb_index[2]]) != 0)
            {
                ifrom = LastOneAB(bb.p[bb_index[1]], sq_index[1], bb.p[bb_index[2]], sq_index[2]);
                bb.p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                bb.p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];

                bb_attacks = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_check, bb_move, bb_attacks);
                bb_check.p[bb_index[0]] &= AbbPieceAttacks[color, (int)Piece.Type.King, BTree.SQ_King[color ^ 1]].p[bb_index[0]];
                bb_check.p[bb_index[1]] &= AbbPieceAttacks[color, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[1]];
                bb_check.p[bb_index[2]] &= AbbPieceAttacks[color, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[2]];
                bb_check.p[bb_index[1]] &= AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[1]];
                bb_check.p[bb_index[2]] &= AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[2]];
                if (BBTest(bb_check) == 0) { continue; }

                BTree.BB_RD[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_RD[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];

                do
                {
                    ito = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    Xor(ito, ref bb_check);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    if (CompareFormula2(color, ito))
                    {
                        if (((int)ADirec[BTree.SQ_King[color ^ 1], ito] & (int)flag_cross) != 0)
                        {
                            BBOr(ref bb_attacks, AbbFileAttacks[ito, 0], AbbRankAttacks[ito, 0]);
                            bb_attacks.p[bb_index[0]] |= AbbPieceAttacks[color, (int)Piece.Type.King, ito].p[bb_index[0]];
                            bb_attacks.p[bb_index[1]] |= AbbPieceAttacks[color, (int)Piece.Type.King, ito].p[bb_index[1]];
                        }
                        else { bb_attacks = BitBoard.Copy(GetDragonAttacks(bb_occupied, ito)); }
                    }
                    else
                    {
                        BBOr(ref bb_attacks, AbbFileAttacks[ito, 0], AbbRankAttacks[ito, 0]);
                    }
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    BTree.BB_RD[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_RD[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    flag_promo = 0;
                    if (CompareFormula1(color, ito))
                        flag_promo = 1;
                    mate_move.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), flag_promo);
                    return 1;
                } while (BBTest(bb_check) != 0);

                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                BTree.BB_RD[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_RD[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
            }

            bb = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Horse]);
            while (BBTest(bb) != 0)
            {
                ifrom = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(ifrom, ref bb);

                bb_attacks = BitBoard.Copy(GetHorseAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_check, bb_move, bb_attacks);
                BBAnd(ref bb_check, bb_check, AbbPieceAttacks[color, (int)Piece.Type.King, BTree.SQ_King[color ^ 1]]);
                if (BBTest(bb_check) == 0) { continue; }

                Xor(ifrom, ref BTree.BB_HDK[color]);
                Xor(ifrom, ref BTree.BB_BH[color]);
                Xor(ifrom, ref BTree.BB_Occupied[color]);

                do
                {
                    ito = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    Xor(ito, ref bb_check);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    BBOr(ref bb_attacks, AbbDiag1Attacks[ito, 0], AbbDiag2Attacks[ito, 0]);
                    BBOr(ref bb_attacks, bb_attacks, AbbPieceAttacks[color, (int)Piece.Type.King, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    Xor(ifrom, ref BTree.BB_Occupied[color]);
                    Xor(ifrom, ref BTree.BB_BH[color]);
                    Xor(ifrom, ref BTree.BB_HDK[color]);
                    mate_move.Push(ifrom, ito, Piece.Type.Horse, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    return 1;
                } while (BBTest(bb_check) != 0);

                Xor(ifrom, ref BTree.BB_Occupied[color]);
                Xor(ifrom, ref BTree.BB_BH[color]);
                Xor(ifrom, ref BTree.BB_HDK[color]);
            }

            bb.p[bb_index[0]] = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[bb_index[0]];
            while (bb.p[bb_index[0]] != 0)
            {
                ifrom = LastOneN(bb.p[bb_index[0]], sq_index[0]);
                bb.p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];

                bb_attacks = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_check, bb_move, bb_attacks);
                BBAnd(ref bb_check, bb_check, AbbPieceAttacks[color, (int)Piece.Type.King, BTree.SQ_King[color ^ 1]]);
                if (BBTest(bb_check) == 0) { continue; }

                BTree.BB_BH[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];

                do
                {
                    ito = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    Xor(ito, ref bb_check);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    BBOr(ref bb_attacks, AbbDiag1Attacks[ito, 0], AbbDiag2Attacks[ito, 0]);
                    BBOr(ref bb_attacks, bb_attacks, AbbPieceAttacks[color, (int)Piece.Type.King, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    BTree.BB_BH[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    mate_move.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);
                    return 1;
                } while (BBTest(bb_check) != 0);

                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_BH[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
            }

            bb.p[bb_index[1]] = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[bb_index[1]];
            bb.p[bb_index[2]] = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[bb_index[2]];
            while ((bb.p[bb_index[1]] | bb.p[bb_index[2]]) != 0)
            {
                ifrom = LastOneAB(bb.p[bb_index[1]], sq_index[1], bb.p[bb_index[2]], sq_index[2]);
                bb.p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                bb.p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];

                bb_attacks = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_check, bb_move, bb_attacks);
                bb_check.p[bb_index[0]] &= AbbPieceAttacks[color, (int)Piece.Type.King, BTree.SQ_King[color ^ 1]].p[bb_index[0]];
                bb_check.p[bb_index[1]] &= AbbPieceAttacks[color, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[1]];
                bb_check.p[bb_index[2]] &= AbbPieceAttacks[color, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[2]];
                bb_check.p[bb_index[1]] &= AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[1]];
                bb_check.p[bb_index[2]] &= AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[2]];
                if (BBTest(bb_check) == 0) { continue; }

                BTree.BB_BH[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_BH[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];

                do
                {
                    ito = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    Xor(ito, ref bb_check);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    BBOr(ref bb_attacks, AbbDiag1Attacks[ito, 0], AbbDiag2Attacks[ito, 0]);
                    if (ito <= (int)I7)
                    {
                        bb_attacks.p[bb_index[0]] |= AbbPieceAttacks[color, (int)Piece.Type.King, ito].p[bb_index[0]];
                        bb_attacks.p[bb_index[1]] |= AbbPieceAttacks[color, (int)Piece.Type.King, ito].p[bb_index[1]];
                    }
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    BTree.BB_BH[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_BH[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    flag_promo = 0;
                    if (CompareFormula1(color, ito))
                        flag_promo = 1;
                    mate_move.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), flag_promo);
                    return 1;
                } while (BBTest(bb_check) != 0);

                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                BTree.BB_BH[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_BH[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
            }

            BBAnd(ref bb, BTree.BB_Total_Gold[color], ChkTbl[color, BTree.SQ_King[color ^ 1]].gold);
            while (BBTest(bb) != 0)
            {
                ifrom = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(ifrom, ref bb);

                BBAnd(ref bb_check, bb_move, AbbPieceAttacks[color, (int)Piece.Type.Gold, ifrom]);
                BBAnd(ref bb_check, bb_check, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]]);
                if (BBTest(bb_check) == 0) { continue; }

                Xor(ifrom, ref BTree.BB_Total_Gold[color]);
                Xor(ifrom, ref BTree.BB_Occupied[color]);

                do
                {
                    ito = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    Xor(ito, ref bb_check);

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Gold, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    Xor(ifrom, ref BTree.BB_Occupied[color]);
                    Xor(ifrom, ref BTree.BB_Total_Gold[color]);
                    mate_move.Push(ifrom, ito, (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]), (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);
                    return 1;
                } while (BBTest(bb_check) != 0);

                Xor(ifrom, ref BTree.BB_Occupied[color]);
                Xor(ifrom, ref BTree.BB_Total_Gold[color]);
            }

            BBAnd(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Silver], ChkTbl[color, BTree.SQ_King[color ^ 1]].silver);
            while (bb.p[bb_index[0]] != 0)
            {
                ifrom = LastOneN(bb.p[bb_index[0]], sq_index[0]);
                bb.p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];

                bb_check_pro.p[bb_index[0]] = bb_move.p[bb_index[0]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[0]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[0]];
                bb_check_pro.p[bb_index[1]] = bb_move.p[bb_index[1]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[1]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[1]];

                bb_check.p[bb_index[0]] = bb_move.p[bb_index[0]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[0]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[0]]
                    & ~AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[0]];
                bb_check.p[bb_index[1]] = bb_move.p[bb_index[1]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[1]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[1]]
                    & ~AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[1]];

                if ((bb_check_pro.p[bb_index[0]] | bb_check_pro.p[bb_index[1]]
                    | bb_check.p[bb_index[0]] | bb_check.p[bb_index[1]]) == 0)
                {
                    continue;
                }

                BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];

                while ((bb_check_pro.p[bb_index[0]] | bb_check_pro.p[bb_index[1]]) != 0)
                {
                    ito = LastOneAB(bb_check_pro.p[bb_index[0]], sq_index[0], bb_check_pro.p[bb_index[1]], sq_index[1]);
                    bb_check_pro.p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];
                    bb_check_pro.p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Gold, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    mate_move.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1);// 成り
                    return 1;
                }

                while ((bb_check.p[bb_index[0]] | bb_check.p[bb_index[1]]) != 0)
                {
                    ito = LastOneAB(bb_check.p[bb_index[0]], sq_index[0], bb_check.p[bb_index[1]], sq_index[1]);
                    bb_check.p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];
                    bb_check.p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Silver, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    mate_move.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0);// 不成
                    return 1;
                }

                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
            }

            ulong temp;
            if (color == 0) { temp = 0x7fc0000U; }
            else { temp = 0x1ffU; }

            ulong ubb = bb.p[bb_index[1]] & temp;
            while (ubb != 0)
            {
                ifrom = LastOneN(ubb, sq_index[1]);
                ubb ^= AbbMask[ifrom].p[bb_index[1]];

                bb_check_pro.p[bb_index[0]] = bb_move.p[bb_index[0]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[0]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[0]];

                bb_check.p[bb_index[0]] = bb_move.p[bb_index[0]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[0]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[0]]
                    & ~AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[0]];
                bb_check.p[bb_index[1]] = bb_move.p[bb_index[1]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[1]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[1]];

                if ((bb_check_pro.p[bb_index[0]] | bb_check.p[bb_index[0]] | bb_check.p[bb_index[1]]) == 0) { continue; }

                BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];

                while (bb_check_pro.p[bb_index[0]] != 0)
                {
                    ito = LastOneN(bb_check_pro.p[bb_index[0]], sq_index[0]);
                    bb_check_pro.p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Gold, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    mate_move.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1); // 成り
                    return 1;
                }

                while ((bb_check.p[bb_index[0]] | bb_check.p[bb_index[1]]) != 0)
                {
                    ito = LastOneAB(bb_check.p[bb_index[0]], sq_index[0], bb_check.p[bb_index[1]], sq_index[1]);
                    bb_check.p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];
                    bb_check.p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Silver, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    mate_move.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0); // 不成
                    return 1;
                }

                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
            }

            if (color == 0) { temp = 0x003ffffU; }
            else { temp = 0x7fffe00U; }

            bb.p[bb_index[1]] &= temp;
            while ((bb.p[bb_index[1]] | bb.p[bb_index[2]]) != 0)
            {
                ifrom = LastOneAB(bb.p[bb_index[1]], sq_index[1], bb.p[bb_index[2]], sq_index[2]);
                bb.p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                bb.p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];

                bb_check.p[bb_index[1]] = bb_move.p[bb_index[1]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[1]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[1]];
                bb_check.p[bb_index[2]] = bb_move.p[bb_index[2]] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[bb_index[2]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, BTree.SQ_King[color ^ 1]].p[bb_index[2]];
                if ((bb_check.p[bb_index[1]] | bb_check.p[bb_index[2]]) == 0) { continue; }

                BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];

                do
                {
                    ito = LastOneAB(bb_check.p[bb_index[1]], sq_index[1], bb_check.p[bb_index[2]], sq_index[2]);
                    bb_check.p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];
                    bb_check.p[bb_index[2]] ^= AbbMask[ito].p[bb_index[2]];

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Silver, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    mate_move.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0); // 不成
                    return 1;
                } while ((bb_check.p[bb_index[1]] | bb_check.p[bb_index[2]]) != 0);

                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_Piece[color, (int)Piece.Type.Silver].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
            }

            BBAnd(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Knight], ChkTbl[color, BTree.SQ_King[color ^ 1]].knight);
            while (BBTest(bb) != 0)
            {
                ifrom = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(ifrom, ref bb);

                bb_check.p[bb_index[0]] = bb_move.p[bb_index[0]] & AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom].p[bb_index[0]]
                    & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[0]];

                if (bb_check.p[bb_index[0]] != 0)
                {
                    BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];

                    do
                    {
                        ito = LastOneN(bb_check.p[bb_index[0]], sq_index[0]);
                        bb_check.p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];

                        if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                        bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Gold, ito]);
                        if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                        if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                        else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                        if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                        BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                        BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                        BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                        BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                        mate_move.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1); // 成り
                        return 1;
                    } while (bb_check.p[bb_index[0]] != 0);

                    BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                    BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                }
                else
                {

                    BBAnd(ref bb_check, bb_move, AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom]);
                    BBAnd(ref bb_check, bb_check, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, BTree.SQ_King[color ^ 1]]);

                    if (BBTest(bb_check) != 0)
                    {
                        BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                        BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                        BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                        BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];

                        do
                        {
                            ito = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                            Xor(ito, ref bb_check);

                            BBIni(ref bb_attacks);
                            if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                            if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                            else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                            if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                            BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                            BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                            BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                            BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                            mate_move.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0); // 不成
                            return 1;
                        } while (BBTest(bb_check)　!= 0);

                        BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                        BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                        BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                        BTree.BB_Piece[color, (int)Piece.Type.Knight].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    }
                }
            }

            BBAnd(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Lance], ChkTbl[color, BTree.SQ_King[color ^ 1]].lance);
            while (BBTest(bb) != 0)
            {
                ifrom = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(ifrom, ref bb);

                bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_attacks, bb_attacks, AbbPlusMinusRays[color, ifrom]);
                BBAnd(ref bb_attacks, bb_attacks, bb_move);

                BBAnd(ref bb_check, bb_attacks, AbbMask[BTree.SQ_King[color ^ 1] + value_nfile]);
                bb_check_pro.p[bb_index[0]] = bb_attacks.p[bb_index[0]] & AbbPieceAttacks[color ^ 1,(int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[0]];

                if ((bb_check_pro.p[bb_index[0]] | bb_check.p[bb_index[0]]
                    | bb_check.p[bb_index[1]] | bb_check.p[bb_index[2]]) == 0)
                {
                    continue;
                }

                Xor(ifrom, ref BTree.BB_Piece[color, (int)Piece.Type.Lance]);
                Xor(ifrom, ref BTree.BB_Occupied[color]);

                if (color == 0) { temp = 0x1ffU; }
                else { temp = 0x7fc0000U; }

                bb_check.p[bb_index[0]] &= temp;
                if (BBTest(bb_check) != 0)
                {

                    ito = BTree.SQ_King[color ^ 1] + value_nfile;
                    if (IsAttacked(BTree, ito, color ^ 1) == 0)
                    {
                        bb_check.p[bb_index[0]] &= ~AbbMask[ito].p[bb_index[0]];
                        goto b_lance_next;
                    }
                    BitBoard bb_temp = BitBoard.Copy(AbbFileAttacks[ito, 0]);
                    if (CanKingEscape(ref BTree, ito, ref bb_temp, color ^ 1) != 0) { goto b_lance_next; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { goto b_lance_next; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { goto b_lance_next; }

                    Xor(ifrom, ref BTree.BB_Occupied[color]);
                    Xor(ifrom, ref BTree.BB_Piece[color, (int)Piece.Type.Lance]);
                    mate_move.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0); // 不成
                    return 1;
                }

            b_lance_next:
                while (bb_check_pro.p[bb_index[0]] != 0)
                {
                    ito = LastOneN(bb_check_pro.p[bb_index[0]], sq_index[0]);
                    bb_check_pro.p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { continue; }

                    bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Gold, ito]);
                    if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                    else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { continue; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { continue; }

                    Xor(ifrom, ref BTree.BB_Occupied[color]);
                    Xor(ifrom, ref BTree.BB_Piece[color, (int)Piece.Type.Lance]);
                    mate_move.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1); // 成り
                    return 1;
                }

                Xor(ifrom, ref BTree.BB_Occupied[color]);
                Xor(ifrom, ref BTree.BB_Piece[color, (int)Piece.Type.Lance]);
            }

            bb_check.p[bb_index[0]] = bb_move.p[bb_index[0]] & BTree.BB_PawnAttacks[color].p[bb_index[0]]
                & AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, BTree.SQ_King[color ^ 1]].p[bb_index[0]];
            while (bb_check.p[bb_index[0]] != 0)
            {
                ito = LastOneN(bb_check.p[bb_index[0]], sq_index[0]);
                ifrom = ito + value_nfile;
                bb_check.p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];

                BTree.BB_PawnAttacks[color].p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];
                BTree.BB_PawnAttacks[color].p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];
                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];

                if (IsAttacked(BTree, ito, color ^ 1) == 0) { goto b_pawn_pro_next; }
                bb_attacks = BitBoard.Copy(AbbPieceAttacks[color, (int)Piece.Type.Gold, ito]);
                if (CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { goto b_pawn_pro_next; }
                if (IsDiscoverKing(BTree, ifrom, ito, color ^ 1) != 0) { }
                else if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { goto b_pawn_pro_next; }
                if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { goto b_pawn_pro_next; }

                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_PawnAttacks[color].p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];
                BTree.BB_PawnAttacks[color].p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];
                mate_move.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 1); // 成り
                return 1;

            b_pawn_pro_next:
                BTree.BB_Occupied[color].p[bb_index[0]] ^= AbbMask[ifrom].p[bb_index[0]];
                BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                BTree.BB_PawnAttacks[color].p[bb_index[0]] ^= AbbMask[ito].p[bb_index[0]];
                BTree.BB_PawnAttacks[color].p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];
            }

            //if (BTree.SQ_King[color ^ 1] >= (int)A7 && BTree.SQ_King[color ^ 1] <= (int)I3)
            if (BTree.SQ_King[color ^ 1] >= (int)A7 && BTree.SQ_King[color ^ 1] <= (int)I3)
            {
                ito = BTree.SQ_King[color ^ 1] + value_nfile;
                ifrom = ito + value_nfile;
                if (CompareFormula3(color, BTree.ShogiBoard[ifrom], BTree.ShogiBoard[ito]))
                {

                    BTree.BB_PawnAttacks[color].p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];
                    BTree.BB_PawnAttacks[color].p[bb_index[2]] ^= AbbMask[ito].p[bb_index[2]];
                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];

                    if (IsAttacked(BTree, ito, color ^ 1) == 0) { goto b_pawn_end; }
                    BBIni(ref bb_attacks);
                    if(CanKingEscape(ref BTree, ito, ref bb_attacks, color ^ 1) != 0) { goto b_pawn_end; }
                    if (CanPieceCapture(BTree, ito, bb_occupied, color ^ 1) != 0) { goto b_pawn_end; }
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 0) { goto b_pawn_end; }

                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    BTree.BB_PawnAttacks[color].p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];
                    BTree.BB_PawnAttacks[color].p[bb_index[2]] ^= AbbMask[ito].p[bb_index[2]];
                    mate_move.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]), 0); // 不成
                    return 1;

                b_pawn_end:
                    BTree.BB_Occupied[color].p[bb_index[1]] ^= AbbMask[ifrom].p[bb_index[1]];
                    BTree.BB_Occupied[color].p[bb_index[2]] ^= AbbMask[ifrom].p[bb_index[2]];
                    BTree.BB_PawnAttacks[color].p[bb_index[1]] ^= AbbMask[ito].p[bb_index[1]];
                    BTree.BB_PawnAttacks[color].p[bb_index[2]] ^= AbbMask[ito].p[bb_index[2]];
                }
            }

            return 0;
        }

        public static int CanPieceCapture(BoardTree BTree, int ito, BitBoard bb_occupied, int color)
        {
            int ifrom;

            if (color == 0)
            {
                ifrom = ito + NFile;
                if (ito <= (int)I2 && BTree.ShogiBoard[ifrom] == (int)Piece.Type.Pawn)
                {
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 1)
                        return 1;
                }
            }
            else
            {
                ifrom = ito - NFile;
                if (ito >= (int)A8 && BTree.ShogiBoard[ifrom] == -(int)Piece.Type.Pawn)
                {
                    if (IsDiscoverKing(BTree, ifrom, ito, color) != 1)
                        return 1;
                }
            }

            BitBoard bb_sum = new BitBoard();
            BBAnd(ref bb_sum, BTree.BB_Piece[color, (int)Piece.Type.Knight], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, ito]);

            BBAndOr(ref bb_sum, BTree.BB_Piece[color, (int)Piece.Type.Silver], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, ito]);
            BBAndOr(ref bb_sum, BTree.BB_Total_Gold[color], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, ito]);

            BitBoard bb = new BitBoard();
            BBOr(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Horse], BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            BBAndOr(ref bb_sum, bb, AbbPieceAttacks[color, (int)Piece.Type.King, ito]);

            bb = BitBoard.Copy(GetBishopAttacks(bb_occupied, ito));
            BBAndOr(ref bb_sum, BTree.BB_BH[color], bb);

            BitBoard bb_attacks = BitBoard.Copy(GetRankAttacks(bb_occupied, ito));
            BBAndOr(ref bb_sum, BTree.BB_RD[color], bb_attacks);
            bb = BitBoard.Copy(BTree.BB_RD[color]);
            BBAndOr(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Lance], AbbPlusMinusRays[color ^ 1, ito]);
            bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, ito));
            BBAndOr(ref bb_sum, bb, bb_attacks);

            while (BBTest(bb_sum) != 0)
            {
                ifrom = LastOne012(bb_sum.p[0], bb_sum.p[1], bb_sum.p[2]);
                Xor(ifrom, ref bb_sum);

                if (IsDiscoverKing(BTree, ifrom, ito, color) != 1)
                    return 1;
            }

            return 0;
        }
        
        public static int CanKingEscape(ref BoardTree BTree, int ito, ref BitBoard bb, int color)
        {
            int iret = 0;
            if (BTree.ShogiBoard[ito] == (int)Piece.Type.Empty)
            {
                Xor(ito, ref BTree.BB_Occupied[color ^ 1]);
            }
            Xor(BTree.SQ_King[color], ref BTree.BB_Occupied[color]);

            BBOr(ref bb, bb, AbbMask[ito]);
            BBOr(ref bb, bb, BTree.BB_Occupied[color]);
            BBNotAnd(ref bb, AbbPieceAttacks[color, (int)Piece.Type.King, BTree.SQ_King[color]], bb);

            while (BBTest(bb) != 0)
            {
                int iescape = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                if (IsAttacked(BTree, iescape, color) == 0)
                {
                    iret = 1;
                    break;
                }
                Xor(iescape, ref bb);
            }

            Xor(BTree.SQ_King[color], ref BTree.BB_Occupied[color]);
            if (BTree.ShogiBoard[ito] == (int)Piece.Type.Empty)
            {
                Xor(ito, ref BTree.BB_Occupied[color ^ 1]);
            }

            return iret;
        }

        static bool CompareFormula0(int color, int sq_king)
        {
            if (color == 0)
            {
                if (sq_king <= (int)I2)
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
                if (sq_king >= (int)A8)
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

        static bool CompareFormula2(int color, int sq)
        {
            if (color == 0)
            {
                if (sq <= (int)I7)
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
                if (sq >= (int)A3)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }

        static bool CompareFormula3(int color, int board0, int board1)
        {
            if (color == 0)
            {
                if (board0 == (int)Piece.Type.Pawn && board1 <= 0)
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
                if (board0 == -(int)Piece.Type.Pawn && board1 >= 0)
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