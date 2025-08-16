using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.ShogiCommon;
using static Asklepios.ShogiCommon.Square;
using static Asklepios.Color;
using static Asklepios.BitOperation;
using static Asklepios.AttackBitBoard;
using System.Drawing;

namespace Asklepios
{
    internal class Check
    {
        public static CheckTable[,] ChkTbl = new CheckTable[ColorNum, NSquare];

        public static void Init()
        {
            BitBoard bb_check = new BitBoard();
            BitBoard bb = new BitBoard();
            int sq_chk, color;

            for (int iking = 0; iking < NSquare; iking++)
            {
                color = 0;

                //先手の金
                ChkTbl[color, iking].gold = new BitBoard();
                bb_check = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking]);
                while (BBTest(bb_check) != 0)
                {
                    sq_chk = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    BBOr(ref ChkTbl[color, iking].gold, ChkTbl[color, iking].gold, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_chk]);
                    Xor(sq_chk, ref bb_check);
                }
                BBOr(ref bb, AbbMask[iking], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking]);
                BBNotAnd(ref ChkTbl[color, iking].gold, ChkTbl[color, iking].gold, bb);

                //先手の銀
                ChkTbl[color, iking].silver = new BitBoard();
                bb_check = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, iking]);
                while (BBTest(bb_check) != 0)
                {
                    sq_chk = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    BBOr(ref ChkTbl[color, iking].silver, ChkTbl[color, iking].silver, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_chk]);
                    Xor(sq_chk, ref bb_check);
                }
                bb_check.p[0] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking].p[0];
                while (bb_check.p[0] != 0)
                {
                    sq_chk = LastOne0(bb_check.p[0]);
                    BBOr(ref ChkTbl[color, iking].silver, ChkTbl[color, iking].silver, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_chk]);
                    bb_check.p[0] ^= AbbMask[sq_chk].p[0];
                }
                bb_check.p[1] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking].p[1];
                while (bb_check.p[1] != 0)
                {
                    sq_chk = LastOne1(bb_check.p[1]);
                    ChkTbl[color, iking].silver.p[0] |= AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_chk].p[0];
                    bb_check.p[1] ^= AbbMask[sq_chk].p[1];
                }
                BBOr(ref bb, AbbMask[iking], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, iking]);
                BBNotAnd(ref ChkTbl[color, iking].silver, ChkTbl[color, iking].silver, bb);

                //先手の桂
                ChkTbl[color, iking].knight = new BitBoard();
                bb_check = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, iking]);
                while (BBTest(bb_check) != 0)
                {
                    sq_chk = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    BBOr(ref ChkTbl[color, iking].knight, ChkTbl[color, iking].knight, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq_chk]);
                    Xor(sq_chk, ref bb_check);
                }
                bb_check.p[0] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking].p[0];
                while (bb_check.p[0] != 0)
                {
                    sq_chk = LastOne0(bb_check.p[0]);
                    BBOr(ref ChkTbl[color, iking].knight, ChkTbl[color, iking].knight, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq_chk]);
                    bb_check.p[0] ^= AbbMask[sq_chk].p[0];
                }

                //先手の香
                ChkTbl[color, iking].lance = new BitBoard();
                if (iking <= (int)I3)
                {
                    //ChkTbl[color, iking].lance = new BitBoard();
                    BBAnd(ref ChkTbl[color, iking].lance, AbbPlusMinusRays[color ^ 1, iking + NFile], AbbFileAttacks[iking, 0]);
                    if (iking <= (int)I7 && iking != (int)A9 && iking != (int)A8 && iking != (int)A7)
                    {
                        BBAnd(ref bb, AbbPlusMinusRays[color ^ 1, iking - 1], AbbFileAttacks[iking - 1, 0]);
                        BBOr(ref ChkTbl[color, iking].lance, ChkTbl[color, iking].lance, bb);
                    }
                    if (iking <= (int)I7 && iking != (int)I9 && iking != (int)I8 && iking != (int)I7)
                    {
                        BBAnd(ref bb, AbbPlusMinusRays[color ^ 1, iking + 1], AbbFileAttacks[iking + 1, 0]);
                        BBOr(ref ChkTbl[color, iking].lance, ChkTbl[color, iking].lance, bb);
                    }
                }

                color = 1;

                //後手の金
                ChkTbl[color, iking].gold = new BitBoard();
                bb_check = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking]);
                while (BBTest(bb_check) != 0)
                {
                    sq_chk = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    BBOr(ref ChkTbl[color, iking].gold, ChkTbl[color, iking].gold, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold,sq_chk]);
                    Xor(sq_chk, ref bb_check);
                }
                BBOr(ref bb, AbbMask[iking], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking]);
                BBNotAnd(ref ChkTbl[color, iking].gold, ChkTbl[color, iking].gold, bb);

                //後手の銀
                ChkTbl[color, iking].silver = new BitBoard();
                bb_check = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, iking]);
                while (BBTest(bb_check) != 0)
                {
                    sq_chk = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    BBOr(ref ChkTbl[color, iking].silver, ChkTbl[color, iking].silver, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_chk]);
                    Xor(sq_chk, ref bb_check);
                }
                bb_check.p[2] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking].p[2];
                while (bb_check.p[2] != 0)
                {
                    sq_chk = LastOne2(bb_check.p[2]);
                    BBOr(ref ChkTbl[color, iking].silver, ChkTbl[color, iking].silver, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_chk]);
                    bb_check.p[2] ^= AbbMask[sq_chk].p[2];
                }
                bb_check.p[1] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking].p[1];
                while (bb_check.p[1] != 0)
                {
                    sq_chk = LastOne1(bb_check.p[1]);
                    ChkTbl[color, iking].silver.p[2] |= AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_chk].p[2];
                    bb_check.p[1] ^= AbbMask[sq_chk].p[1];
                }
                BBOr(ref bb, AbbMask[iking], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, iking]);
                BBNotAnd(ref ChkTbl[color, iking].silver, ChkTbl[color , iking].silver, bb);

                //後手の桂
                ChkTbl[color, iking].knight = new BitBoard();
                bb_check = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, iking]);
                while (BBTest(bb_check) != 0)
                {
                    sq_chk = LastOne012(bb_check.p[0], bb_check.p[1], bb_check.p[2]);
                    BBOr(ref ChkTbl[color, iking].knight, ChkTbl[color, iking].knight, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq_chk]);
                    Xor(sq_chk, ref bb_check);
                }
                bb_check.p[2] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, iking].p[2];
                while (bb_check.p[2] != 0)
                {
                    sq_chk = LastOne2(bb_check.p[2]);
                    BBOr(ref ChkTbl[color, iking].knight, ChkTbl[color, iking].knight, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq_chk]);
                    bb_check.p[2] ^= AbbMask[sq_chk].p[2];
                }

                //後手の香
                ChkTbl[color, iking].lance = new BitBoard();
                if (iking >= (int)A7)
                {
                    BBAnd(ref ChkTbl[color, iking].lance, AbbPlusMinusRays[color ^ 1, iking - NFile], AbbFileAttacks[iking, 0]);
                    if (iking >= (int)A3 && iking != (int)A3 && iking != (int)A2 && iking != (int)A1)
                    {
                        BBAnd(ref bb, AbbPlusMinusRays[color ^ 1, iking - 1], AbbFileAttacks[iking - 1, 0]);
                        BBOr(ref ChkTbl[color, iking].lance, ChkTbl[color, iking].lance, bb);
                    }
                    if (iking >= (int)A3 && iking != (int)A3 && iking != (int)I2 && iking != (int)I1)
                    {
                        BBAnd(ref bb, AbbPlusMinusRays[color ^ 1, iking + 1], AbbFileAttacks[iking + 1, 0]);
                        BBOr(ref ChkTbl[color, iking].lance, ChkTbl[color, iking].lance, bb);
                    }
                }
            }
        }
    }
}
