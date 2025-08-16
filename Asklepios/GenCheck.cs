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

namespace Asklepios
{
    internal class GenCheck
    {
        //後手玉を王手する
        public static void BGenChecks(ref BoardTree BTree, ref List<CheckMove> move_list)
        {
            int color = (int)Color.Type.Black;
            BitBoard bb_occupied = new BitBoard();
            BitBoard bb_attacks = new BitBoard();
            BitBoard bb_mask = BitBoard.SetFull();
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            int sq_wk = BTree.SQ_King[color ^ 1];

            //後手玉のあるところの縦横の利きを取得
            BitBoard bb_file_chk = BitBoard.Copy(GetFileAttacks(bb_occupied, sq_wk));
            BitBoard bb_rank_chk = BitBoard.Copy(GetRankAttacks(bb_occupied, sq_wk));
            BitBoard bb_rook_chk = new BitBoard();
            BBOr(ref bb_rook_chk, bb_file_chk, bb_rank_chk);

            //後手玉のあるところの斜めの利きを取得
            BitBoard bb_diag1_chk = BitBoard.Copy(GetDiag1Attacks(bb_occupied, sq_wk));
            BitBoard bb_diag2_chk = BitBoard.Copy(GetDiag2Attacks(bb_occupied, sq_wk));
            BitBoard bb_bishop_chk = new BitBoard();
            BBOr(ref bb_bishop_chk, bb_diag1_chk, bb_diag2_chk);

            //先手の駒を動かせる位置をbb_move_toに取得
            BitBoard bb_move_to = new BitBoard();
            BBNot(ref bb_move_to, BTree.BB_Occupied[color]);
            BBAnd(ref bb_move_to, bb_move_to, bb_mask);

            //駒を打てる位置をbb_drop_toに取得
            BitBoard bb_drop_to = new BitBoard();
            BBOr(ref bb_drop_to, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            BBNot(ref bb_drop_to, bb_drop_to);
            BBAnd(ref bb_drop_to, bb_drop_to, bb_mask);

            //先手玉を動かして後手玉が王手になる手
            BitBoard bb_chk, bb_desti;
            int ito;
            int ifrom = BTree.SQ_King[color];
            int idirec = (int)ADirec[sq_wk, ifrom];
            if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
            {
                bb_chk = new BitBoard();
                AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                BBAnd(ref bb_chk, bb_chk, AbbPieceAttacks[color, (int)Piece.Type.King, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.King, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            BitBoard bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            while(BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = new BitBoard();
                BBOr(ref bb_chk, bb_rook_chk, AbbPieceAttacks[color, (int)Piece.Type.King, sq_wk]);
                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);

                //龍が動ける位置を取得
                bb_desti = BitBoard.Copy(GetDragonAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_chk, bb_chk, bb_desti);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Dragon, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Horse]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = new BitBoard();
                BBOr(ref bb_chk, bb_bishop_chk, AbbPieceAttacks[color, (int)Piece.Type.King, sq_wk]);
                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);

                //馬が動ける位置を取得
                bb_desti = BitBoard.Copy(GetHorseAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_chk, bb_chk, bb_desti);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Horse, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            /*先手の飛を動かして後手玉が王手になる手*/
            //from位置が四段目～九段目の場合
            ulong u1 = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[1];
            ulong u2 = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[2];
            while ((u1 | u2) != 0)
            {
                ifrom = LastOne12(u1, u2);
                u1 ^= AbbMask[ifrom].p[1];
                u2 ^= AbbMask[ifrom].p[2];
                bb_desti = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));
                idirec = (int)ADirec[sq_wk, ifrom];
                bb_chk = new BitBoard();
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    BBAnd(ref bb_chk, bb_desti, bb_move_to);
                }
                else
                {
                    bb_chk = BitBoard.Copy(bb_rook_chk);
                    bb_chk.p[0] |= AbbPieceAttacks[color, (int)Piece.Type.King, sq_wk].p[0];
                    BBAnd(ref bb_chk, bb_chk, bb_desti);
                    BBAnd(ref bb_chk, bb_chk, bb_move_to);
                }

                //to位置が一段目～三段目なので成る
                while (bb_chk.p[0] != 0)
                {
                    ito = LastOne0(bb_chk.p[0]);
                    bb_chk.p[0] ^= AbbMask[ito].p[0];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }

                //to位置が四段目～九段目なので成らない（というか成れない）
                while ((bb_chk.p[1] | bb_chk.p[2]) != 0)
                {
                    ito = LastOne12(bb_chk.p[1], bb_chk.p[2]);
                    bb_chk.p[1] ^= AbbMask[ito].p[1];
                    bb_chk.p[2] ^= AbbMask[ito].p[2];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            //from位置が一段目～三段目の場合
            ulong u0 = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[0];
            while (u0 != 0)
            {
                ifrom = LastOne0(u0);
                u0 ^= AbbMask[ifrom].p[0];
                bb_desti = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));
                idirec = (int)ADirec[sq_wk, ifrom];
                bb_chk = new BitBoard();
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    BBAnd(ref bb_chk, bb_desti, bb_move_to);
                }
                else
                {
                    BBOr(ref bb_chk, bb_rook_chk, AbbPieceAttacks[color, (int)Piece.Type.King, sq_wk]);
                    BBAnd(ref bb_chk, bb_chk, bb_desti);
                    BBAnd(ref bb_chk, bb_chk, bb_move_to);
                }

                //成る手のみ生成
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            /*先手の角を動かして後手玉が王手になる手*/
            //from位置が四段目～九段目の場合
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[1];
            u2 = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[2];
            while ((u1 | u2) != 0)
            {
                ifrom = LastOne12(u1, u2);
                u1 ^= AbbMask[ifrom].p[1];
                u2 ^= AbbMask[ifrom].p[2];

                bb_desti = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));
                idirec = (int)ADirec[sq_wk, ifrom];
                bb_chk = new BitBoard();
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    BBAnd(ref bb_chk, bb_desti, bb_move_to);
                }
                else
                {
                    bb_chk = BitBoard.Copy(bb_bishop_chk);
                    bb_chk.p[0] |= AbbPieceAttacks[color, (int)Piece.Type.King, sq_wk].p[0];
                    BBAnd(ref bb_chk, bb_chk, bb_desti);
                    BBAnd(ref bb_chk, bb_chk, bb_move_to);
                }

                //to位置が一段目～三段目なので成る
                while (bb_chk.p[0] != 0)
                {
                    ito = LastOne0(bb_chk.p[0]);
                    bb_chk.p[0] ^= AbbMask[ito].p[0];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }

                //to位置が四段目～九段目なので成らない（というか成れない）
                while ((bb_chk.p[1] | bb_chk.p[2]) != 0)
                {
                    ito = LastOne12(bb_chk.p[1], bb_chk.p[2]);
                    bb_chk.p[1] ^= AbbMask[ito].p[1];
                    bb_chk.p[2] ^= AbbMask[ito].p[2];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            //from位置が一段目～三段目の場合
            u0 = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[0];
            while (u0 != 0)
            {
                ifrom = LastOne0(u0);
                u0 ^= AbbMask[ifrom].p[0];

                bb_desti = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));
                idirec = (int)ADirec[sq_wk, ifrom];
                bb_chk = new BitBoard();
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    BBAnd(ref bb_chk, bb_desti, bb_move_to);
                }
                else
                {
                    BBOr(ref bb_chk, bb_bishop_chk, AbbPieceAttacks[color, (int)Piece.Type.King, sq_wk]);
                    BBAnd(ref bb_chk, bb_chk, bb_desti);
                    BBAnd(ref bb_chk, bb_chk, bb_move_to);
                }

                //成る手のみ生成
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //先手の金、成銀、成桂、成香、と金を動かして後手玉が王手になる手
            bb_piece = BitBoard.Copy(BTree.BB_Total_Gold[color]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_wk]);

                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                }

                BBAnd(ref bb_chk, bb_chk, AbbPieceAttacks[color, (int)Piece.Type.Gold, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, (Piece.Type)BTree.ShogiBoard[ifrom], (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            /*先手の銀を動かして後手玉が王手になる手*/
            //from位置が一段目～三段目の場合
            u0 = BTree.BB_Piece[color, (int)Piece.Type.Silver].p[0];
            while (u0 != 0)
            {
                ifrom = LastOne0(u0);
                u0 ^= AbbMask[ifrom].p[0];

                bb_chk = new BitBoard();
                bb_chk.p[0] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_wk].p[0];
                bb_chk.p[1] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_wk].p[1];
                bb_chk.p[2] = 0;

                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                }

                bb_chk.p[0] &= bb_move_to.p[0] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[0];
                bb_chk.p[1] &= bb_move_to.p[1] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[1];

                //成る手のみ生成
                while ((bb_chk.p[0] | bb_chk.p[1]) != 0)
                {
                    ito = LastOne01(bb_chk.p[0], bb_chk.p[1]);
                    bb_chk.p[0] ^= AbbMask[ito].p[0];
                    bb_chk.p[1] ^= AbbMask[ito].p[1];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //from位置が四段目の場合
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Silver].p[1] & 0x7fc0000U;// 111111111 000000000 000000000
            while (u1 != 0)
            {
                ifrom = LastOne1(u1);
                u1 ^= AbbMask[ifrom].p[1];

                bb_chk = new BitBoard();
                bb_chk.p[0] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_wk].p[0];
                bb_chk.p[1] = bb_chk.p[2] = 0;

                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                }

                //成る手のみ生成
                bb_chk.p[0] &= bb_move_to.p[0] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[0];
                while (bb_chk.p[0] != 0)
                {
                    ito = LastOne0(bb_chk.p[0]);
                    bb_chk.p[0] ^= AbbMask[ito].p[0];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //不成の手のみ生成
            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Silver]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_wk]);

                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                }

                BBAnd(ref bb_chk, bb_chk, AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            /*先手の桂を動かして後手玉が王手になる手*/
            u0 = BTree.BB_Piece[color, (int)Piece.Type.Knight].p[0];
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Knight].p[1] & 0x7fffe00U;// 111111111 111111111 000000000
            while ((u0 | u1) != 0)
            {
                ifrom = LastOne01(u0, u1);
                u0 ^= AbbMask[ifrom].p[0];
                u1 ^= AbbMask[ifrom].p[1];

                bb_chk = new BitBoard();
                bb_chk.p[0] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_wk].p[0];
                bb_chk.p[1] = bb_chk.p[2] = 0;

                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0&& IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                }

                bb_chk.p[0] &= AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom].p[0] & bb_move_to.p[0];

                //成る手のみ生成
                while (bb_chk.p[0] != 0)
                {
                    ito = LastOne0(bb_chk.p[0]);
                    bb_chk.p[0] ^= AbbMask[ito].p[0];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //不成の手のみ生成
            u2 = BTree.BB_Piece[color, (int)Piece.Type.Knight].p[2];
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Knight].p[1] & 0x3ffffU;// 000000000 111111111 111111111
            while ((u2 | u1) != 0)
            {
                ifrom = LastOne12(u1, u2);
                u2 ^= AbbMask[ifrom].p[2];
                u1 ^= AbbMask[ifrom].p[1];

                bb_chk = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq_wk]);

                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                }

                BBAnd(ref bb_chk, bb_chk, AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            /* 先手の香を動かして後手玉が王手になる手 */
            //成る手のみ生成
            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Lance]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = new BitBoard();
                bb_chk.p[0] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_wk].p[0];
                bb_chk.p[1] = bb_chk.p[2] = 0;

                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                }

                bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_chk, bb_chk, bb_attacks);
                BBAnd(ref bb_chk, bb_chk, AbbPlusMinusRays[color, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //不成の手のみ生成
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Lance].p[1];
            u2 = BTree.BB_Piece[color, (int)Piece.Type.Lance].p[2];
            while ((u1 | u2) != 0)
            {
                ifrom = LastOne12(u1, u2);
                u1 ^= AbbMask[ifrom].p[1];
                u2 ^= AbbMask[ifrom].p[2];

                bb_chk = BitBoard.Copy(bb_file_chk);
                idirec = (int)ADirec[sq_wk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_wk);
                    BBAnd(ref bb_chk, bb_chk, AbbPlusMinusRays[color, ifrom]);
                }
                else 
                { 
                    BBAnd(ref bb_chk, bb_file_chk, AbbPlusMinusRays[color ^ 1, sq_wk]);
                }

                bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_chk, bb_chk, bb_attacks);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);
                bb_chk.p[0] = bb_chk.p[0] & 0x1ffU;// 000000000 000000000 111111111

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            bb_chk = new BitBoard();
            bb_chk.p[0] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_wk].p[0];
            if (sq_wk < (int)A2) { BBOr(ref bb_chk, bb_chk, AbbMask[sq_wk + NFile]); }
            BBAnd(ref bb_chk, bb_chk, bb_move_to);
            BBAnd(ref bb_chk, bb_chk, BTree.BB_PawnAttacks[color]);

            //先手の歩を突いて王手になる手を生成（馬筋・角筋を通す手）
            BBAnd(ref bb_piece, bb_diag1_chk, BTree.BB_Piece[color, (int)Piece.Type.Pawn]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                ito = ifrom - NFile;
                if (BTree.ShogiBoard[ito] > 0) { continue; }

                bb_desti = BitBoard.Copy(GetDiag1Attacks(bb_occupied, ifrom));
                if (BBContract(bb_desti, BTree.BB_BH[color]) != 0)
                {
                    BBNotAnd(ref bb_chk, bb_chk, AbbMask[ito]);

                    //成れる時は必ず成る
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    if (ifrom < (int)A5)
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    }
                    else
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    }
                    move_list.Add(cm);
                }
            }

            BBAnd(ref bb_piece, bb_diag2_chk, BTree.BB_Piece[color, (int)Piece.Type.Pawn]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                ito = ifrom - NFile;
                if (BTree.ShogiBoard[ito] > 0) { continue; }

                bb_desti = BitBoard.Copy(GetDiag2Attacks(bb_occupied, ifrom));
                if (BBContract(bb_desti, BTree.BB_BH[color]) != 0)
                {
                    BBNotAnd(ref bb_chk, bb_chk, AbbMask[ito]);

                    //成れる時は必ず成る
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    if (ifrom < (int)A5)
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    }
                    else
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    }
                    move_list.Add(cm);
                }
            }

            //先手の歩を突いて王手になる手を生成（龍・飛の横利きを通す手）
            BBAnd(ref bb_piece, bb_rank_chk, BTree.BB_Piece[color, (int)Piece.Type.Pawn]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                ito = ifrom - NFile;
                if (BTree.ShogiBoard[ito] > 0) { continue; }

                bb_desti = BitBoard.Copy(GetRankAttacks(bb_occupied, ifrom));
                if (BBContract(bb_desti, BTree.BB_RD[color]) != 0)
                {
                    BBNotAnd(ref bb_chk, bb_chk, AbbMask[ito]);

                    //成れる時は必ず成る
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    if (ifrom < (int)A5)
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                    }
                    else
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                    }
                    move_list.Add(cm);
                }
            }

            //玉頭に歩を突く手、成る手
            while (BBTest(bb_chk) != 0)
            {
                ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                Xor(ito, ref bb_chk);

                ifrom = ito + NFile;

                //成れる時は必ず成る
                CheckMove cm = new CheckMove();
                cm.check_type = 0;
                if (ifrom < (int)A5)
                {
                    cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(-BTree.ShogiBoard[ito]), 1);
                }
                else
                {
                    cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(-BTree.ShogiBoard[ito]), 0);
                }
                move_list.Add(cm);
            }

            /* 駒を打つ手 */
            //金を打つ手
            if (BTree.Hand[color, (int)Piece.Type.Gold] > 0) 
            {
                BBAnd(ref bb_chk, bb_drop_to, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_wk]);
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Gold, 0, 0);
                    move_list.Add(cm);
                }
            }

            //銀を打つ手
            if (BTree.Hand[color, (int)Piece.Type.Silver] > 0)
            {
                BBAnd(ref bb_chk, bb_drop_to, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_wk]);
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Silver, 0, 0);
                    move_list.Add(cm);
                }
            }

            //桂を打つ手
            //後手玉が7段目より上にいなければならない
            if (BTree.Hand[color, (int)Piece.Type.Knight] > 0 && sq_wk < (int)A2)
            {
                ito = sq_wk + 2 * NFile - 1;
                if (AiFile[sq_wk] != (int)file1 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty)
                {
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Knight, 0, 0);
                    move_list.Add(cm);
                }

                ito = sq_wk + 2 * NFile + 1;
                if (AiFile[sq_wk] != (int)file9 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty)
                {
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Knight, 0, 0);
                    move_list.Add(cm);
                }
            }

            //歩を打つ手
            //後手玉が8段目より上にいなければならない
            //二歩のチェック
            if (BTree.Hand[color, (int)Piece.Type.Pawn] > 0
                && sq_wk < (int)A1
                && (BBTest(BTree.BB_Piece[color, (int)Piece.Type.Pawn]) & (MaskFile[AiFile[sq_wk]])) == 0)
            {
                ito = sq_wk + NFile;

                //打ち歩詰めチェック
                if (BTree.ShogiBoard[ito] == (int)Piece.Type.Empty && IsMatePawnDrop(ref BTree, ito, color ^ 1) == 0)
                {
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Pawn, 0, 0);
                    move_list.Add(cm);
                }
            }

            //香を打つ手
            if (BTree.Hand[color, (int)Piece.Type.Lance] > 0)
            {
                int dist, min_dist;

                if ((int)AiFile[sq_wk] == (int)file1
                    || (int)AiFile[sq_wk] == (int)file9)
                {
                    min_dist = 2;
                }
                else { min_dist = 3; }

                for (ito = sq_wk + NFile, dist = 1; ito < NSquare && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                ito += NFile, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Lance, 0, 0);
                    move_list.Add(cm);
                }
            }

            /* 飛を打つ手 */
            if (BTree.Hand[color, (int)Piece.Type.Rook] > 0)
            {
                int file, dist, min_dist;

                if ((int)AiFile[sq_wk] == (int)file1
                    || (int)AiFile[sq_wk] == (int)file9)
                {
                    min_dist = 2;
                }
                else { min_dist = 3; }

                //後手玉より下の段から飛を打つ場合
                for (ito = sq_wk + NFile, dist = 1; ito < NSquare && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                ito += NFile, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(cm);
                }

                //後手玉の左側から飛を打つ場合
                for (file = (int)AiFile[sq_wk] - 1, ito = sq_wk - 1, dist = 1;
                file >= (int)file1 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    file -= 1, ito -= 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(cm);
                }

                if (sq_wk < (int)A8 || (int)I2 < sq_wk) { min_dist = 2; }
                else { min_dist = 3; }

                //後手玉の右側から飛を打つ場合
                for (file = (int)AiFile[sq_wk] + 1, ito = sq_wk + 1, dist = 1;
                file <= (int)file9 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    file += 1, ito += 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(cm);
                }

                //後手玉より上の段から飛を打つ場合
                for (ito = sq_wk - NFile, dist = 1; ito >= 0 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                ito -= NFile, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(cm);
                }
            }

            /*角を打つ手*/
            if (BTree.Hand[color, (int)Piece.Type.Bishop] > 0)
            {
                int file, rank, dist;

                ito = sq_wk;
                file = (int)AiFile[sq_wk];
                rank = (int)AiRank[sq_wk];

                //後手玉の左上から角を打つ場合
                for (ito -= 10, file -= 1, rank -= 1, dist = 1;
                file >= 0 && rank >= 0 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    ito -= 10, file -= 1, rank -= 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (rank == (int)rank3) { cm.check_type = 1; }
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > 2) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(cm);
                }

                ito = sq_wk;
                file = (int)AiFile[sq_wk];
                rank = (int)AiRank[sq_wk];

                //後手玉の右上から角を打つ場合
                for (ito -= 8, file += 1, rank -= 1, dist = 1;
                file <= (int)file9 && rank >= 0 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    ito -= 8, file += 1, rank -= 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (rank == (int)rank3) { cm.check_type = 1; }
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > 2) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(cm);
                }

                ito = sq_wk;
                file = (int)AiFile[sq_wk];
                rank = (int)AiRank[sq_wk];

                //後手玉の左下から角を打つ場合
                for (ito += 8, file -= 1, rank += 1, dist = 1;
                file >= 0 && rank <= (int)rank9 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    ito += 8, file -= 1, rank += 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > 2) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(cm);
                }

                ito = sq_wk;
                file = (int)AiFile[sq_wk];
                rank = (int)AiRank[sq_wk];

                //後手玉の右下から角を打つ場合
                for (ito += 10, file += 1, rank += 1, dist = 1;
                file <= (int)file9 && rank <= (int)rank9 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    ito += 10, file += 1, rank += 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > 2) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(cm);
                }
            }
        }

        //後手玉を王手する
        public static void WGenChecks(ref BoardTree BTree, ref List<CheckMove> move_list)
        {
            int color = (int)Color.Type.White;
            BitBoard bb_occupied = new BitBoard();
            BitBoard bb_attacks = new BitBoard();
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            int sq_bk = BTree.SQ_King[color ^ 1];

            //後手玉のあるところの縦横の利きを取得
            BitBoard bb_file_chk = BitBoard.Copy(GetFileAttacks(bb_occupied, sq_bk));
            BitBoard bb_rank_chk = BitBoard.Copy(GetRankAttacks(bb_occupied, sq_bk));
            BitBoard bb_rook_chk = new BitBoard();
            BBOr(ref bb_rook_chk, bb_file_chk, bb_rank_chk);

            //後手玉のあるところの斜めの利きを取得
            BitBoard bb_diag1_chk = BitBoard.Copy(GetDiag1Attacks(bb_occupied, sq_bk));
            BitBoard bb_diag2_chk = BitBoard.Copy(GetDiag2Attacks(bb_occupied, sq_bk));
            BitBoard bb_bishop_chk = new BitBoard();
            BBOr(ref bb_bishop_chk, bb_diag1_chk, bb_diag2_chk);

            //先手の駒を動かせる位置をbb_move_toに取得
            BitBoard bb_move_to = new BitBoard();
            BBNot(ref bb_move_to, BTree.BB_Occupied[color]);

            //駒を打てる位置をbb_drop_toに取得
            BitBoard bb_drop_to = new BitBoard();
            BBOr(ref bb_drop_to, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            BBNot(ref bb_drop_to, bb_drop_to);

            //先手玉を動かして後手玉が王手になる手
            BitBoard bb_chk, bb_desti;
            int ito;
            int ifrom = BTree.SQ_King[color];
            int idirec = (int)ADirec[sq_bk, ifrom];
            if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
            {
                bb_chk = new BitBoard();
                AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                BBAnd(ref bb_chk, bb_chk, AbbPieceAttacks[color, (int)Piece.Type.King, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.King, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            //後手の龍を動かして先手玉が王手になる手
            BitBoard bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = new BitBoard();
                BBOr(ref bb_chk, bb_rook_chk, AbbPieceAttacks[color, (int)Piece.Type.King, sq_bk]);
                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                //龍が動ける位置を取得
                //AttackDragon(bb_desti, from);
                bb_desti = BitBoard.Copy(GetDragonAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_chk, bb_chk, bb_desti);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Dragon, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            //後手の馬を動かして先手玉が王手になる手
            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Horse]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = new BitBoard();
                BBOr(ref bb_chk, bb_bishop_chk, AbbPieceAttacks[color, (int)Piece.Type.King, sq_bk]);
                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                //馬が動ける位置を取得
                bb_desti = BitBoard.Copy(GetHorseAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_chk, bb_chk, bb_desti);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Horse, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            /*後手の飛を動かして先手玉が王手になる手*/
            //from位置が一段目～六段目の場合
            ulong u0 = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[0];
            ulong u1 = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[1];
            while ((u0 | u1) != 0)
            {
                ifrom = LastOne01(u0, u1);
                u0 ^= AbbMask[ifrom].p[0];
                u1 ^= AbbMask[ifrom].p[1];
                bb_desti = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));
                idirec = (int)ADirec[sq_bk, ifrom];
                bb_chk = new BitBoard();
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    BBAnd(ref bb_chk, bb_desti, bb_move_to);
                }
                else
                {
                    bb_chk = BitBoard.Copy(bb_rook_chk);
                    bb_chk.p[2] |= AbbPieceAttacks[color, (int)Piece.Type.King, sq_bk].p[2];
                    BBAnd(ref bb_chk, bb_chk, bb_desti);
                    BBAnd(ref bb_chk, bb_chk, bb_move_to);
                }

                //to位置が七段目～九段目なので成る
                while (bb_chk.p[2] != 0)
                {
                    ito = LastOne2(bb_chk.p[2]);
                    bb_chk.p[2] ^= AbbMask[ito].p[2];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }

                //to位置が一段目～六段目なので成らない（というか成れない）
                while ((bb_chk.p[0] | bb_chk.p[1]) != 0)
                {
                    ito = LastOne01(bb_chk.p[0], bb_chk.p[1]);
                    bb_chk.p[0] ^= AbbMask[ito].p[0];
                    bb_chk.p[1] ^= AbbMask[ito].p[1];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            //from位置が七段目～九段目の場合
            ulong u2 = BTree.BB_Piece[color, (int)Piece.Type.Rook].p[2];
            while (u2 != 0)
            {
                ifrom = LastOne2(u2);
                u2 ^= AbbMask[ifrom].p[2];
                bb_desti = BitBoard.Copy(GetRookAttacks(bb_occupied, ifrom));
                idirec = (int)ADirec[sq_bk, ifrom];
                bb_chk = new BitBoard();
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    BBAnd(ref bb_chk, bb_desti, bb_move_to);
                }
                else
                {
                    BBOr(ref bb_chk, bb_rook_chk, AbbPieceAttacks[color, (int)Piece.Type.King, sq_bk]);
                    BBAnd(ref bb_chk, bb_chk, bb_desti);
                    BBAnd(ref bb_chk, bb_chk, bb_move_to);
                }

                //成る手のみ生成
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Rook, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            /*後手の角を動かして先手玉が王手になる手*/
            //from位置が一段目～六段目までの場合
            u0 = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[0];
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[1];
            while ((u0 | u1) != 0)
            {
                ifrom = LastOne01(u0, u1);
                u0 ^= AbbMask[ifrom].p[0];
                u1 ^= AbbMask[ifrom].p[1];

                bb_desti = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));

                idirec = (int)ADirec[sq_bk, ifrom];
                bb_chk = new BitBoard();
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    BBAnd(ref bb_chk, bb_desti, bb_move_to);
                }
                else
                {
                    bb_chk = BitBoard.Copy(bb_bishop_chk);
                    bb_chk.p[2] |= AbbPieceAttacks[color, (int)Piece.Type.King, sq_bk].p[2];
                    BBAnd(ref bb_chk, bb_chk, bb_desti);
                    BBAnd(ref bb_chk, bb_chk, bb_move_to);
                }

                //to位置が七段目～九段目なので成る
                while (bb_chk.p[2] != 0)
                {
                    ito = LastOne2(bb_chk.p[2]);
                    bb_chk.p[2] ^= AbbMask[ito].p[2];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }

                //to位置が一段目～六段目なので成らない（というか成れない）
                while ((bb_chk.p[0] | bb_chk.p[1]) != 0)
                {
                    ito = LastOne01(bb_chk.p[0], bb_chk.p[1]);
                    bb_chk.p[0] ^= AbbMask[ito].p[0];
                    bb_chk.p[1] ^= AbbMask[ito].p[1];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            //from位置が七段目～九段目の場合
            u2 = BTree.BB_Piece[color, (int)Piece.Type.Bishop].p[2];
            while (u2 != 0)
            {
                ifrom = LastOne2(u2);
                u2 ^= AbbMask[ifrom].p[2];

                bb_desti = BitBoard.Copy(GetBishopAttacks(bb_occupied, ifrom));

                idirec = (int)ADirec[sq_bk, ifrom];
                bb_chk = new BitBoard();
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    BBAnd(ref bb_chk, bb_desti, bb_move_to);
                }
                else
                {
                    BBOr(ref bb_chk, bb_bishop_chk, AbbPieceAttacks[color, (int)Piece.Type.King, sq_bk]);
                    BBAnd(ref bb_chk, bb_chk, bb_desti);
                    BBAnd(ref bb_chk, bb_chk, bb_move_to);
                }

                //成る手のみ生成
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Bishop, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //後手の金、成銀、成桂、成香、と金を動かして先手玉が王手になる手
            bb_piece = BitBoard.Copy(BTree.BB_Total_Gold[color]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk =BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_bk]);

                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                BBAnd(ref bb_chk, bb_chk, AbbPieceAttacks[color, (int)Piece.Type.Gold, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, (Piece.Type)(-BTree.ShogiBoard[ifrom]), (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            /*後手の銀を動かして先手玉が王手になる手*/
            //from位置が七段目～九段目の場合
            u2 = BTree.BB_Piece[color, (int)Piece.Type.Silver].p[2];
            while (u2 != 0)
            {
                ifrom = LastOne2(u2);
                u2 ^= AbbMask[ifrom].p[2];

                bb_chk = new BitBoard();
                bb_chk.p[2] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_bk].p[2];
                bb_chk.p[1] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_bk].p[1];
                bb_chk.p[0] = 0;

                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                bb_chk.p[2] &= bb_move_to.p[2] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[2];
                bb_chk.p[1] &= bb_move_to.p[1] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[1];

                //成る手のみ生成
                while ((bb_chk.p[2] | bb_chk.p[1]) != 0)
                {
                    ito = LastOne12(bb_chk.p[1], bb_chk.p[2]);
                    bb_chk.p[1] ^= AbbMask[ito].p[1];
                    bb_chk.p[2] ^= AbbMask[ito].p[2];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //from位置が六段目の場合
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Silver].p[1] & 0x1ffU;// 000000000 000000000 111111111
            while (u1 != 0)
            {
                ifrom = LastOne1(u1);
                u1 ^= AbbMask[ifrom].p[1];

                bb_chk = new BitBoard();
                bb_chk.p[2] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_bk].p[2];
                bb_chk.p[1] = bb_chk.p[0] = 0;

                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                bb_chk.p[2] &= bb_move_to.p[2] & AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom].p[2];

                //成る手のみ生成
                while (bb_chk.p[2] != 0)
                {
                    ito = LastOne2(bb_chk.p[2]);
                    bb_chk.p[2] ^= AbbMask[ito].p[2];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //不成の手のみ生成
            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Silver]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_bk]);

                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                BBAnd(ref bb_chk, bb_chk, AbbPieceAttacks[color, (int)Piece.Type.Silver, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Silver, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            /*後手の桂を動かして先手玉が王手になる手*/
            u2 = BTree.BB_Piece[color, (int)Piece.Type.Knight].p[2];
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Knight].p[1] & 0x3ffffU;// 000000000 111111111 111111111
            while ((u2 | u1) != 0)
            {
                ifrom = LastOne12(u1, u2);
                u2 ^= AbbMask[ifrom].p[2];
                u1 ^= AbbMask[ifrom].p[1];

                bb_chk = new BitBoard();
                bb_chk.p[2] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_bk].p[2];
                bb_chk.p[1] = bb_chk.p[0] = 0;

                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                bb_chk.p[2] &= AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom].p[2] & bb_move_to.p[2];

                //成る手のみ生成
                while (bb_chk.p[2] != 0)
                {
                    ito = LastOne2(bb_chk.p[2]);
                    bb_chk.p[2] ^= AbbMask[ito].p[2];
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //不成の手のみ生成
            u0 = BTree.BB_Piece[color, (int)Piece.Type.Knight].p[0];
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Knight].p[1] & 0x7fffe00U;// 111111111 111111111 000000000
            while ((u0 | u1) != 0)
            {
                ifrom = LastOne01(u0, u1);
                u0 ^= AbbMask[ifrom].p[0];
                u1 ^= AbbMask[ifrom].p[1];

                bb_chk = BitBoard.Copy(AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq_bk]);

                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                BBAnd(ref bb_chk, bb_chk, AbbPieceAttacks[color, (int)Piece.Type.Knight, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Knight, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            /*後手の香を動かして先手玉が王手になる手*/
            //成る手のみ生成
            bb_piece = BitBoard.Copy(BTree.BB_Piece[color, (int)Piece.Type.Lance]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                bb_chk = new BitBoard();
                bb_chk.p[2] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_bk].p[2];
                bb_chk.p[1] = bb_chk.p[0] = 0;

                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                }

                bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_chk, bb_chk, bb_attacks);
                BBAnd(ref bb_chk, bb_chk, AbbPlusMinusRays[color, ifrom]);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    move_list.Add(cm);
                }
            }

            //不成の手のみ生成
            u0 = BTree.BB_Piece[color, (int)Piece.Type.Lance].p[0];
            u1 = BTree.BB_Piece[color, (int)Piece.Type.Lance].p[1];
            while ((u0 | u1) != 0)
            {
                ifrom = LastOne01(u0, u1);
                u0 ^= AbbMask[ifrom].p[0];
                u1 ^= AbbMask[ifrom].p[1];

                bb_chk = BitBoard.Copy(bb_file_chk);
                idirec = (int)ADirec[sq_bk, ifrom];
                if (idirec != 0 && IsPinnedOnKing(BTree, ifrom, idirec, color ^ 1) != 0)
                {
                    AddBehindAttacks(ref bb_chk, idirec, sq_bk);
                    BBAnd(ref bb_chk, bb_chk, AbbPlusMinusRays[color, ifrom]);
                }
                else { BBAnd(ref bb_chk, bb_file_chk, AbbPlusMinusRays[color ^ 1, sq_bk]); }

                bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, ifrom));
                BBAnd(ref bb_chk, bb_chk, bb_attacks);
                BBAnd(ref bb_chk, bb_chk, bb_move_to);
                bb_chk.p[2] = bb_chk.p[2] & 0x7fc0000U;// 111111111 000000000 000000000

                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(ifrom, ito, Piece.Type.Lance, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    move_list.Add(cm);
                }
            }

            bb_chk = new BitBoard();
            bb_chk.p[2] = AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_bk].p[2];
            if (sq_bk > (int)I8) { BBOr(ref bb_chk, bb_chk, AbbMask[sq_bk - NFile]); }
            BBAnd(ref bb_chk, bb_chk, bb_move_to);
            BBAnd(ref bb_chk, bb_chk, BTree.BB_PawnAttacks[color]);

            //後手の歩を突いて王手になる手を生成（馬筋・角筋を通す手）
            BBAnd(ref bb_piece, bb_diag1_chk, BTree.BB_Piece[color, (int)Piece.Type.Pawn]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                ito = ifrom + NFile;
                if (BTree.ShogiBoard[ito] < 0) { continue; }

                //bb_desti = AttackDiag1(from);
                bb_desti = BitBoard.Copy(GetDiag1Attacks(bb_occupied, ifrom));
                if (BBContract(bb_desti, BTree.BB_BH[color]) != 0)
                {
                    BBNotAnd(ref bb_chk, bb_chk, AbbMask[ito]);

                    //成れる時は必ず成る
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    if (ifrom > (int)I5)
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    }
                    else
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    }
                    move_list.Add(cm);
                }
            }

            BBAnd(ref bb_piece, bb_diag2_chk, BTree.BB_Piece[color, (int)Piece.Type.Pawn]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                ito = ifrom + NFile;
                if (BTree.ShogiBoard[ito] < 0) { continue; }

                bb_desti = BitBoard.Copy(GetDiag2Attacks(bb_occupied, ifrom));
                if (BBContract(bb_desti, BTree.BB_BH[color]) != 0)
                {
                    BBNotAnd(ref bb_chk, bb_chk, AbbMask[ito]);

                    //成れる時は必ず成る
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    if (ifrom > (int)I5)
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    }
                    else
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    }
                    move_list.Add(cm);
                }
            }

            //後手の歩を突いて王手になる手を生成（龍筋・飛筋を通す手）
            BBAnd(ref bb_piece, bb_rank_chk, BTree.BB_Piece[color, (int)Piece.Type.Pawn]);
            while (BBTest(bb_piece) != 0)
            {
                ifrom = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                Xor(ifrom, ref bb_piece);

                ito = ifrom + NFile;
                if (BTree.ShogiBoard[ito] < 0) { continue; }

                bb_desti = BitBoard.Copy(GetRankAttacks(bb_occupied, ifrom));
                if (BBContract(bb_desti, BTree.BB_RD[color]) != 0)
                {
                    BBNotAnd(ref bb_chk, bb_chk, AbbMask[ito]);

                    //成れる時は必ず成る
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    if (ifrom > (int)I5)
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                    }
                    else
                    {
                        cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                    }
                    move_list.Add(cm);
                }
            }

            //玉頭に歩を突く手、成る手（成れる時は必ず成る）
            while (BBTest(bb_chk) != 0)
            {
                ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                Xor(ito, ref bb_chk);

                ifrom = ito - NFile;

                //成れる時は必ず成る
                CheckMove cm = new CheckMove();
                cm.check_type = 0;
                if (ifrom > (int)I5)
                {
                    cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(BTree.ShogiBoard[ito]), 1);
                }
                else
                {
                    cm.Push(ifrom, ito, Piece.Type.Pawn, (Piece.Type)(BTree.ShogiBoard[ito]), 0);
                }
                move_list.Add(cm);
            }

            /*駒を打つ手*/
            //金を打つ手
            if (BTree.Hand[color, (int)Piece.Type.Gold] > 0)
            {
                BBAnd(ref bb_chk, bb_drop_to, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_bk]);
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Gold, 0, 0);
                    move_list.Add(cm);
                }
            }

            //銀を打つ手
            if (BTree.Hand[color, (int)Piece.Type.Silver] > 0)
            {
                BBAnd(ref bb_chk, bb_drop_to, AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_bk]);
                while (BBTest(bb_chk) != 0)
                {
                    ito = LastOne012(bb_chk.p[0], bb_chk.p[1], bb_chk.p[2]);
                    Xor(ito, ref bb_chk);
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Silver, 0, 0);
                    move_list.Add(cm);
                }
            }

            //桂を打つ手
            if (BTree.Hand[color, (int)Piece.Type.Knight] > 0 && sq_bk > (int)I8)
            {
                ito = sq_bk - 2 * NFile - 1;
                if (AiFile[sq_bk] != (int)file1 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty)
                {
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Knight, 0, 0);
                    move_list.Add(cm);
                }

                ito = sq_bk - 2 * NFile + 1;
                if (AiFile[sq_bk] != (int)file9 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty)
                {
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Knight, 0, 0);
                    move_list.Add(cm);
                }
            }

            //歩を打つ手
            //先手玉が2段目より下にいなければならない
            //二歩のチェック
            if (BTree.Hand[color, (int)Piece.Type.Pawn] > 0
                && sq_bk > (int)I9
                && (BBTest(BTree.BB_Piece[color, (int)Piece.Type.Pawn]) & (MaskFile[AiFile[sq_bk]])) == 0)
            {
                ito = sq_bk - NFile;

                //打ち歩詰めチェック
                if (BTree.ShogiBoard[ito] == (int)Piece.Type.Empty && IsMatePawnDrop(ref BTree, ito, color) == 0)
                {
                    CheckMove cm = new CheckMove();
                    cm.check_type = 0;
                    cm.Push(NSquare, ito, Piece.Type.Pawn, 0, 0);
                    move_list.Add(cm);
                }
            }

            //香を打つ手
            if (BTree.Hand[color, (int)Piece.Type.Lance] > 0)
            {
                int dist, min_dist;

                if ((int)AiFile[sq_bk] == (int)file1
                    || (int)AiFile[sq_bk] == (int)file9)
                {
                    min_dist = 2;
                }
                else { min_dist = 3; }

                for (ito = sq_bk - NFile, dist = 1; ito >= 0 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                ito -= NFile, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Lance, 0, 0);
                    move_list.Add(cm);
                }
            }

            /* 飛を打つ手 */
            if (BTree.Hand[color, (int)Piece.Type.Rook] > 0)
            {
                int file, dist, min_dist;

                if ((int)AiFile[sq_bk] == (int)file1
                    || (int)AiFile[sq_bk] == (int)file9)
                {
                    min_dist = 2;
                }
                else { min_dist = 3; }

                //先手玉より上の段から飛を打つ場合
                for (ito = sq_bk - NFile, dist = 1; ito >= 0 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                ito -= NFile, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(cm);
                }

                //先手玉より下の段から飛を打つ場合
                for (ito = sq_bk + NFile, dist = 1; ito < NSquare && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                ito += NFile, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if ((int)AiRank[ito] == (int)rank7) { cm.check_type = 1; }
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(cm);
                }


                if (sq_bk < (int)A8 || (int)I2 < sq_bk) { min_dist = 2; }
                else { min_dist = 3; }

                //先手玉より右の筋から飛を打つ場合
                for (file = (int)AiFile[sq_bk] + 1, ito = sq_bk + 1, dist = 1;
                file <= (int)file9 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    file += 1, ito += 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(cm);
                }

                //先手玉より左の筋から飛を打つ場合
                for (file = (int)AiFile[sq_bk] - 1, ito = sq_bk - 1, dist = 1;
                file >= (int)file1 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    file -= 1, ito -= 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > min_dist) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Rook, 0, 0);
                    move_list.Add(cm);
                }
            }

            // 角を打つ手
            if (BTree.Hand[color, (int)Piece.Type.Bishop] > 0)
            {
                int file, rank, dist;

                ito = sq_bk;
                file = (int)AiFile[sq_bk];
                rank = (int)AiRank[sq_bk];

                //先手玉の右下から角を打つ手
                for (ito += 10, file += 1, rank += 1, dist = 1;
                file <= (int)file9 && rank <= (int)rank9 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    ito += 10, file += 1, rank += 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (rank == (int)rank7) { cm.check_type = 1; }
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > 2) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(cm);
                }

                ito = sq_bk;
                file = (int)AiFile[sq_bk];
                rank = (int)AiRank[sq_bk];

                //先手玉の左下から角を打つ手
                for (ito += 8, file -= 1, rank += 1, dist = 1;
                file >= 0 && rank <= (int)rank9 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    ito += 8, file -= 1, rank += 1, dist += 1)
                {
                    CheckMove cm = new CheckMove(); ;
                    if (rank == (int)rank7) { cm.check_type = 1; }
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > 2) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(cm);
                }

                ito = sq_bk;
                file = (int)AiFile[sq_bk];
                rank = (int)AiRank[sq_bk];

                //先手玉の右上から角を打つ手
                for (ito -= 8, file += 1, rank -= 1, dist = 1;
                file <= (int)file9 && rank >= 0 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    ito -= 8, file += 1, rank -= 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > 2) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(cm);
                }

                ito = sq_bk;
                file = (int)AiFile[sq_bk];
                rank = (int)AiRank[sq_bk];

                //先手玉の左上から角を打つ手
                for (ito -= 10, file -= 1, rank -= 1, dist = 1;
                file >= 0 && rank >= 0 && BTree.ShogiBoard[ito] == (int)Piece.Type.Empty;
                    ito -= 10, file -= 1, rank -= 1, dist += 1)
                {
                    CheckMove cm = new CheckMove();
                    if (dist == 1) { cm.check_type = 1; }
                    else if (dist > 2) { cm.check_type = 2; }
                    cm.Push(NSquare, ito, Piece.Type.Bishop, 0, 0);
                    move_list.Add(cm);
                }
            }
        }

        public static void AddBehindAttacks(ref BitBoard bb, int idirec, int ik)
        {
            BitBoard bb_tmp;

            switch(idirec)
            {
                case (int)direc_diag1:
                    bb_tmp = BitBoard.Copy(AbbDiag1Attacks[ik, 0]);
                    break;
                case (int)direc_diag2:
                    bb_tmp = BitBoard.Copy(AbbDiag2Attacks[ik, 0]);
                    break;
                case (int)direc_file:
                    bb_tmp = BitBoard.Copy(AbbFileAttacks[ik, 0]);
                    break;
                //case (int)direc_rank:
                default:
                    bb_tmp = BitBoard.Copy(AbbRankAttacks[ik, 0]);
                    break;
            }

            BBNot(ref bb_tmp, bb_tmp);
            BBOr(ref bb, bb, bb_tmp);
        }
    }
}
