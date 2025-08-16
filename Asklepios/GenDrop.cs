using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.AttackBitBoard;
using static Asklepios.BitOperation;
using static Asklepios.ShogiCommon;

namespace Asklepios
{
    internal class GenDrop
    {
        public static void Generate(ref BoardTree BTree, ref List<Move> move_list, int color)
        {
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

            int ito, bb_index, num;
            BitBoard bb_target = new BitBoard();
            ulong lbb_target_a, lbb_target_b;
            BBOr(ref bb_target, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            BBNot(ref bb_target, bb_target);

            if (color == 0)
            {
                lbb_target_a = bb_target.p[0] & 0x7fc0000U;
                lbb_target_b = bb_target.p[0] & 0x003fe00U;
                bb_target.p[0] &= 0x00001ffU;
                bb_target.p[1] &= 0x7ffffffU;
                bb_target.p[2] &= 0x7ffffffU;
                bb_index = 0;
                num = 26;
            }
            else
            {
                lbb_target_a = bb_target.p[2] & 0x00001ffU;
                lbb_target_b = bb_target.p[2] & 0x003fe00U;
                bb_target.p[0] &= 0x7ffffffU;
                bb_target.p[1] &= 0x7ffffffU;
                bb_target.p[2] &= 0x7fc0000U;
                bb_index = 2;
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
                    if (ais_pawn[AiFile[ito]] == 0 && IsMatePawnDrop(ref BTree, ito, color) == 0)// IsMatePawnDropはバグがあるかもしれないので注意！
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

                    lbb_target_b ^= AbbMask[ito].p[bb_index];
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

                    lbb_target_b ^= AbbMask[ito].p[bb_index];
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

                lbb_target_a ^= AbbMask[ito].p[bb_index];
            }
        }
    }
}
