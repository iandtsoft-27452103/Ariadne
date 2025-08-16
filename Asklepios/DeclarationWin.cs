using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.AttackBitBoard;
using static Asklepios.ShogiCommon;
using static Asklepios.BitOperation;
using System.Reflection.Metadata;

namespace Asklepios
{
    internal class DeclarationWin
    {
        // 0:宣言勝ちでない
        // 1:先手の勝ち
        // 2:後手の勝ち
        public static int JudgeDeclarationWin(BoardTree BTree)
        {
            //if (IsAttacked(BTree, BTree.SQ_King[(int)Color.Type.Black], (int)Color.Type.Black) != 0) { return 0; }
            //if (IsAttacked(BTree, BTree.SQ_King[(int)Color.Type.White], (int)Color.Type.White) != 0) { return 0; }

            ulong lbb_piece0, lbb_piece2;
            int sq_bk, sq_wk, black_score, white_score, piece_count, piece_count2;
            int b_tekijin_piece_count, w_tekijin_piece_count;

            sq_bk = (int)BTree.SQ_King[(int)Color.Type.Black];
            sq_wk = (int)BTree.SQ_King[(int)Color.Type.White]; ;

            // 宣言勝ちできる位置に玉がいるかチェック
            if (sq_bk > (int)Square.I7 && sq_wk < (int)Square.A3) { return 0; }

            black_score = white_score = piece_count = piece_count2 = 0;
            b_tekijin_piece_count = w_tekijin_piece_count = 0;

            // 先手の点数と駒数の計算
            if (sq_bk < (int)Square.A6)
            {
                piece_count = BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Rook] + BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Rook];
                black_score = 5 * piece_count;
                piece_count = BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Gold] + BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Silver] + BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Knight] + BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Lance] + BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Pawn];
                black_score += piece_count;
                lbb_piece0 = BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Dragon].p[0];
                piece_count = PopuCount(lbb_piece0);
                lbb_piece0 = BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Rook].p[0];
                piece_count += PopuCount(lbb_piece0);
                lbb_piece0 = BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Horse].p[0];
                piece_count += PopuCount(lbb_piece0);
                lbb_piece0 = BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Bishop].p[0];
                piece_count += PopuCount(lbb_piece0);
                black_score += 5 * piece_count;

                lbb_piece0 = BTree.BB_Total_Gold[(int)Color.Type.Black].p[0];
                piece_count2 = PopuCount(lbb_piece0);
                lbb_piece0 = BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Silver].p[0];
                piece_count2 += PopuCount(lbb_piece0);
                lbb_piece0 = BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Knight].p[0];
                piece_count2 += PopuCount(lbb_piece0);
                lbb_piece0 = BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Lance].p[0];
                piece_count2 += PopuCount(lbb_piece0);
                lbb_piece0 = BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Pawn].p[0];
                piece_count2 += PopuCount(lbb_piece0);
                black_score += piece_count2;

                b_tekijin_piece_count = piece_count + piece_count2;
            }

            // 後手の点数と駒数の計算
            if (sq_wk > (int)Square.I4)
            {
                piece_count = BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Rook] + BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Bishop];
                white_score = 5 * piece_count;
                piece_count = BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Gold] + BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Silver] + BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Knight] + BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Lance] + BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Pawn];
                white_score += piece_count;
                lbb_piece2 = BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Dragon].p[2];
                piece_count = PopuCount(lbb_piece2);
                lbb_piece2 = BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Rook].p[2];
                piece_count += PopuCount(lbb_piece2);
                lbb_piece2 = BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Horse].p[2];
                piece_count += PopuCount(lbb_piece2);
                lbb_piece2 = BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Bishop].p[2];
                piece_count += PopuCount(lbb_piece2);
                white_score += 5 * piece_count;

                lbb_piece2 = BTree.BB_Total_Gold[(int)Color.Type.White].p[2];
                piece_count2 = PopuCount(lbb_piece2);
                lbb_piece2 = BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Silver].p[2];
                piece_count2 += PopuCount(lbb_piece2);
                lbb_piece2 = BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Knight].p[2];
                piece_count2 += PopuCount(lbb_piece2);
                lbb_piece2 = BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Lance].p[2];
                piece_count2 += PopuCount(lbb_piece2);
                lbb_piece2 = BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Pawn].p[2];
                piece_count2 += PopuCount(lbb_piece2);
                white_score += piece_count2;

                w_tekijin_piece_count = piece_count + piece_count2;
            }


            // 先手が宣言勝ちの局面である
            if (sq_bk < (int)Square.A6 && black_score >= 28 && b_tekijin_piece_count >= 10) { return 1; }


            // 後手が宣言勝ちの局面である
            if (sq_wk > (int)Square.I4 && white_score >= 27 && w_tekijin_piece_count >= 10) { return 2; }

            return 0;
        }
    }
}
