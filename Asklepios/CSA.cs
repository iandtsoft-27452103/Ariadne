using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.ShogiCommon;

namespace Asklepios
{
    internal class CSA
    {
        public static Move CSA2Move(BoardTree BTree, string str_csa)
        {
            string str = "";
            Move move = new Move();
            str = str_csa.Substring(0, 2);
            int ifrom = Array.IndexOf(StrCSA, str);
            str = str_csa.Substring(2, 2);
            int ito = Array.IndexOf(StrCSA, str);
            str = str_csa.Substring(4, 2);
            Piece.Type piece;
            int flag_promo = 0;
            switch(str)
            {
                case "FU":
                    piece = Piece.Type.Pawn;
                    break;
                case "KY":
                    piece = Piece.Type.Lance;
                    break;
                case "KE":
                    piece = Piece.Type.Knight;
                    break;
                case "GI":
                    piece = Piece.Type.Silver;
                    break;
                case "KI":
                    piece = Piece.Type.Gold;
                    break;
                case "KA":
                    piece = Piece.Type.Bishop;
                    break;
                case "HI":
                    piece = Piece.Type.Rook;
                    break;
                case "OU":
                    piece = Piece.Type.King;
                    break;
                case "TO":
                    if (Math.Abs(BTree.ShogiBoard[ifrom]) == (int)Piece.Type.Pawn)
                    {
                        piece = Piece.Type.Pawn;
                        flag_promo = 1;
                    }
                    else
                    {
                        piece = Piece.Type.Pro_Pawn;
                    }
                    break;
                case "NY":
                    if (Math.Abs(BTree.ShogiBoard[ifrom]) == (int)Piece.Type.Lance)
                    {
                        piece = Piece.Type.Lance;
                        flag_promo = 1;
                    }
                    else
                    {
                        piece = Piece.Type.Pro_Lance;
                    }
                    break;
                case "NK":
                    if (Math.Abs(BTree.ShogiBoard[ifrom]) == (int)Piece.Type.Knight)
                    {
                        piece = Piece.Type.Knight;
                        flag_promo = 1;
                    }
                    else
                    {
                        piece = Piece.Type.Pro_Knight;
                    }
                    break;
                case "NG":
                    if (Math.Abs(BTree.ShogiBoard[ifrom]) == (int)Piece.Type.Silver)
                    {
                        piece = Piece.Type.Silver;
                        flag_promo = 1;
                    }
                    else
                    {
                        piece = Piece.Type.Pro_Silver;
                    }
                    break;
                case "UM":
                    if (Math.Abs(BTree.ShogiBoard[ifrom]) == (int)Piece.Type.Bishop)
                    {
                        piece = Piece.Type.Bishop;
                        flag_promo = 1;
                    }
                    else
                    {
                        piece = Piece.Type.Horse;
                    }
                    break;
                default:// "RY"
                    if (Math.Abs(BTree.ShogiBoard[ifrom]) == (int)Piece.Type.Rook)
                    {
                        piece = Piece.Type.Rook;
                        flag_promo = 1;
                    }
                    else
                    {
                        piece = Piece.Type.Dragon;
                    }
                    break;
            }
            Piece.Type cap_piece = (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]);

            move.Push(ifrom, ito, piece, cap_piece, flag_promo);
            return move;
        }

        public static string Move2CSA(Move move)
        {
            string str;

            str = StrCSA[move.From];
            str += StrCSA[move.To];
            if (move.FlagPromo == 0)
            {
                str += StrPiece[(int)move.PieceType];
            }
            else
            {
                str += StrPiece[(int)move.PieceType + 8];
            }

            return str;
        }
    }
}
