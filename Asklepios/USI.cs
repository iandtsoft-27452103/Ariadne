using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.ShogiCommon;

namespace Asklepios
{
    internal class USI
    {
        public static Move USIToMove(BoardTree BTree, string str_usi_move)
        {
            Move move = new Move();
            Piece.Type pc = Piece.Type.Empty;
            Piece.Type cap = Piece.Type.Empty;
            int ifrom = NSquare;
            int flag_promo = 0;
            int ito;

            char c = str_usi_move[0];
            switch (c)
            {
                case 'P':
                case 'p':
                    pc = Piece.Type.Pawn;
                    break;
                case 'L':
                case 'l':
                    pc = Piece.Type.Lance;
                    break;
                case 'S':
                case 's':
                    pc = Piece.Type.Silver;
                    break;
                case 'G':
                case 'g':
                    pc = Piece.Type.Gold;
                    break;
                case 'B':
                case 'b':
                    pc = Piece.Type.Bishop;
                    break;
                case 'R':
                case 'r':
                    pc = Piece.Type.Rook;
                    break;
                default:
                    ifrom = USIToSquare(str_usi_move[0].ToString(), str_usi_move[1].ToString());
                    pc = (Piece.Type)Math.Abs(BTree.ShogiBoard[ifrom]);
                    break;
            }

            ito = USIToSquare(str_usi_move[2].ToString(), str_usi_move[3].ToString());
            cap = (Piece.Type)Math.Abs(BTree.ShogiBoard[ito]);

            if (str_usi_move.Length == 5)
                flag_promo = 1;

            move.Push(ifrom, ito, pc, cap, flag_promo);

            return move;
        }

        public static int USIToSquare(string str_file, string str_rank)
        {
            int f = NFile - int.Parse(str_file);
            int r;
            char c = str_rank[0];

            switch(c)
            {
                case 'a':
                    r = (int)Rank.rank1;
                    break;
                case 'b':
                    r = (int)Rank.rank2;
                    break;
                case 'c':
                    r = (int)Rank.rank3;
                    break;
                case 'd':
                    r = (int)Rank.rank4;
                    break;
                case 'e':
                    r = (int)Rank.rank5;
                    break;
                case 'f':
                    r = (int)Rank.rank6;
                    break;
                case 'g':
                    r = (int)Rank.rank7;
                    break;
                case 'h':
                    r = (int)Rank.rank8;
                    break;
                default:
                    r = (int)Rank.rank9;
                    break;
            }
            return r * NRank + f;
        }

        public static string BoardToUSI(Move move)
        {
            string str_usi = "";

            if (move.From == NSquare)
            {
                switch(move.PieceType)
                {
                    case Piece.Type.Pawn:
                        str_usi = "P*" + StrUSI[move.To];
                        break;
                    case Piece.Type.Lance:
                        str_usi = "L*" + StrUSI[move.To];
                        break;
                    case Piece.Type.Knight:
                        str_usi = "N*" + StrUSI[move.To];
                        break;
                    case Piece.Type.Silver:
                        str_usi = "S*" + StrUSI[move.To];
                        break;
                    case Piece.Type.Gold:
                        str_usi = "G*" + StrUSI[move.To];
                        break;
                    case Piece.Type.Bishop:
                        str_usi = "B*" + StrUSI[move.To];
                        break;
                    case Piece.Type.Rook:
                        str_usi = "R*" + StrUSI[move.To];
                        break;
                }
                return str_usi;
            }

            str_usi = StrUSI[move.From] + StrUSI[move.To];

            if (move.FlagPromo == 1)
                str_usi += '+';

            return str_usi;
        }
    }
}
