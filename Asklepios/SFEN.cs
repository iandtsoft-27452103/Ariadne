using Asklepios;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.AttackBitBoard;
using static Asklepios.ShogiCommon;

namespace Asklepios
{
    internal class SFEN
    {
        public static string str_sfen_start_position = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

        public static string ToSFEN(BoardTree BTree, int color)
        {
            string str_sfen = "";
            int i = 0;
            while (i < NSquare)
            {
                int pc = BTree.ShogiBoard[i];
                switch (pc)
                {
                    case (int)Piece.Type.Pawn:
                        str_sfen += "P";
                        break;
                    case (int)Piece.Type.Lance:
                        str_sfen += "L";
                        break;
                    case (int)Piece.Type.Knight:
                        str_sfen += "N";
                        break;
                    case (int)Piece.Type.Silver:
                        str_sfen += "S";
                        break;
                    case (int)Piece.Type.Gold:
                        str_sfen += "G";
                        break;
                    case (int)Piece.Type.Bishop:
                        str_sfen += "B";
                        break;
                    case (int)Piece.Type.Rook:
                        str_sfen += "R";
                        break;
                    case (int)Piece.Type.King:
                        str_sfen += "K";
                        break;
                    case (int)Piece.Type.Pro_Pawn:
                        str_sfen += "+P";
                        break;
                    case (int)Piece.Type.Pro_Lance:
                        str_sfen += "+L";
                        break;
                    case (int)Piece.Type.Pro_Knight:
                        str_sfen += "+N";
                        break;
                    case (int)Piece.Type.Pro_Silver:
                        str_sfen += "+S";
                        break;
                    case (int)Piece.Type.Horse:
                        str_sfen += "+B";
                        break;
                    case (int)Piece.Type.Dragon:
                        str_sfen += "+R";
                        break;
                    case -(int)Piece.Type.Pawn:
                        str_sfen += "p";
                        break;
                    case -(int)Piece.Type.Lance:
                        str_sfen += "l";
                        break;
                    case -(int)Piece.Type.Knight:
                        str_sfen += "n";
                        break;
                    case -(int)Piece.Type.Silver:
                        str_sfen += "s";
                        break;
                    case -(int)Piece.Type.Gold:
                        str_sfen += "g";
                        break;
                    case -(int)Piece.Type.Bishop:
                        str_sfen += "b";
                        break;
                    case -(int)Piece.Type.Rook:
                        str_sfen += "r";
                        break;
                    case -(int)Piece.Type.King:
                        str_sfen += "k";
                        break;
                    case -(int)Piece.Type.Pro_Pawn:
                        str_sfen += "+p";
                        break;
                    case -(int)Piece.Type.Pro_Lance:
                        str_sfen += "+l";
                        break;
                    case -(int)Piece.Type.Pro_Knight:
                        str_sfen += "+n";
                        break;
                    case -(int)Piece.Type.Pro_Silver:
                        str_sfen += "+s";
                        break;
                    case -(int)Piece.Type.Horse:
                        str_sfen += "+b";
                        break;
                    case -(int)Piece.Type.Dragon:
                        str_sfen += "+r";
                        break;
                    case (int)Piece.Type.Empty:
                        int empty_cnt = 1;
                        bool is_rank_end = false;
                        while (true)
                        {
                            uint current_rank = AiRank[i++];
                            if (i == NSquare) { break; }
                            if (current_rank != AiRank[i])
                            {
                                is_rank_end = true;
                                break;
                            }

                            if (BTree.ShogiBoard[i] == (int)Piece.Type.Empty)
                            {
                                empty_cnt++;
                            }
                            else
                            {
                                break;
                            }
                        }
                        str_sfen += empty_cnt.ToString();
                        if (is_rank_end == true)
                            str_sfen += "/";
                        continue;// breakではない
                }
                if (AiFile[i] == (uint)Asklepios.ShogiCommon.File.file9 && i != 80)
                    str_sfen += "/";
                i++;
            }

            str_sfen += " ";

            if (color == (int)Color.Type.Black)
            {
                str_sfen += "b ";
            }
            else
            {
                str_sfen += "w ";
            }

            bool flag = false;
            for (i = 0; i < Color.ColorNum; i++)
            {
                int j = (int)Piece.Type.Rook;
                while (j > (int)Piece.Type.Empty)
                {
                    string s = BTree.Hand[i, j].ToString();
                    if (s == "0")
                    {
                        j--;
                        continue;
                    }

                    flag = true;
                    if (s != "1")
                        str_sfen += s;

                    if (i == (int)Color.Type.Black)
                    {
                        switch(j)
                        {
                            case (int)Piece.Type.Pawn:
                                str_sfen += "P";
                                break;
                            case (int)Piece.Type.Lance:
                                str_sfen += "L";
                                break;
                            case (int)Piece.Type.Knight:
                                str_sfen += "N";
                                break;
                            case (int)Piece.Type.Silver:
                                str_sfen += "S";
                                break;
                            case (int)Piece.Type.Gold:// 2025.3.24 修正
                                str_sfen += "G";
                                break;
                            case (int)Piece.Type.Bishop:
                                str_sfen += "B";
                                break;
                            case (int)Piece.Type.Rook:
                                str_sfen += "R";
                                break;
                        }
                    }
                    else
                    {
                        switch (j)
                        {
                            case (int)Piece.Type.Pawn:
                                str_sfen += "p";
                                break;
                            case (int)Piece.Type.Lance:
                                str_sfen += "l";
                                break;
                            case (int)Piece.Type.Knight:
                                str_sfen += "n";
                                break;
                            case (int)Piece.Type.Silver:
                                str_sfen += "s";
                                break;
                            case (int)Piece.Type.Gold:// 2025.3.24 修正
                                str_sfen += "g";
                                break;
                            case (int)Piece.Type.Bishop:
                                str_sfen += "b";
                                break;
                            case (int)Piece.Type.Rook:
                                str_sfen += "r";
                                break;
                        }
                    }
                    j--;
                }
            }

            if (!flag)
                str_sfen += "-";

            str_sfen += " 1";

            return str_sfen;
        }

        public static BoardTree ParseSFEN(string str_sfen)
        {
            BoardTree BTree = new BoardTree();
            BoardTreeAlloc(ref BTree);
            string[] str_temp = str_sfen.Split(' ');
            string str_pos = str_temp[0];
            int length = str_pos.Length;
            int i = 0;
            int pos = 0;
            bool flag_promo = false;

            while (pos < length)
            {
                char c = str_pos[pos];
                switch (c)
                {
                    case 'P':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Pawn;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Pawn]);
                            Xor(i - 9, ref BTree.BB_PawnAttacks[(int)Color.Type.Black]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Pro_Pawn;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Pro_Pawn]);
                            Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.Black]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.Black]);
                        i++;
                        pos++;
                        break;
                    case 'L':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Lance;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Lance]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Pro_Lance;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Pro_Lance]);
                            Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.Black]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.Black]);
                        i++;
                        pos++;
                        break;
                    case 'N':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Knight;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Knight]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Pro_Knight;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Pro_Knight]);
                            Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.Black]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.Black]);
                        i++;
                        pos++;
                        break;
                    case 'S':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Silver;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Silver]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Pro_Silver;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Pro_Silver]);
                            Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.Black]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.Black]);
                        i++;
                        pos++;
                        break;
                    case 'G':
                        BTree.ShogiBoard[i] = (int)Piece.Type.Gold;
                        Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Gold]);
                        Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.Black]);
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.Black]);
                        i++;
                        pos++;
                        break;
                    case 'B':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Bishop;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Bishop]);
                            Xor(i, ref BTree.BB_BH[(int)Color.Type.Black]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Horse;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Horse]);
                            Xor(i, ref BTree.BB_BH[(int)Color.Type.Black]);
                            Xor(i, ref BTree.BB_HDK[(int)Color.Type.Black]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.Black]);
                        i++;
                        pos++;
                        break;
                    case 'R':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Rook;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Rook]);
                            Xor(i, ref BTree.BB_RD[(int)Color.Type.Black]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = (int)Piece.Type.Dragon;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.Dragon]);
                            Xor(i, ref BTree.BB_RD[(int)Color.Type.Black]);
                            Xor(i, ref BTree.BB_HDK[(int)Color.Type.Black]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.Black]);
                        i++;
                        pos++;
                        break;
                    case 'K':
                        BTree.ShogiBoard[i] = (int)Piece.Type.King;
                        BTree.SQ_King[(int)Color.Type.Black] = i;
                        Xor(i, ref BTree.BB_Piece[(int)Color.Type.Black, (int)Piece.Type.King]);
                        Xor(i, ref BTree.BB_HDK[(int)Color.Type.Black]);
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.Black]);
                        i++;
                        pos++;
                        break;
                    case 'p':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Pawn;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Pawn]);
                            Xor(i + 9, ref BTree.BB_PawnAttacks[(int)Color.Type.White]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Pro_Pawn;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Pro_Pawn]);
                            Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.White]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.White]);
                        i++;
                        pos++;
                        break;
                    case 'l':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Lance;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Lance]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Pro_Lance;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Pro_Lance]);
                            Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.White]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.White]);
                        i++;
                        pos++;
                        break;
                    case 'n':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Knight;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Knight]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Pro_Knight;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Pro_Knight]);
                            Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.White]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.White]);
                        i++;
                        pos++;
                        break;
                    case 's':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Silver;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Silver]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Pro_Silver;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Pro_Silver]);
                            Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.White]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.White]);
                        i++;
                        pos++;
                        break;
                    case 'g':
                        BTree.ShogiBoard[i] = -(int)Piece.Type.Gold;
                        Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Gold]);
                        Xor(i, ref BTree.BB_Total_Gold[(int)Color.Type.White]);
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.White]);
                        i++;
                        pos++;
                        break;
                    case 'b':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Bishop;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Bishop]);
                            Xor(i, ref BTree.BB_BH[(int)Color.Type.White]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Horse;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Horse]);
                            Xor(i, ref BTree.BB_BH[(int)Color.Type.White]);
                            Xor(i, ref BTree.BB_HDK[(int)Color.Type.White]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.White]);
                        i++;
                        pos++;
                        break;
                    case 'r':
                        if (flag_promo == false)
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Rook;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Rook]);
                            Xor(i, ref BTree.BB_RD[(int)Color.Type.White]);
                        }
                        else
                        {
                            BTree.ShogiBoard[i] = -(int)Piece.Type.Dragon;
                            Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.Dragon]);
                            Xor(i, ref BTree.BB_RD[(int)Color.Type.White]);
                            Xor(i, ref BTree.BB_HDK[(int)Color.Type.White]);
                            flag_promo = false;
                        }
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.White]);
                        i++;
                        pos++;
                        break;
                    case 'k':
                        BTree.ShogiBoard[i] = -(int)Piece.Type.King;
                        BTree.SQ_King[(int)Color.Type.White] = i;
                        Xor(i, ref BTree.BB_Piece[(int)Color.Type.White, (int)Piece.Type.King]);
                        Xor(i, ref BTree.BB_HDK[(int)Color.Type.White]);
                        Xor(i, ref BTree.BB_Occupied[(int)Color.Type.White]);
                        i++;
                        pos++;
                        break;
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7':
                    case '8':
                    case '9':
                        int limit = int.Parse(c.ToString());
                        for (int j = 0; j < limit; j++)
                            BTree.ShogiBoard[i++] = (int)Piece.Type.Empty;
                        pos++;
                        break;
                    case '+':
                        flag_promo = true;
                        pos++;
                        break;
                    case '/':
                        pos++;
                        break;
                    default:
                        pos++;
                        break;
                }
            }

            if (str_temp[1] == "b")
            {
                BTree.RootColor = Color.Type.Black;
            }
            else
            {
                BTree.RootColor = Color.Type.White;
            }

            str_pos = str_temp[2];
            length = str_pos.Length;
            pos = 0;
            int hand_count = 0;
            bool flag = false;

            while (pos < length)
            {
                char c = str_pos[pos];
                switch (c)
                {
                    case '-':
                        break;
                    case '0':
                        if (flag == true)
                            hand_count = 10;
                        flag = false;
                        break;
                    case '1':
                        flag = true;
                        break;
                    case '2':
                        if (flag == true)
                        {
                            hand_count = 12;
                        }
                        else
                        {
                            hand_count = 2;
                        }
                        flag = false;
                        break;
                    case '3':
                        if (flag == true)
                        {
                            hand_count = 13;
                        }
                        else
                        {
                            hand_count = 3;
                        }
                        flag = false;
                        break;
                    case '4':
                        if (flag == true)
                        {
                            hand_count = 14;
                        }
                        else
                        {
                            hand_count = 4;
                        }
                        flag = false;
                        break;
                    case '5':
                        if (flag == true)
                        {
                            hand_count = 15;
                        }
                        else
                        {
                            hand_count = 5;
                        }
                        flag = false;
                        break;
                    case '6':
                        if (flag == true)
                        {
                            hand_count = 16;
                        }
                        else
                        {
                            hand_count = 6;
                        }
                        flag = false;
                        break;
                    case '7':
                        if (flag == true)
                        {
                            hand_count = 17;
                        }
                        else
                        {
                            hand_count = 7;
                        }
                        flag = false;
                        break;
                    case '8':
                        if (flag == true)
                        {
                            hand_count = 18;
                        }
                        else
                        {
                            hand_count = 8;
                        }
                        flag = false;
                        break;
                    case '9':
                        hand_count = 9;
                        break;
                    case 'P':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Pawn] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Pawn] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'L':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Lance] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Lance] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'N':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Knight] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Knight] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'S':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Silver] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Silver] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'G':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Gold] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Gold] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'B':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Bishop] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Bishop] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'R':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Rook] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.Black, (int)Piece.Type.Rook] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'p':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Pawn] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Pawn] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'l':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Lance] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Lance] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'n':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Knight] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Knight] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 's':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Silver] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Silver] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'g':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Gold] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Gold] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'b':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Bishop] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Bishop] = 1;
                        }
                        hand_count = 0;
                        break;
                    case 'r':
                        if (hand_count > 1)
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Rook] = hand_count;
                        }
                        else
                        {
                            BTree.Hand[(int)Color.Type.White, (int)Piece.Type.Rook] = 1;
                        }
                        hand_count = 0;
                        break;
                }
                pos++;
            }

            if (str_temp.Length == 4)
                return BTree;

            int color = 0;
            for (i = 5; i < str_temp.Length; i++)// plyは一旦削除
            {
                string str_move = str_temp[i];
                Move m = new Move();
                m = USI.USIToMove(BTree, str_move);
                MakeMove.Do(ref BTree, color, m);
                color ^= 1;
            }

            return BTree;
        }
    }
}
