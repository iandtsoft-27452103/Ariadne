using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.MakeMove;
using static Asklepios.BitOperation;
using static Asklepios.AttackBitBoard;
using System.Net.NetworkInformation;

namespace Asklepios
{
    internal class Test
    {
        // 局面更新のテスト
        public static void TestMakeMove()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            List<Record> records = IO.ReadRecordFile("test_records.txt");

            BoardIni(ref bt);
            Record record = records[2];

            int color = 0;
            for (int i = 0; i < record.str_moves.Length; i++)
            {
                Move move = CSA.CSA2Move(bt, record.str_moves[i]);

                BoardTree bt_before = bt.DeepCopy(bt, false);
                Do(ref bt, color, move);
                UnDo(ref bt, color, move);
                BoardTree bt_after = bt.DeepCopy(bt, false);
                /*if (!CompareBoard(bt_before, bt_after))
                {
                    string err_msg = "Error in ply = " + (i + 1).ToString();
                    Console.WriteLine(err_msg);
                    return;
                }*/

                Do(ref bt, color, move);
                color ^= 1;

                if (i == 89)
                {
                    OutBoard(bt);
                    return;
                }                
            }
        }

        public static void TestDropMove()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);

            List<string> comments_sfen = new List<string>();
            List<string> comments_answer = new List<string>();
            List<string> str_sfen = IO.ReadTestFile("test_data_drop.txt", ref comments_sfen);
            List<string> str_moves = IO.ReadTestFile("answer_data_drop.txt", ref comments_answer);

            for (int i = 0; i < str_sfen.Count; i++)
            {
                BoardIni(ref bt);
                bt = SFEN.ParseSFEN(str_sfen[i]);
                string[] str_move = str_moves[i].Split(' ');

                List<Move> moves = new List<Move>();
                for (int j = 0; j < str_move.Length; j++)
                {
                    if (str_move[j].Length == 0)
                        continue;
                    Move move = CSA.CSA2Move(bt, str_move[j]);
                    moves.Add(move);
                }

                if (moves.Count == 0)
                    continue;

                List<Move> gen_moves = new List<Move>();
                GenDrop.Generate(ref bt, ref gen_moves, (int)bt.RootColor);

                if (gen_moves.Count != moves.Count)
                {
                    string err_msg = "assertion in line " + (i + 1).ToString();
                    err_msg += " genmoves != answer_moves";
                    Console.WriteLine(err_msg);
                    err_msg = "gen_count = " + gen_moves.Count.ToString() + " ans_count = " + moves.Count.ToString();
                    Console.WriteLine(err_msg);
                }

                /*for (int j = 0; j < moves.Count; j++)
                {
                    bool error_flag = true;
                    for (int k = 0; k < moves.Count; k++)
                    {
                        if (gen_moves[j].Value == moves[k].Value)
                        {
                            error_flag = false;
                            break;
                        }
                    }

                    if (error_flag)
                    {
                        string err_msg = "error in line " + (i + 1).ToString();
                        err_msg += " gen_moves don't contain " + str_move[j];
                        Console.WriteLine(err_msg);
                        //return;
                    }
                }*/
            }
        }

        public static void TestNoCapMove()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);

            List<string> comments_sfen = new List<string>();
            List<string> comments_answer = new List<string>();
            List<string> str_sfen = IO.ReadTestFile("test_data_gennocap.txt", ref comments_sfen);
            List<string> str_moves = IO.ReadTestFile("answer_data_gennocap.txt", ref comments_answer);

            for (int i = 0; i < str_sfen.Count; i++)
            {
                BoardIni(ref bt);
                bt = SFEN.ParseSFEN(str_sfen[i]);
                string[] str_move = str_moves[i].Split(' ');

                List<Move> moves = new List<Move>();
                for (int j = 0; j < str_move.Length; j++)
                {
                    if (str_move[j].Length == 0)
                        continue;
                    Move move = CSA.CSA2Move(bt, str_move[j]);
                    moves.Add(move);
                }

                if (moves.Count == 0)
                    continue;

                List<Move> gen_moves = new List<Move>();
                GenNoCap.Generate(bt, ref gen_moves, (int)bt.RootColor);

                if (gen_moves.Count != moves.Count)
                {
                    string err_msg = "assertion in line " + (i + 1).ToString();
                    err_msg += " genmoves != answer_moves";
                    Console.WriteLine(err_msg);
                    err_msg = "gen_count = " + gen_moves.Count.ToString() + " ans_count = " + moves.Count.ToString();
                    Console.WriteLine(err_msg);
                }

                /*for (int j = 0; j < moves.Count; j++)
                {
                    bool error_flag = true;
                    for (int k = 0; k < moves.Count; k++)
                    {
                        if (gen_moves[j] == moves[k])
                        {
                            error_flag = false;
                            break;
                        }
                    }

                    if (error_flag)
                    {
                        string err_msg = "error in line " + (j + 1).ToString();
                        err_msg += " gen_moves don't contain " + str_move[j];
                        Console.WriteLine(err_msg);
                        return;
                    }
                }*/
            }
        }

        public static void TestCapMove()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);

            List<string> comments_sfen = new List<string>();
            List<string> comments_answer = new List<string>();
            List<string> str_sfen = IO.ReadTestFile("test_data_gencap.txt", ref comments_sfen);
            List<string> str_moves = IO.ReadTestFile("answer_data_gencap.txt", ref comments_answer);

            for (int i = 0; i < str_sfen.Count; i++)
            {
                BoardIni(ref bt);
                bt = SFEN.ParseSFEN(str_sfen[i]);
                string[] str_move = str_moves[i].Split(' ');

                List<Move> moves = new List<Move>();
                for (int j = 0; j < str_move.Length; j++)
                {
                    if (str_move[j].Length == 0)
                        continue;
                    Move move = CSA.CSA2Move(bt, str_move[j]);
                    moves.Add(move);
                }

                if (moves.Count == 0)
                    continue;

                List<Move> gen_moves = new List<Move>();
                GenCap.Generate(bt, ref gen_moves, (int)bt.RootColor);

                if (gen_moves.Count != moves.Count)
                {
                    string err_msg = "assertion in line " + (i + 1).ToString();
                    err_msg += " genmoves != answer_moves";
                    Console.WriteLine(err_msg);
                    err_msg = "gen_count = " + gen_moves.Count.ToString() + " ans_count = " + moves.Count.ToString();
                    Console.WriteLine(err_msg);
                }

                /*for (int j = 0; j < moves.Count; j++)
                {
                    bool error_flag = true;
                    for (int k = 0; k < moves.Count; k++)
                    {
                        if (gen_moves[j] == moves[k])
                        {
                            error_flag = false;
                            break;
                        }
                    }

                    if (error_flag)
                    {
                        string err_msg = "error in line " + (j + 1).ToString();
                        err_msg += " gen_moves don't contain " + str_move[j];
                        Console.WriteLine(err_msg);
                        return;
                    }
                }*/
            }
        }

        public static void TestEvasionMove()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);

            List<string> comments_sfen = new List<string>();
            List<string> comments_answer = new List<string>();
            List<string> str_sfen = IO.ReadTestFile("test_data_evasion.txt", ref comments_sfen);
            List<string> str_moves = IO.ReadTestFile("answer_data_evasion.txt", ref comments_answer);

            for (int i = 0; i < str_sfen.Count; i++)
            {
                BoardIni(ref bt);
                bt = SFEN.ParseSFEN(str_sfen[i]);
                string[] str_move = str_moves[i].Split(' ');

                List<Move> moves = new List<Move>();
                for (int j = 0; j < str_move.Length; j++)
                {
                    if (str_move[j].Length == 0)
                        continue;
                    Move move = CSA.CSA2Move(bt, str_move[j]);
                    moves.Add(move);
                }

                if (moves.Count == 0)
                    continue;

                List<Move> gen_moves = new List<Move>();
                GenEvasion.Generate(ref bt, ref gen_moves, (int)bt.RootColor);

                if (gen_moves.Count != moves.Count)
                {
                    string err_msg = "assertion in line " + (i + 1).ToString();
                    err_msg += " genmoves != answer_moves";
                    Console.WriteLine(err_msg);
                }

                /*for (int j = 0; j < moves.Count; j++)
                {
                    bool error_flag = true;
                    for (int k = 0; k < moves.Count; k++)
                    {
                        if (gen_moves[j] == moves[k])
                        {
                            error_flag = false;
                            break;
                        }
                    }

                    if (error_flag)
                    {
                        string err_msg = "error in line " + (j + 1).ToString();
                        err_msg += " gen_moves don't contain " + str_move[j];
                        Console.WriteLine(err_msg);
                        return;
                    }
                }*/
            }
        }

        // Checker駒がPINの場合でも王手生成している。詰み探索ではInCheckの確認が必要。おそらく3手詰めでも。
        public static void TestCheckMove()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);

            List<string> comments_sfen = new List<string>();
            List<string> comments_answer = new List<string>();
            List<string> str_sfen = IO.ReadTestFile("test_data_check.txt", ref comments_sfen);
            List<string> str_moves = IO.ReadTestFile("answer_data_check.txt", ref comments_answer);

            // i == 165でエラー　count = 168
            for (int i = 0; i < str_sfen.Count; i++)
            {
                BoardIni(ref bt);
                bt = SFEN.ParseSFEN(str_sfen[i]);
                string[] str_move = str_moves[i].Split(' ');

                List<Move> moves = new List<Move>();
                for (int j = 0; j < str_move.Length; j++)
                {
                    if (str_move[j].Length == 0)
                        continue;
                    Move move = CSA.CSA2Move(bt, str_move[j]);
                    moves.Add(move);
                }

                if (moves.Count == 0)
                    continue;

                List<CheckMove> gen_moves = new List<CheckMove>();
                //GenEvasion.Generate(ref bt, ref gen_moves, (int)bt.RootColor);
                if (bt.RootColor == Color.Type.Black)
                {
                    GenCheck.BGenChecks(ref bt, ref gen_moves);
                }
                else
                {
                    GenCheck.WGenChecks(ref bt, ref gen_moves);
                }

                if (gen_moves.Count != moves.Count)
                {
                    Console.WriteLine(comments_sfen[i]);
                    string err_msg = "assertion in line " + (i + 1).ToString();
                    err_msg += " genmoves != answer_moves";
                    Console.WriteLine(err_msg);
                    err_msg = "gen_count = " + gen_moves.Count.ToString() + " ans_count = " + moves.Count.ToString();
                    Console.WriteLine(err_msg);
                }
                else
                {
                    string msg = "";
                    msg = "gen_count = " + gen_moves.Count.ToString() + " ans_count = " + moves.Count.ToString();
                    Console.WriteLine(msg);
                }

                string gen_csa = "";
                for (int j = 0; j < gen_moves.Count; j++)
                {
                    gen_csa += CSA.Move2CSA(gen_moves[j]);
                    if (gen_csa.Length == 4)
                    {
                        int x = 0;
                    }
                    if (j != gen_moves.Count - 1)
                    {
                        gen_csa += ", ";
                    }
                }

                Console.WriteLine(gen_csa);

                /*for (int j = 0; j < moves.Count; j++)
                {
                    bool error_flag = true;
                    for (int k = 0; k < moves.Count; k++)
                    {
                        if (gen_moves[j] == moves[k])
                        {
                            error_flag = false;
                            break;
                        }
                    }

                    if (error_flag)
                    {
                        Console.WriteLine(comments_sfen[i]);
                        string err_msg = "error in line " + (j + 1).ToString();
                        err_msg += " gen_moves don't contain " + str_move[j];
                        Console.WriteLine(err_msg);
                        //return;
                    }
                }*/
            }
        }

        public static void TestMate1Ply(int color)
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);

            string str_test_data_file_name;
            string str_answer_data_file_name;

            if (color == 0)
            {
                str_test_data_file_name = "test_data_b_mate1ply.txt";
                str_answer_data_file_name = "answer_data_b_mate1ply.txt";
            }
            else
            {
                str_test_data_file_name = "test_data_w_mate1ply.txt";
                str_answer_data_file_name = "answer_data_w_mate1ply.txt";
            }

            List<string> comments_sfen = new List<string>();
            List<string> comments_answer = new List<string>();
            List<string> str_sfen = IO.ReadTestFile(str_test_data_file_name, ref comments_sfen);
            List<string> str_moves = IO.ReadTestFile(str_answer_data_file_name, ref comments_answer);

            for (int i = 0; i < str_sfen.Count; i++)
            {
                BoardIni(ref bt);
                bt = SFEN.ParseSFEN(str_sfen[i]);
                string str_move = str_moves[i];

                Move move = new Move();

                if (str_move != "")
                {
                    move = CSA.CSA2Move(bt, str_move);
                }

                //List<CheckMove> gen_moves = new List<CheckMove>();
                Move mate_move = new Move();
                int iret = Mate1Ply.IsMateIn1Ply(ref bt, ref mate_move, color);

                if (iret == 0 && str_move != "")
                {
                    string err_msg = "It's mate, but mate move isn't generated in line " + (i + 1).ToString() + ".";
                    Console.WriteLine(comments_sfen[i]);
                    Console.WriteLine(err_msg);
                    //return;
                }

                if (iret == 1 && str_move == "")
                {
                    string err_msg = "It isn't mate, but mate move generated in line " + (i + 1).ToString() + ".";
                    Console.WriteLine(comments_sfen[i]);
                    Console.WriteLine(err_msg);
                    //return;
                }

                if (move.Value != mate_move.Value)
                {
                    //string err_msg = "Mate move generated, but isn't correct in line " + (i + 1).ToString() + ".";
                    //Console.WriteLine(err_msg);
                    //return;
                }
            }
        }

        public static void TestMate3Ply()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);

            Mate3Ply mate_object = new Mate3Ply();

            string str_test_data_file_name;

            str_test_data_file_name = "test_data_mate3ply.txt";

            List<string> comments_sfen = new List<string>();
            List<string> str_sfen = IO.ReadTestFile(str_test_data_file_name, ref comments_sfen);
            string[] answer = { "mate", "mate", "mate", "mate", "mate", "not mate", "not mate", "not mate", "not mate", "not mate" };

            for (int i = 0; i < str_sfen.Count; i++)
            {
                BoardIni(ref bt);
                bt = SFEN.ParseSFEN(str_sfen[i]);

                //List<CheckMove> gen_moves = new List<CheckMove>();
                Move[] mate_moves = new Move[3];
                for (int j = 0; j < mate_moves.Length; j++)
                    mate_moves[j] = new Move();
                bool bret = mate_object.IsMateIn3Ply(ref bt, (int)bt.RootColor, false, ref mate_moves);
                //int iret = Mate1Ply.IsMateIn1Ply(ref bt, ref mate_move, color);

                if (bret)
                {
                    string str_msg = "line " + (i + 1).ToString() + ": mate ";
                    for (int j = 0; j < mate_moves.Length; j++)
                    {
                        if (mate_moves[j].Value != 0)
                        {
                            str_msg += CSA.Move2CSA(mate_moves[j]);
                        }
                    }
                    Console.WriteLine(str_msg);
                }
                else
                {
                    string str_msg = "not mate";
                    Console.WriteLine(str_msg);
                }
            }
        }

        public static void TestDeclarationWin()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);

            List<string> str_sfen = new List<string>();
            string s;

            s = "+L+NSGKGS+N+L/1+R5+B1/+P+P+P+P+P+P+P+P+P/9/9/9/+p+p+p+p+p+p+p+p+p/1+r5+b1/+l+nsgkgs+n+l b - 1";// 後手勝ち
            str_sfen.Add(s);
            s = "+L+NSGKGS+N+L/+P+R5+B1/+P+P+P+P+P+P+P+P+P/9/9/9/+p+p+p+p+p+p+p+p1/1+r5+b1/+l+nsgkgs+n+l b - 1";// 先手勝ち
            str_sfen.Add(s);
            s = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";// どちらの勝ちでもない → 初期局面
            str_sfen.Add(s);
            s = "+L+NSGK4/1+R7/+P+P+P+P+P4/9/9/9/+p+p+p+p+p4/7+b1/+l+nsgk4 b BGSNL4Prgsnl4p 1";// 後手勝ち
            str_sfen.Add(s);
            s = "+L+NSGK4/1+R7/+P+P+P+P+P4/9/9/9/+p+p+p+p+p4/7+b1/+l+nsgk4 b BGSNL5Prgsnl3p 1";// 先手勝ち
            str_sfen.Add(s);
            s = "4k4/9/9/9/9/9/9/4p4/4K4 b RB2G2S2N2L9Prb2g2s2n2l8p 1";// どちらの勝ちでもない → 先手玉が王手
            str_sfen.Add(s);
            s = "4k4/4P4/9/9/9/9/9/9/4K4 b RB2G2S2N2L8Prb2g2s2n2l9p 1";// どちらの勝ちでもない → 後手玉が王手
            str_sfen.Add(s);
            s = "+L+NSGKGS+N+L/1+R5+B1/+P+P+P+P+P+P+P+P+P/9/9/8k/+p+p+p+p+p+p+p+p+p/1+r5+b1/+l+nsg1gs+n+l b - 1";// どちらの勝ちでもない → 後手玉が宣言勝ちの位置にいない
            str_sfen.Add(s);
            s = "+L+NSG1GS+N+L/+P+R5+B1/+P+P+P+P+P+P+P+P+P/K8/9/9/+p+p+p+p+p+p+p+p1/1+r5+b1/+l+nsgkgs+n+l b - 1";// どちらの勝ちでもない → 先手玉が宣言勝ちの位置にいない
            str_sfen.Add(s);
            s = "+L+NSGK4/1+R7/+P+P+P+P+P4/9/9/8k/+p+p+p+p+p4/7+b1/+l+nsg5 b BGSNL4Prgsnl4p 1";// どちらの勝ちでもない → 後手玉が宣言勝ちの位置にいない
            str_sfen.Add(s);
            s = "+L+NSG5/1+R7/+P+P+P+P+P4/K8/9/9/+p+p+p+p+p4/7+b1/+l+nsgk4 b BGSNL5Prgsnl3p 1";// どちらの勝ちでもない → 先手玉が宣言勝ちの位置にいない
            str_sfen.Add(s);

            for (int i = 0; i < str_sfen.Count; i++)
            {
                BoardIni(ref bt);
                bt = SFEN.ParseSFEN(str_sfen[i]);

                int iret = DeclarationWin.JudgeDeclarationWin(bt);

                string str_msg = "";
                if (iret == 0)
                {
                    str_msg = "line " + (i + 1).ToString() + " : You cannot declare.";
                }
                else if (iret == 1)
                {
                    str_msg = "line " + (i + 1).ToString() + " : Black win.";
                }
                else if (iret == 2)
                {
                    str_msg = "line " + (i + 1).ToString() + " : White win.";
                }
                Console.WriteLine(str_msg);
            }
        }

        static void Init()
        {
            Hash.IniRand(5489U);
            Hash.IniRandomTable();
            AttackBitBoard.IniTables();
            AttackBitBoard.IniLongAttacks();
            Check.Init();
        }

        // バグあり
        public static bool CompareBoard(BoardTree bt1, BoardTree bt2)
        {
            for (int i = 0; i < Color.ColorNum; i++)
            {
                if (bt1.BB_BH[i] != bt2.BB_BH[i])
                    return false;
                if (bt1.BB_RD[i] != bt2.BB_RD[i])
                    return false;
                if (bt1.BB_HDK[i] != bt2.BB_HDK[i])
                    return false;
                if (bt1.BB_Total_Gold[i] != bt2.BB_Total_Gold[i])
                    return false;
                if (bt1.BB_Occupied[i] != bt2.BB_Occupied[i])
                    return false;
                if (bt1.BB_PawnAttacks[i] != bt2.BB_PawnAttacks[i])
                    return false;
                if (bt1.SQ_King[i] != bt2.SQ_King[i])
                    return false;

                for (int j = 0; j < Piece.PieceNum; j++)
                {
                    if (bt1.BB_Piece[i, j] != bt2.BB_Piece[i, j])
                        return false;
                }

                for (int j = 0; j < ShogiCommon.NHand; j++)
                {
                    if (bt1.Hand[i, j] != bt2.Hand[i, j])
                        return false;
                }
            }

            for (int i = 0; i < ShogiCommon.NSquare; i++)
            {
                if (bt1.ShogiBoard[i] != bt2.ShogiBoard[i])
                    return false;
            }

            if (bt1.CurrentHash != bt2.CurrentHash)
                return false;
            if (bt1.PrevHash != bt2.PrevHash)
                return false;
            if (bt1.RootColor != bt2.RootColor)
                return false;
            if (bt1.ply != bt2.ply)
                return false;

            return true;
        }

        public static void OutBoard(BoardTree bt)
        {
            string out_board = "";
            int count = 0;
            bool flag;
            for (int i = 0; i < ShogiCommon.NSquare; i++)
            {
                if (count == 9)
                {
                    out_board += "\n";
                    count = 0;
                }

                flag = false;
                if (bt.ShogiBoard[i] < 0)
                    flag = true;
                Piece.Type t = new Piece.Type();
                t = (Piece.Type)Math.Abs(bt.ShogiBoard[i]);
                string str_piece = "";
                switch (t)
                {
                    case Piece.Type.Empty:
                        str_piece = "  ";
                        break;
                    case Piece.Type.Pawn:
                        str_piece = "歩";
                        break;
                    case Piece.Type.Lance:
                        str_piece = "香";
                        break;
                    case Piece.Type.Knight:
                        str_piece = "桂";
                        break;
                    case Piece.Type.Silver:
                        str_piece = "銀";
                        break;
                    case Piece.Type.Gold:
                        str_piece = "金";
                        break;
                    case Piece.Type.Bishop:
                        str_piece = "角";
                        break;
                    case Piece.Type.Rook:
                        str_piece = "飛";
                        break;
                    case Piece.Type.King:
                        str_piece = "玉";
                        break;
                    case Piece.Type.Pro_Pawn:
                        str_piece = "と";
                        break;
                    case Piece.Type.Pro_Lance:
                        str_piece = "杏";
                        break;
                    case Piece.Type.Pro_Knight:
                        str_piece = "圭";
                        break;
                    case Piece.Type.Pro_Silver:
                        str_piece = "吟";
                        break;
                    case Piece.Type.Horse:
                        str_piece = "馬";
                        break;
                    case Piece.Type.Dragon:
                        str_piece = "龍";
                        break;
                }
                if (!flag)
                {
                    str_piece = " " + str_piece;
                }
                else
                {
                    str_piece = "v" + str_piece;
                }
                out_board += str_piece + "|";
                count++;
            }

            string[] str_hand = new string[2];
            str_hand[0] = str_hand[1] = "";
            for (int i = 0; i < Color.ColorNum; i++)
            {
                for (int j = ShogiCommon.NHand - 1; j > 0; j--)
                {
                    switch ((Piece.Type)j)
                    {
                        case Piece.Type.Pawn:
                            str_hand[i] = str_hand[i] + "歩" + bt.Hand[i, j].ToString();
                            break;
                        case Piece.Type.Lance:
                            str_hand[i] = str_hand[i] + "香" + bt.Hand[i, j].ToString();
                            break;
                        case Piece.Type.Knight:
                            str_hand[i] = str_hand[i] + "桂" + bt.Hand[i, j].ToString();
                            break;
                        case Piece.Type.Silver:
                            str_hand[i] = str_hand[i] + "銀" + bt.Hand[i, j].ToString();
                            break;
                        case Piece.Type.Gold:
                            str_hand[i] = str_hand[i] + "金" + bt.Hand[i, j].ToString();
                            break;
                        case Piece.Type.Bishop:
                            str_hand[i] = str_hand[i] + "角" + bt.Hand[i, j].ToString();
                            break;
                        case Piece.Type.Rook:
                            str_hand[i] = str_hand[i] + "飛" + bt.Hand[i, j].ToString();
                            break;
                    }
                }
            }

            BitBoard bb = new BitBoard();
            int sq;
            string[,] str_bb_piece = new string[Color.ColorNum, Piece.PieceNum];
            string[] str_bb_pawn_attacks = new string[Color.ColorNum];
            string[] str_bb_total_gold = new string[Color.ColorNum];
            string[] str_bb_bh = new string[Color.ColorNum];
            string[] str_bb_rd = new string[Color.ColorNum];
            string[] str_bb_hdk = new string[Color.ColorNum];
            string[] str_sq_king = new string[Color.ColorNum];


            str_bb_piece[0, 1] = "BB_BPAWN : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 1]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 1] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 2] = "BB_BLANCE : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 2]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 2] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 3] = "BB_BKNIGHT : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 3]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 3] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 4] = "BB_BSILVER : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 4]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 4] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 5] = "BB_BGOLD : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 5]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 5] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 6] = "BB_BBISHOP : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 6]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 6] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 7] = "BB_BROOK : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 7]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 7] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 8] = "BB_BKING : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 8]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 8] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 9] = "BB_BPRO_PAWN : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 9]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 9] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 10] = "BB_BPRO_LANCE : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 10]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 10] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 11] = "BB_BPRO_KNIGHT : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 11]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 11] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 12] = "BB_BPRO_SILVER : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 12]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 12] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 14] = "BB_BHORSE : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 14]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 14] += sq.ToString() + ", ";
            }

            str_bb_piece[0, 15] = "BB_BDRAGON : ";
            bb = BitBoard.Copy(bt.BB_Piece[0, 15]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[0, 15] += sq.ToString() + ", ";
            }

            str_bb_pawn_attacks[0] = "BB_BPAWN_ATTACKS : ";
            bb = BitBoard.Copy(bt.BB_PawnAttacks[0]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_pawn_attacks[0] += sq.ToString() + ", ";
            }

            str_bb_total_gold[0] = "BB_BTOTAL_GOLD : ";
            bb = BitBoard.Copy(bt.BB_Total_Gold[0]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_total_gold[0] += sq.ToString() + ", ";
            }

            str_bb_bh[0] = "BB_B_BH : ";
            bb = BitBoard.Copy(bt.BB_BH[0]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_bh[0] += sq.ToString() + ", ";
            }

            str_bb_rd[0] = "BB_B_RD : ";
            bb = BitBoard.Copy(bt.BB_RD[0]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_rd[0] += sq.ToString() + ", ";
            }

            str_bb_hdk[0] = "BB_B_HDK : ";
            bb = BitBoard.Copy(bt.BB_HDK[0]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_hdk[0] += sq.ToString() + ", ";
            }

            str_sq_king[0] = "SQ_BKING : " + bt.SQ_King[0].ToString();

            str_bb_piece[1, 1] = "BB_WPAWN : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 1]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 1] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 2] = "BB_WLANCE : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 2]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 2] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 3] = "BB_WKNIGHT : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 3]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 3] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 4] = "BB_WSILVER : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 4]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 4] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 5] = "BB_WGOLD : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 5]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 5] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 6] = "BB_WBISHOP : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 6]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 6] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 7] = "BB_WROOK : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 7]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 7] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 8] = "BB_WKING : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 8]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 8] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 9] = "BB_WPRO_PAWN : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 9]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 9] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 10] = "BB_WPRO_LANCE : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 10]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 10] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 11] = "BB_WPRO_KNIGHT : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 11]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 11] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 12] = "BB_WPRO_SILVER : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 12]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 12] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 14] = "BB_WHORSE : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 14]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 14] += sq.ToString() + ", ";
            }

            str_bb_piece[1, 15] = "BB_WDRAGON : ";
            bb = BitBoard.Copy(bt.BB_Piece[1, 15]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_piece[1, 15] += sq.ToString() + ", ";
            }

            str_bb_pawn_attacks[1] = "BB_WPAWN_ATTACKS : ";
            bb = BitBoard.Copy(bt.BB_PawnAttacks[1]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_pawn_attacks[1] += sq.ToString() + ", ";
            }

            str_bb_total_gold[1] = "BB_WTOTAL_GOLD : ";
            bb = BitBoard.Copy(bt.BB_Total_Gold[1]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_total_gold[1] += sq.ToString() + ", ";
            }

            str_bb_bh[1] = "BB_W_BH : ";
            bb = BitBoard.Copy(bt.BB_BH[1]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_bh[1] += sq.ToString() + ", ";
            }

            str_bb_rd[1] = "BB_W_RD : ";
            bb = BitBoard.Copy(bt.BB_RD[1]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_rd[1] += sq.ToString() + ", ";
            }

            str_bb_hdk[1] = "BB_W_HDK : ";
            bb = BitBoard.Copy(bt.BB_HDK[1]);
            while (BBTest(bb) != 0)
            {
                sq = LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                Xor(sq, ref bb);
                str_bb_hdk[1] += sq.ToString() + ", ";
            }

            str_sq_king[1] = "SQ_WKING : " + bt.SQ_King[1].ToString();

            string AppPath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            string FilePath = AppPath + "\\盤面.txt";
            Encoding ec = Encoding.UTF8;
            StreamWriter sw = new StreamWriter(FilePath, false, ec);
            sw.WriteLine(out_board);
            sw.WriteLine("先手持ち駒：" + str_hand[0]);
            sw.WriteLine("後手持ち駒：" + str_hand[1]);
            for (int c = 0; c < Color.ColorNum; c++)
            {
                for (int pc = 1; pc < Piece.PieceNum; pc++)
                {
                    if (pc == 13) { continue; }
                    sw.WriteLine(str_bb_piece[c, pc]);
                }
                sw.WriteLine(str_bb_pawn_attacks[c]);
                sw.WriteLine(str_bb_total_gold[c]);
                sw.WriteLine(str_bb_bh[c]);
                sw.WriteLine(str_bb_rd[c]);
                sw.WriteLine(str_bb_hdk[c]);
                sw.WriteLine(str_sq_king[c]);
            }
            sw.Close();
        }

        public static List<Move> TestMoveValid(ref List<Move> move_list, int color, BoardTree bt)
        {
            List<Move> illegal_move_list = new List<Move>();

            for (int i = 0; i < move_list.Count; i++)
            {
                int distance;
                bool is_illegal = false;
                Move move = move_list[i];

                if (move.From != 81)
                {
                    switch (move.PieceType)
                    {
                        case Piece.Type.Pawn:
                            if (color == 0)
                            {
                                distance = move.From - move.To;
                                if (distance != 9)
                                {
                                    is_illegal = true;
                                }

                                if (move.FlagPromo == 1 && move.To > 26)
                                {
                                    is_illegal = true;
                                }

                                if (move.FlagPromo == 0 && move.To < 27)
                                {
                                    is_illegal = true;
                                }
                            }
                            else
                            {
                                distance = move.From - move.To;
                                if (distance != -9)
                                {
                                    is_illegal = true;
                                }

                                if (move.FlagPromo == 1 && move.To < 54)
                                {
                                    is_illegal = true;
                                }

                                if (move.FlagPromo == 0 && move.To > 53)
                                {
                                    is_illegal = true;
                                }
                            }
                            break;
                        case Piece.Type.Lance:
                            if (color == 0)
                            {
                                distance = move.From - move.To;
                                switch (distance)
                                {
                                    case 9:
                                    case 18:
                                    case 27:
                                    case 36:
                                    case 45:
                                    case 54:
                                    case 63:
                                    case 72:
                                        break;
                                    default:
                                        is_illegal = true;
                                        break;
                                }
                                if (move.FlagPromo == 1 && move.To > 26)
                                {
                                    is_illegal = true;
                                }

                                if (move.FlagPromo == 0 && move.To < 9)
                                {
                                    is_illegal = true;
                                }
                            }
                            else
                            {
                                distance = move.To - move.From;
                                switch (distance)
                                {
                                    case 9:
                                    case 18:
                                    case 27:
                                    case 36:
                                    case 45:
                                    case 54:
                                    case 63:
                                    case 72:
                                        break;
                                    default:
                                        is_illegal = true;
                                        break;
                                }

                                if (move.FlagPromo == 1 && move.To < 54)
                                {
                                    is_illegal = true;
                                }

                                if (move.FlagPromo == 0 && move.To > 71)
                                {
                                    is_illegal = true;
                                }
                            }
                            break;
                        case Piece.Type.Knight:
                            if (color == 0)
                            {
                                distance = move.From - move.To;
                                switch (distance)
                                {
                                    case 17:
                                    case 19:
                                        break;
                                    default:
                                        is_illegal = true;
                                        break;
                                }
                                if (move.FlagPromo == 1 && move.To > 26)
                                {
                                    is_illegal = true;
                                }

                                if (move.FlagPromo == 0 && move.To < 18)
                                {
                                    is_illegal = true;
                                }
                            }
                            else
                            {
                                distance = move.To - move.From;
                                switch (distance)
                                {
                                    case 17:
                                    case 19:
                                        break;
                                    default:
                                        is_illegal = true;
                                        break;
                                }

                                if (move.FlagPromo == 1 && move.To < 54)
                                {
                                    is_illegal = true;
                                }

                                if (move.FlagPromo == 0 && move.To > 62)
                                {
                                    is_illegal = true;
                                }
                            }
                            break;
                        case Piece.Type.Silver:
                            if (color == 0)
                            {
                                distance = move.From - move.To;
                                switch (distance)
                                {
                                    case 8:
                                    case 9:
                                    case 10:
                                    case -8:
                                    case -10:
                                        break;
                                    default:
                                        is_illegal = true;
                                        break;
                                }
                                if (move.FlagPromo == 1)
                                {
                                    if (move.From > 26)
                                    {
                                        if (move.To > 26)
                                        {
                                            is_illegal = true;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                distance = move.From - move.To;
                                switch (distance)
                                {
                                    case -8:
                                    case -9:
                                    case -10:
                                    case 8:
                                    case 10:
                                        break;
                                    default:
                                        is_illegal = true;
                                        break;
                                }
                                if (move.FlagPromo == 1)
                                {
                                    if (move.From < 54)
                                    {
                                        if (move.To < 54)
                                        {
                                            is_illegal = true;
                                        }
                                    }
                                }
                            }
                            break;
                        case Piece.Type.Gold:
                        case Piece.Type.Pro_Pawn:
                        case Piece.Type.Pro_Lance:
                        case Piece.Type.Pro_Knight:
                        case Piece.Type.Pro_Silver:
                            if (color == 0)
                            {
                                distance = move.From - move.To;
                                switch (distance)
                                {
                                    case 8:
                                    case 9:
                                    case 10:
                                    case 1:
                                    case -1:
                                    case -9:
                                        break;
                                    default:
                                        is_illegal = true;
                                        break;
                                }
                                if (move.FlagPromo == 1)
                                {
                                    is_illegal = true;
                                }
                            }
                            else
                            {
                                distance = move.From - move.To;
                                switch (distance)
                                {
                                    case -8:
                                    case -9:
                                    case -10:
                                    case -1:
                                    case 1:
                                    case 9:
                                        break;
                                    default:
                                        is_illegal = true;
                                        break;
                                }
                                if (move.FlagPromo == 1)
                                {
                                    is_illegal = true;
                                }
                            }
                            break;
                        case Piece.Type.Bishop:
                            distance = Math.Abs(move.From - move.To);
                            switch (distance)
                            {
                                case 8:
                                case 16:
                                case 24:
                                case 32:
                                case 40:
                                case 48:
                                case 56:
                                case 64:
                                case 10:
                                case 20:
                                case 30:
                                case 50:
                                case 60:
                                case 70:
                                case 80:
                                    break;
                                default:
                                    is_illegal = true;
                                    break;
                            }
                            if (color == 0)
                            {
                                if (move.FlagPromo == 1)
                                {
                                    if (move.From > 26)
                                    {
                                        if (move.To > 26)
                                        {
                                            is_illegal = true;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (move.FlagPromo == 1)
                                {
                                    if (move.From < 54)
                                    {
                                        if (move.To < 54)
                                        {
                                            is_illegal = true;
                                        }
                                    }
                                }
                            }
                            break;
                        case Piece.Type.Rook:
                            distance = Math.Abs(move.From - move.To);
                            switch (distance)
                            {
                                case 0:
                                case 1:
                                case 2:
                                case 3:
                                case 4:
                                case 5:
                                case 6:
                                case 7:
                                case 8:
                                case 9:
                                case 18:
                                case 27:
                                case 36:
                                case 45:
                                case 54:
                                case 63:
                                case 72:
                                    break;
                                default:
                                    is_illegal = true;
                                    break;
                            }
                            if (color == 0)
                            {
                                if (move.FlagPromo == 1)
                                {
                                    if (move.From > 26)
                                    {
                                        if (move.To > 26)
                                        {
                                            is_illegal = true;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (move.FlagPromo == 1)
                                {
                                    if (move.From < 54)
                                    {
                                        if (move.To < 54)
                                        {
                                            is_illegal = true;
                                        }
                                    }
                                }
                            }
                            break;
                        case Piece.Type.King:
                            distance = move.From - move.To;
                            switch (distance)
                            {
                                case 8:
                                case 9:
                                case 10:
                                case 1:
                                case -1:
                                case -8:
                                case -9:
                                case -10:
                                    break;
                                default:
                                    is_illegal = true;
                                    break;
                            }
                            if (move.FlagPromo == 1)
                            {
                                is_illegal = true;
                            }
                            break;
                        case Piece.Type.Horse:
                            distance = Math.Abs(move.From - move.To);
                            switch (distance)
                            {
                                case 8:
                                case 16:
                                case 24:
                                case 32:
                                case 40:
                                case 48:
                                case 56:
                                case 64:
                                case 10:
                                case 20:
                                case 30:
                                case 50:
                                case 60:
                                case 70:
                                case 80:
                                case 9:
                                case 1:
                                case -1:
                                case -8:
                                case -9:
                                case -10:
                                    break;
                                default:
                                    is_illegal = true;
                                    break;
                            }
                            if (move.FlagPromo == 1)
                            {
                                is_illegal = true;
                            }
                            break;
                        case Piece.Type.Dragon:
                            distance = Math.Abs(move.From - move.To);
                            switch (distance)
                            {
                                case 0:
                                case 1:
                                case 2:
                                case 3:
                                case 4:
                                case 5:
                                case 6:
                                case 7:
                                case 8:
                                case 9:
                                case 18:
                                case 27:
                                case 36:
                                case 45:
                                case 54:
                                case 63:
                                case 72:
                                case 10:
                                case -1:
                                case -8:
                                case -9:
                                case -10:
                                    break;
                                default:
                                    is_illegal = true;
                                    break;
                            }
                            if (move.FlagPromo == 1)
                            {
                                is_illegal = true;
                            }
                            break;
                        case Piece.Type.Empty:
                            is_illegal = true;
                            break;
                    }
                }
                else
                {
                    if (bt.ShogiBoard[move.To] != (int)Piece.Type.Empty)
                    {
                        is_illegal = true;
                    }

                    switch(move.PieceType)
                    {
                        case Piece.Type.Pawn:
                        case Piece.Type.Lance:
                        case Piece.Type.Knight:
                        case Piece.Type.Silver:
                        case Piece.Type.Gold:
                        case Piece.Type.Bishop:
                        case Piece.Type.Rook:
                            break;
                        default:
                            is_illegal = true; 
                            break;
                    }

                    if(move.CapPiece != Piece.Type.Empty)
                    {
                        is_illegal = true;
                    }
                }

                if (is_illegal)
                    illegal_move_list.Add(move);
            }

            return illegal_move_list;
        }

        public static bool TestPosition(BoardTree bt)
        {
            bool is_error = true;

            for (int i = 0; i < ShogiCommon.NSquare; i++)
            {
                int piece = bt.ShogiBoard[i];
                if (piece > 0)
                {
                    BitBoard bb_piece = BitBoard.Copy(bt.BB_Piece[0, piece]);
                    while(BBTest(bb_piece) != 0)
                    {
                        int sq = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                        Xor(sq, ref bb_piece);
                        if (sq == i) { is_error = false; }
                    }

                    if (piece == 5 && !is_error)
                    {
                        bb_piece = BitBoard.Copy(bt.BB_Total_Gold[0]);
                        while (BBTest(bb_piece) != 0)
                        {
                            int sq = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                            Xor(sq, ref bb_piece);
                            if (sq == i) { is_error = false; }
                        }
                    }
                }
                else if (piece < 0)
                {
                    BitBoard bb_piece = BitBoard.Copy(bt.BB_Piece[1, -piece]);
                    while (BBTest(bb_piece) != 0)
                    {
                        int sq = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                        Xor(sq, ref bb_piece);
                        if (sq == i) { is_error = false; }
                    }

                    if (piece == 5 && !is_error)
                    {
                        bb_piece = BitBoard.Copy(bt.BB_Total_Gold[1]);
                        while (BBTest(bb_piece) != 0)
                        {
                            int sq = LastOne012(bb_piece.p[0], bb_piece.p[1], bb_piece.p[2]);
                            Xor(sq, ref bb_piece);
                            if (sq == i) { is_error = false; }
                        }
                    }
                }
            }

            return is_error;
        }

        public static bool TestPawnAttacks(BoardTree bt)
        {
            BitBoard bb_b_pawn = BitBoard.Copy(bt.BB_Piece[0, 1]);
            BitBoard bb_b_pawn_attacks = BitBoard.Copy(bt.BB_PawnAttacks[0]);
            BitBoard bb_w_pawn = BitBoard.Copy(bt.BB_Piece[1, 1]);
            BitBoard bb_w_pawn_attacks = BitBoard.Copy(bt.BB_PawnAttacks[1]);
            int popu_count_bb_b_pawn = PopuCount(bb_b_pawn.p[0]) + PopuCount(bb_b_pawn.p[1]) + PopuCount(bb_b_pawn.p[2]);
            int popu_count_bb_b_pawn_attacks = PopuCount(bb_b_pawn_attacks.p[0]) + PopuCount(bb_b_pawn_attacks.p[1]) + PopuCount(bb_b_pawn_attacks.p[2]);
            int popu_count_bb_w_pawn = PopuCount(bb_w_pawn.p[0]) + PopuCount(bb_w_pawn.p[1]) + PopuCount(bb_w_pawn.p[2]);
            int popu_count_bb_w_pawn_attacks = PopuCount(bb_w_pawn_attacks.p[0]) + PopuCount(bb_w_pawn_attacks.p[1]) + PopuCount(bb_w_pawn_attacks.p[2]);

            if (popu_count_bb_b_pawn != popu_count_bb_b_pawn_attacks) { return false; }
            if (popu_count_bb_w_pawn != popu_count_bb_w_pawn_attacks) { return false; }
            return true;
        }

        public static void TestMate()
        {
            Init();
            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);
            string str_sfen = "6s2/6R2/6Bk1/6p2/7N1/9/9/9/9 b GN 1";
            //string str_sfen = "6k2/5p1r1/5BN2/9/9/9/9/9/9 b BN 1";
            //string str_sfen = "5g2l/7s1/5B1kb/6R2/6P2/9/9/9/9 b G 1";
            //string str_sfen = "5sknl/3R5/4p1bp1/9/9/9/9/9/9 b RBS 1";
            //string str_sfen = "9/4+RB1k1/5+r1pp/5b3/9/9/9/9/9 b 2G 1";
            //string str_sfen = "7k1/4p3r/6S2/9/6rN1/9/9/9/9 b BNL 1"; // 13手詰め
            bt = SFEN.ParseSFEN(str_sfen);
            List<Move> pv = new List<Move>();
            int rest_depth = 5;
            //int pv_length = 0;
            int i, j, k;
            Mate mate = new Mate();

            mate.max_ply = 5;

            for (i = 0; i < 34; i++)
            {
                mate.move_cur[i] = new Move();
            }

            List<CheckMove> checkMoves = mate.GenRootCheckMoves(ref bt);
            mate.RootCheckMoves = checkMoves;

            if (mate.Offend(ref bt, (int)bt.RootColor, rest_depth, 1))
            {
                /*i = 1;
                while (Mate.move_cur[i].Value != 0)
                {
                    pv.Add(Mate.move_cur[i++]);
                }

                string str_pv = "";
                for (i = 0; i < pv.Count; i++)
                {
                    str_pv += CSA.Move2CSA(pv[i]);
                    if (i != pv.Count - 1)
                        str_pv += ", ";
                }
                Console.WriteLine(str_pv);*/
                List<List<Move>> l = mate.mate_proc;
                List<List<Move>> nl = mate.no_mate_proc;
                Move fm = mate.first_move;
                Move sm = mate.second_move;
                int index = 0;

                for (i = 0; i < l.Count; i++)
                {
                    if (l[i][0].Value == fm.Value && l[i][1].Value == sm.Value)
                    {
                        index = i;
                        break;
                    }
                }

                /*string str_pv = "";
                for (i = 0; i < rest_depth; i++)
                {
                    str_pv += CSA.Move2CSA(l[index][i]);
                    if (i != rest_depth - 1)
                        str_pv += ", ";
                }*/

                bool b = false;
                List<int> idxes = new List<int>();
                int a = 0;
                for (i = 0; i < l.Count; i++ )
                {
                    string s = i.ToString() + " / " + l.Count.ToString();
                    Console.WriteLine(s);
                    for (j = 0; j < nl.Count; j++)
                    {
                        b = false;
                        for (k = 0; k < nl[j].Count; k++)
                        {
                            if (l[i][k].Value != nl[j][k].Value)
                            {
                                b = true;
                                break;
                            }
                            else
                            {
                                int z = 0;
                            }
                        }
                        if (!b)
                        {
                            idxes.Add(i);
                        }
                    }
                }

                for (i = 0; i < l.Count; i++)
                {
                    if (idxes.Contains(i))
                        continue;
                    string str_pv = "";
                    for (j = 0; j < rest_depth; j++)
                    {
                        //str_pv += CSA.Move2CSA(l[index][i]);
                        str_pv += CSA.Move2CSA(l[i][j]);
                        if (j != rest_depth - 1)
                            str_pv += ", ";
                    }
                    Console.WriteLine(str_pv);
                }

                /*for (i = 0; i < rest_depth - 1; i++)
                {
                    List<Move> l_cur = Mate.mate_proc[i];
                    for (j = 0; j < rest_depth - 1; j++)
                    {
                        List<Move> nl_cur = Mate.no_mate_proc[j];
                        bool f = false;
                        for (k = 0; k < rest_depth - 1; k++)
                        {
                            if (l_cur[k].Value != nl_cur[k].Value)
                            {
                                f = true;
                            }
                        }
                        if (!f) { Mate.mate_proc.RemoveAt(i); }
                    }
                }*/

                //Console.WriteLine(str_pv);
            }
            int m = 0;
        }

        //
        public static void OutCheckTables()
        {
            StreamWriter[] sws = new StreamWriter[4];
            string[] file_names = new string[4];
            const string cm = ",";
            string str = "";
            file_names[0] = "check_table_lance.txt";
            file_names[1] = "check_table_knight.txt";
            file_names[2] = "check_table_silver.txt";
            file_names[3] = "check_table_gold.txt";
            for (int i = 0; i < 4; i++)
            {
                sws[i] = IO.OpenStreamWriter(file_names[i]);
            }
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 81; j++)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        switch (k)
                        {
                            case 0:
                                str = i.ToString() + cm;
                                str += j.ToString() + cm;
                                str += Check.ChkTbl[i, j].lance.p[0].ToString() + cm;
                                str += Check.ChkTbl[i, j].lance.p[1].ToString() + cm;
                                str += Check.ChkTbl[i, j].lance.p[2].ToString();
                                sws[k].WriteLine(str);
                                break;
                            case 1:
                                str = i.ToString() + cm;
                                str += j.ToString() + cm;
                                str += Check.ChkTbl[i, j].knight.p[0].ToString() + cm;
                                str += Check.ChkTbl[i, j].knight.p[1].ToString() + cm;
                                str += Check.ChkTbl[i, j].knight.p[2].ToString();
                                sws[k].WriteLine(str);
                                break;
                            case 2:
                                str = i.ToString() + cm;
                                str += j.ToString() + cm;
                                str += Check.ChkTbl[i, j].silver.p[0].ToString() + cm;
                                str += Check.ChkTbl[i, j].silver.p[1].ToString() + cm;
                                str += Check.ChkTbl[i, j].silver.p[2].ToString();
                                sws[k].WriteLine(str);
                                break;
                            case 3:
                                str = i.ToString() + cm;
                                str += j.ToString() + cm;
                                str += Check.ChkTbl[i, j].gold.p[0].ToString() + cm;
                                str += Check.ChkTbl[i, j].gold.p[1].ToString() + cm;
                                str += Check.ChkTbl[i, j].gold.p[2].ToString();
                                sws[k].WriteLine(str);
                                break;
                        }
                    }
                }
            }
            for (int i = 0; i < 4; i++)
            {
                sws[i].Close();
            }
        }
    }
}
