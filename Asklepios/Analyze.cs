using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.MakeMove;
using static Asklepios.AttackBitBoard;
using System.ComponentModel.Design;

namespace Asklepios
{
    internal class Analyze
    {
        public static void AnalyzeRecord(string read_file_name, string out_file_name, string out_eval_file_name, string out_pv_file_name, int task_num, int mate_task_num, string[] str_header, int thinking_time)
        {
            List<Record> records = new List<Record>();
            records = IO.ReadRecordFile(read_file_name);
            StreamWriter sw = IO.OpenStreamWriter(out_file_name);
            StreamWriter sw_eval = IO.OpenStreamWriter(out_eval_file_name);
            StreamWriter sw_pv = IO.OpenStreamWriter(out_pv_file_name);
            const int seq_length = 128;

            sw.WriteLine("対局日：" + str_header[0] + "\n");
            sw.WriteLine("棋戦名：" + str_header[1] + "\n");
            sw.WriteLine("先手：" + str_header[2] + "\n");
            sw.WriteLine("後手：" + str_header[3] + "\n");
            sw_eval.WriteLine("対局日：" + str_header[0] + "\n");
            sw_eval.WriteLine("棋戦名：" + str_header[1] + "\n");
            sw_eval.WriteLine("先手：" + str_header[2] + "\n");
            sw_eval.WriteLine("後手：" + str_header[3] + "\n");
            sw_pv.WriteLine("対局日：" + str_header[0] + "\n");
            sw_pv.WriteLine("棋戦名：" + str_header[1] + "\n");
            sw_pv.WriteLine("先手：" + str_header[2] + "\n");
            sw_pv.WriteLine("後手：" + str_header[3] + "\n");

            BoardTree bt = new BoardTree();
            BoardTreeAlloc(ref bt);
            BoardIni(ref bt);
            Controller controller = new Controller();
            Communicate cm = new Communicate();
            Mate3Ply m3p = new Mate3Ply();
            cm.Boot();

            try
            {
                int color_out = 0;
                int[] correct_count = new int[2];
                correct_count[0] = 0;
                correct_count[1] = 0;
                int[] correct_count_within3 = new int[2];
                correct_count_within3[0] = 0;
                correct_count_within3[1] = 0;
                int[] color_count = new int[2];
                color_count[0] = 0;
                color_count[1] = 0;
                string str_out = "";
                string str_out_eval = "";
                string str_out_pv = "";

                for (int i = 0; i < records[0].str_moves.Count(); i++)
                {
                    //if (i != 105)
                        //continue;
                    
                    BoardIni(ref bt);
                    int limit = i;
                    int color = 0;
                    string str_color;

                    if (color_out == 0)
                    {
                        str_color = "+";
                    }
                    else
                    {
                        str_color = "-";
                    }

                    str_out = "";
                    str_out_eval = "";
                    str_out_pv = "";

                    Console.WriteLine("ply = " + (i + 1).ToString());
                    str_out += "ply=" + (i + 1).ToString();
                    str_out += "   ";
                    str_out += "pro =";
                    str_out += str_color;
                    str_out += records[0].str_moves[i];
                    str_out += ",   ";
                    str_out += "com =";

                    for (int j = 0; j < limit; j++)
                    {
                        Move move = CSA.CSA2Move(bt, records[0].str_moves[j]);
                        Do(ref bt, color, move);
                        color ^= 1;
                    }

                    //Move[] three_moves = new Move[3];// ToDo: 初期化を確認
                    //bool is_mate = m3p.IsMateIn3Ply(ref bt, color, false, ref three_moves);

                    Move mate_move = new Move();
                    int iret = 0;
                    bool in_check = false;

                    if (IsAttacked(bt, bt.SQ_King[color], color) == 0)
                    {
                        iret = Mate1Ply.IsMateIn1Ply(ref bt, ref mate_move, color);
                    }
                    else
                    {
                        in_check = true;
                    }

                    // 詰みがあったら探索に回さない
                    if (iret == 0)
                    {
                        List<Move> m = new List<Move>();
                        List<float> f = new List<float>();
                        List<int> t = new List<int>();
                        List<Move> p = new List<Move>();

                        string[] str_sfen = new string[1];
                        str_sfen[0] = "p," + SFEN.ToSFEN(bt, color);
                        string str_moves = "";
                        int start_counter = 0;
                        if (limit > seq_length)
                        {
                            start_counter = limit - seq_length;
                        }
                        for (int j = start_counter; j < limit; j++)
                        {
                            str_moves += ":" + records[0].str_moves[j];
                        }
                        cm.ThrowRequest(str_sfen[0] + str_moves);
                        string s = cm.ReceiveResponse();
                        float[] policy_result = new float[1];
                        SetRootPos(ref bt, s, color, ref policy_result);// ToDo: Discovered Checkと王手放置の手を除去する処理を入れた方がよい
                        string str_mate_pv = "";
                        controller.ReceiveLoop(ref bt, cm, task_num, mate_task_num, ref policy_result, ref f, ref t, ref m, ref p, ref str_mate_pv, in_check, thinking_time);// fの大きい順に並べ替える

                        if (str_mate_pv == "")
                        {
                            List<Move> moves = new List<Move>();
                            List<int> trial_counts = new List<int>();
                            /*List<float> win_rates = new List<float>();
                            for (int j = 0; j < m.Count; j++)
                            {
                                float value_max = f.Max();
                                win_rates.Add(value_max);
                                int index = Array.IndexOf(f.ToArray(), value_max);
                                f[index] = float.MinValue;
                                moves.Add(m[index]);
                            }*/

                            for (int j = 0; j < m.Count; j++)
                            {
                                int value_max = t.Max();
                                if (value_max < 0)
                                    break;
                                trial_counts.Add(value_max);
                                int index = Array.IndexOf(t.ToArray(), value_max);
                                t[index] = int.MinValue;
                                moves.Add(m[index]);
                            }

                            for (int j = 0; j < moves.Count; j++)
                            {
                                string str_csa_move = CSA.Move2CSA(moves[j]);
                                if (j == 0)
                                {
                                    str_out += str_color;
                                    str_out += str_csa_move;
                                    str_out_eval = str_out;// 2023.11.27 追加
                                    str_out_pv = str_out;// 2023.11.29 追加
                                    str_out += "   ";
                                }

                                if (records[0].str_moves[i] == str_csa_move)
                                {
                                    str_out += "result= ○ ";
                                    correct_count_within3[color]++;
                                    if (j == 0)
                                        correct_count[color]++;
                                }
                                else
                                {
                                    str_out += "result= × ";
                                }

                                str_out += "  ";
                                str_out += "候補手" + (j + 1).ToString() + "：" + str_color + str_csa_move;
                                str_out += " 訪問回数 " + trial_counts[j].ToString();
                                //str_out += " 勝率 " + win_rates[j].ToString("P", CultureInfo.InvariantCulture);
                                if (j != moves.Count - 1)
                                    str_out += ",   ";
                            }
                            sw.WriteLine(str_out);
                            str_out_eval += "   ";
                            str_out_pv += "   ";

                            float win_rate_black = 0.0f;
                            float win_rate_white = 0.0f;
                            if (color == 0)
                            {
                                win_rate_black = f[0];
                                win_rate_white = 1.0f - f[0];
                            }
                            else
                            {
                                win_rate_black = 1.0f - f[0];
                                win_rate_white = f[0];
                            }
                            str_out_eval += "先手の勝率：" + win_rate_black.ToString("P", CultureInfo.InvariantCulture) + ", ";
                            str_out_eval += "後手の勝率：" + win_rate_white.ToString("P", CultureInfo.InvariantCulture);
                            sw_eval.WriteLine(str_out_eval);

                            int temp_color = color;
                            str_out_pv += "pv: ";
                            for (int j = 0; j < p.Count; j++)
                            {
                                if (j != 0)
                                    str_out_pv += ", ";
                                if (temp_color == 0)
                                {
                                    str_out_pv += "+";
                                }
                                else
                                {
                                    str_out_pv += "-";
                                }
                                str_out_pv += CSA.Move2CSA(p[j]);
                                temp_color ^= 1;
                            }
                            sw_pv.WriteLine(str_out_pv);
                        }
                        else
                        {
                            //str_mate_pv = str_mate_pv.Replace("+", "");
                            //str_mate_pv = str_mate_pv.Replace("-", "");
                            str_mate_pv = str_mate_pv.Replace(" ", "");
                            string[] str_temp = str_mate_pv.Split(',');
                            //str_out += str_color;
                            str_out += str_temp[0];
                            str_out += "   ";
                            str_out_eval = str_out;
                            str_out_pv = str_out;
                            //sw_pv.WriteLine(str_out_pv);
                            str_temp[0] = str_temp[0].Replace("+", "");
                            str_temp[0] = str_temp[0].Replace("-", "");
                            if (records[0].str_moves[i] == str_temp[0])
                            {
                                str_out += "result= ○ ";
                                correct_count_within3[color]++;
                                //if (j == 0)
                                correct_count[color]++;
                            }
                            else
                            {
                                str_out += "result= × ";
                            }
                            str_out += "詰みあり： " + str_mate_pv;
                            sw.WriteLine(str_out);
                            sw_pv.WriteLine(str_out);
                            float win_rate_black = 0.0f;
                            float win_rate_white = 0.0f;
                            if (color == 0)
                            {
                                win_rate_black = 1.0f;
                                win_rate_white = 0.0f;
                            }
                            else
                            {
                                win_rate_black = 0.0f;
                                win_rate_white = 1.0f;
                            }
                            str_out_eval += "先手の勝率：" + win_rate_black.ToString("P", CultureInfo.InvariantCulture) + ", ";
                            str_out_eval += "後手の勝率：" + win_rate_white.ToString("P", CultureInfo.InvariantCulture);
                            sw_eval.WriteLine(str_out_eval);
                        }

                    }

                    color_count[color]++;
                    color_out ^= 1;
                }

                float v;

                str_out = "\n";
                str_out += "先手一致率：" + correct_count[0].ToString() + " / " + color_count[0].ToString();
                v = (float)((float)correct_count[0] / (float)color_count[0]);
                str_out += " " + v.ToString("P", CultureInfo.InvariantCulture);
                str_out += "\n\n";
                str_out += "後手一致率：" + correct_count[1].ToString() + " / " + color_count[1].ToString();
                v = (float)((float)correct_count[1] / (float)color_count[1]);
                str_out += " " + v.ToString("P", CultureInfo.InvariantCulture);
                str_out += "\n\n";
                str_out += "全体一致率： " + (correct_count[0] + correct_count[1]).ToString() + " / " + records[0].str_moves.Count().ToString();
                v = (float)((float)(correct_count[0] + correct_count[1]) / (float)records[0].str_moves.Count());
                str_out += " " + v.ToString("P", CultureInfo.InvariantCulture);
                str_out += "\n\n";
                str_out += "候補手3位以内の率： " + (correct_count_within3[0] + correct_count_within3[1]).ToString() + " / " + records[0].str_moves.Count().ToString();
                v = (float)((float)(correct_count_within3[0] + correct_count_within3[1]) / (float)records[0].str_moves.Count());
                str_out += " " + v.ToString("P", CultureInfo.InvariantCulture);
                str_out += "\n\n";
                str_out += "解析解析エンジン名：Ariadnē Ver.1.0.0";
                sw.WriteLine(str_out);
                str_out_eval = "\n";
                str_out_eval += "解析解析エンジン名：Ariadnē Ver.1.0.0";// 2024.9.12 変更
                str_out_pv = "解析解析エンジン名：Ariadnē Ver.1.0.0";
                sw_eval.WriteLine(str_out_eval);
                sw_pv.Write("\n");
                sw_pv.WriteLine(str_out_pv);
            }
            catch (Exception ex)
            {
                sw.WriteLine (ex.ToString());
            }

            cm.Quit();
            cm.Dispose();
            sw.Close();
            sw_eval.Close();
            sw_pv.Close();
        }
    }
}
