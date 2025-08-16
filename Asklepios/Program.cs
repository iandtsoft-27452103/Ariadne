using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq.Expressions;
using System.Runtime.Intrinsics;
using System.Text;
using static Asklepios.Board;

namespace Asklepios
{
    class Program
    {
        // コマンドライン引数
        // (0) 対局日 YYYY/MM/DD
        // (1) 棋戦名 ex) 第72回NHK杯1回戦
        // (2) 先手の棋士名
        // (3) 後手の棋士名
        // (4) 読み取り用棋譜ファイル名
        // (5) 出力用テキストファイル名
        // (6) 探索用タスク数
        // (7) 思考時間 (秒)
        static void Main(string[] args)
        {
            //Console.WriteLine("Hello World!");
            Hash.IniRand(5489U);
            Hash.IniRandomTable();
            AttackBitBoard.IniTables();
            AttackBitBoard.IniLongAttacks();
            Check.Init();

            //Test.OutCheckTables();
            //return;

            string[] str_header = new string[4];
            /*str_header[0] = "2022/04/03";
            str_header[1] = "第72回NHK杯1回戦";
            str_header[2] = "木村一基九段";
            str_header[3] = "黒田尭之五段";*/
            str_header[0] = args[0];
            str_header[1] = args[1];
            str_header[2] = args[2];
            str_header[3] = args[3];
            string[] temp = args[5].Split('.');
            string out_eval_file_name = temp[0] + "_eval." + temp[1];
            string out_pv_file_name = temp[0] + "_pv." + temp[1];
            //Analyze.AnalyzeRecord("20220403_nhk_hai.txt", "analyze_result.txt", 10, 2, str_header);
            // 2024/06/25 詰み探索のタスク数を1にロールバックした。
            Analyze.AnalyzeRecord(args[4], args[5], out_eval_file_name, out_pv_file_name, int.Parse(args[6]), 1, str_header, int.Parse(args[7]));
            return;
        }
    }
}