using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class IO
    {
        public static List<Record> ReadRecordFile(string file_name)
        {
            string AppPath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            string FilePath = AppPath + "\\" + file_name;
            List<Record> records = new List<Record>();
            string line;
            StreamReader sr = new StreamReader(FilePath, Encoding.UTF8);
            while((line = sr.ReadLine()) != null)
            {
                Record record = new Record();
                string[] s = line.Split(',');
                if (s[0] == "B")
                {
                    record.winner = 0;
                }
                else if (s[0] == "W")
                {
                    record.winner = 1;
                }
                else
                {
                    record.winner = 2;// 引き分け
                }

                record.ply = int.Parse(s[1]);
                record.str_moves = new string[s.Length - 2];

                for (int i = 2; i < s.Length; i++)
                {
                    record.str_moves[i - 2] = s[i];
                }

                records.Add(record);
            }
            sr.Close();
            return records;
        }

        public static List<string> ReadTestFile(string file_name, ref List<string> comments)
        {
            string AppPath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            string FilePath = AppPath + "\\" + file_name;
            List<string> str_return = new List<string>();
            string line;
            int flag = 0;
            StreamReader sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                if (flag == 0)
                {
                    comments.Add(line);
                }
                else
                {
                    str_return.Add(line);
                }
                
                flag ^= 1;
            }
            sr.Close();
            return str_return;
        }

        public static StreamWriter OpenStreamWriter(string file_name)
        {
            string AppPath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            string FilePath = AppPath + "\\" + file_name;
            Encoding enc = new UTF8Encoding(false);// ここでEncodingを指定しないとBOMが入ってしまう。
            StreamWriter sw = new StreamWriter(FilePath, false, enc);
            return sw;
        }
    }
}
