using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.MakeMove;

namespace Asklepios
{
    internal class MoveFeature
    {
        const int feature_size = 67108864;
        const int progress_size = 512;
        public uint[] move_feature = new uint[feature_size];
        public uint[] progress_table = new uint[progress_size];

        public void MakeFeature()
        {
            string AppPath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            const string move_feature_file_name = "move_feature.bin";
            const int file_count = 17;
            string line;
            int color, ply, progress;
            uint hash; // int でも良かったか？
            //uint[,] temp_move_feature = new uint[8, 8388608];

            for (int i = 0; i < file_count; i++)
            {
                string record_file_name = AppPath + "\\records" + i.ToString() + ".txt";
                StreamReader sr = new StreamReader(record_file_name, Encoding.UTF8);

                int row_number = 1;
                while ((line = sr.ReadLine()) != null)
                {
                    color = 0;
                    ply = 0;
                    progress = new int();
                    string[] s = line.Split(',');
                    for (int j = 2; j < s.Length; j++)
                    {
                        BoardTree bt = new BoardTree();
                        BoardTreeAlloc(ref bt);
                        BoardIni(ref bt);
                        string str_csa = s[j];
                        Move move = CSA.CSA2Move(bt, str_csa);
                        Do(ref bt, color, move);

                        if (ply < 16)
                        {
                            progress = 0;
                        }
                        else if (ply >= 16 && ply < 32)
                        {
                            progress = 1;
                        }
                        else if (ply >= 32 && ply < 48)
                        {
                            progress = 2;
                        }
                        else if (ply >= 48 && ply < 64)
                        {
                            progress = 3;
                        }
                        else if (ply >= 64 && ply < 80)
                        {
                            progress = 4;
                        }
                        else if (ply >= 80 && ply < 96)
                        {
                            progress = 5;
                        }
                        else if (ply >= 96 && ply < 112)
                        {
                            progress = 6;
                        }
                        else if (ply >= 112)
                        {
                            progress = 7;
                        }

                        //temp_move_feature[progress, move.Value]++;

                        hash = (uint)(progress << 23) | (move.Value);
                        move_feature[hash] += 1;
                        color ^= 1;
                        ply++;
                    }
                    string str_msg = "File No = " + (i + 1).ToString() + ", Row Number = " + row_number.ToString();
                    Console.WriteLine(str_msg);
                    row_number++;
                }

                //Console.WriteLine(i.ToString());
                //BinaryWriter bw = new BinaryWriter(File.OpenWrite(move_feature_file_name));
            }

            Console.WriteLine("parameter file output...");
            BinaryWriter bw = new BinaryWriter(File.OpenWrite(move_feature_file_name));
            for (int i = 0; i < move_feature.Length; i++)
            {
                bw.Write(move_feature[i]);
            }
            bw.Close();
            /*uint x;
            for (int i  = 0; i < 8; i++)
            {
                x = 0;
                for (int j = 0; j < 8388608; j++)
                {
                    x += temp_move_feature[i, j];
                    //if (temp_move_feature[i, j] != 0)
                        //Console.WriteLine(temp_move_feature[i, j].ToString());
                }

                for (int j = 0; j < 8388608; j++)
                {
                    hash = (uint)i << 23 | (uint)j;
                    move_feature[hash] = (temp_move_feature[i, j] / x);
                    float f = (float)((float)temp_move_feature[i, j] / (float)x) * 100;
                    if (f > 1)
                    {
                        Console.WriteLine(f.ToString());
                    }                        
                }
            }*/
        }

        public void LoadFeature()
        {
            string AppPath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            const string move_feature_file_name = "move_feature.bin";
            string file_path = AppPath + "\\" + move_feature_file_name;
            BinaryReader br = new BinaryReader(File.OpenRead(file_path));
            for (int i = 0; i < feature_size; i++)
            {
                move_feature[i] = br.ReadUInt32();
            }
            br.Close();
        }

        public void ProgressInit()
        {
            for (int i = 0; i < progress_size; i++)
            {
                if (i < 16)
                {
                    progress_table[i] = 0;
                }
                else if (i >= 16 && i < 32)
                {
                    progress_table[i] = 1;
                }
                else if (i >= 32 && i < 48)
                {
                    progress_table[i] = 2;
                }
                else if (i >= 48 && i < 64)
                {
                    progress_table[i] = 3;
                }
                else if (i >= 64 && i < 80)
                {
                    progress_table[i] = 4;
                }
                else if (i >= 80 && i < 96)
                {
                    progress_table[i] = 5;
                }
                else if (i >= 96 && i < 112)
                {
                    progress_table[i] = 6;
                }
                else if (i >= 112)
                {
                    progress_table[i] = 7;
                }
            }
        }
    }
}
