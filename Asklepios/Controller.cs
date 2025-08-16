using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;

namespace Asklepios
{
    internal class Controller
    {
        //public MoveFeature mf;
        public Mate3Ply m3p;
        public TT tt;

        public Controller()
        {
            //mf = new MoveFeature();
            //mf.LoadFeature();
            //mf.ProgressInit();
            //Console.WriteLine(mf.move_feature.Max());
            m3p = new Mate3Ply();
            tt = new TT();
        }

        public void ReceiveLoop(ref BoardTree bt, Communicate cm, int task_num, int mate_task_num, ref float[] root_policy_result, ref List<float> f, ref List<int> t, ref List<Move> m, ref List<Move> p, ref string str_mate_pv, bool in_check, int thinking_time)
        {
            MCTS[] mcts_array = new MCTS[task_num];
            Mate[] mate_array = new Mate[mate_task_num];
            bool[] is_mate = new bool[mate_task_num];
            Task[] tasks = new Task[task_num];
            Task[] mate_tasks = new Task[mate_task_num];

            List<CheckMove> checkMoves = new List<CheckMove>();

            if (!in_check)
            {
                for (int i = 0; i < mate_task_num; i++)
                {
                    mate_array[i] = new Mate();
                    /*for (int j = 0; j < 34; j++)
                    {
                        mate_array[i].move_cur[j] = new Move();
                    }*/
                }             

                checkMoves = mate_array[0].GenRootCheckMoves(ref bt);
            }

            try
            {
                for (int i = 0; i < task_num; i++)
                {
                    mcts_array[i] = new MCTS();
                    mcts_array[i].BTree = bt.DeepCopy(bt, true);
                    //mcts_array[i].mf = mf;
                    //mcts_array[i].m3p = m3p;
                    mcts_array[i].m3p = new Mate3Ply();
                    //mcts_array[i].tt = tt; // ToDo: 探索終了後のTTを保存しておくと良いかも
                    mcts_array[i].is_abort = false;
                    mcts_array[i].is_finished = false;
                    mcts_array[i].RootOutput = root_policy_result;
                    //mcts_array[i].SearchTimeLimit = 30000;
                    mcts_array[i].SearchTimeLimit = thinking_time * 1000;
                    mcts_array[i].sw.Start();                 
                    tasks[i] = Task.Run(mcts_array[i].Root);
                }

                if (checkMoves.Count > 0)
                {
                    if (checkMoves.Count <= 10 || mate_array.Length == 1)
                    {
                        mate_task_num = 1;
                        mate_array[0].BTree = bt.DeepCopy(bt, true);
                        mate_array[0].RootCheckMoves = checkMoves;
                        mate_array[0].is_mate_root = false;
                        mate_tasks[0] = Task.Run(mate_array[0].MateSearchWrapper);
                    }
                    else
                    {
                        int[] limit = new int[2];
                        if ((checkMoves.Count % 2) == 0)
                        {
                            limit[0] = limit[1] = checkMoves.Count / 2;
                        }
                        else
                        {
                            limit[0] = (checkMoves.Count / 2) + 1;
                            limit[1] = checkMoves.Count / 2;
                        }

                        int index = 0;
                        for (int i = 0; i < limit.Length; i++)
                        {
                            mate_array[i].BTree = bt.DeepCopy(bt, true);
                            for (int j = 0; j < limit[i]; j++)
                            {
                                mate_array[i].RootCheckMoves.Add(checkMoves[index++]);
                            }
                            mate_array[i].is_mate_root = false;
                            mate_tasks[i] = Task.Run(mate_array[i].MateSearchWrapper);
                        }
                    }
                }

                while (true)
                {
                    if(IsCompleted(ref tasks, task_num))
                    {
                        if (checkMoves.Count > 0)
                        {
                            for (int i = 0; i < mate_task_num; i++)
                            {
                                mate_array[i].is_abort = true;
                                while (true)
                                {
                                    if (mate_tasks[i].Status == TaskStatus.RanToCompletion)
                                        break;
                                }
                            }
                        }
                        break;
                    }

                    if (checkMoves.Count > 0 && IsMateSearchFinished(ref mate_tasks, ref mate_array, mate_task_num, ref str_mate_pv))
                    {
                        if (str_mate_pv != "")
                        {
                            for (int i = 0; i < task_num; i++)
                            {
                                mcts_array[i].is_abort = true;
                                while(true)
                                {
                                    if (tasks[i].Status == TaskStatus.RanToCompletion)
                                        break;
                                }
                            }
                            break;
                        }
                    }

                    ConfirmQueue(mcts_array, task_num, cm);
                }

                if (!in_check)
                {
                    for (int i = 0; i < mate_task_num; i++)
                    {
                        if (mate_array[i].is_mate_root)
                        {
                            if (mate_array[i].is_mate_root)
                            {
                                if (i == 0)
                                {
                                    str_mate_pv = mate_array[i].root_str_pv;
                                }
                                else
                                {
                                    if (str_mate_pv.Length > mate_array[i].root_str_pv.Length)
                                    {
                                        str_mate_pv = mate_array[i].root_str_pv;
                                    }
                                }
                            }
                        }
                    }
                }

                if (str_mate_pv != "")
                    return;

                List<Move> moves = new List<Move>();
                List<float> win_rates = new List<float>();
                List<int> trial_counts = new List<int>();
                List<Move> pv = new List<Move>();

                moves = mcts_array[0].TotalParam(mcts_array, ref win_rates, ref trial_counts, ref pv, true);

                m = moves;
                f = win_rates;
                t = trial_counts;
                p = pv;
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
        }

        private bool IsCompleted(ref Task[] tasks, int task_num)
        {
            bool iret = false;
            int counter = 0;
            for (int i = 0; i < task_num; i++)
            {
                if (tasks[i].Status == TaskStatus.RanToCompletion)
                    counter++;
            }

            if (counter == task_num)
                iret = true;

            return iret;
        }

        private bool IsMateSearchFinished(ref Task[] mate_tasks, ref Mate[] mate_array, int mate_task_num, ref string str_pv)
        {
            bool iret = false;
            int counter = 0;
            for (int i = 0; i < mate_task_num; i++)
            {
                if (mate_tasks[i].Status == TaskStatus.RanToCompletion)
                    counter++;
            }

            // ToDo: 2個のタスクで別々に詰みを見つけた場合、短い方を選択するようにする
            // →修正したが未テスト
            if (counter > 0)
            {
                iret = true;
                for (int i = 0; i < mate_task_num; i++)
                {
                    if (mate_array[i].is_mate_root)
                    {
                        if (i == 0)
                        {
                            str_pv = mate_array[i].root_str_pv;
                        }
                        else
                        {
                            if (str_pv.Length == 0 || str_pv.Length > mate_array[i].root_str_pv.Length)
                            {
                                str_pv = mate_array[i].root_str_pv;
                            }
                        }
                    }
                }
            }

            return iret;
        }

        private void ConfirmQueue(MCTS[] mcts_array, int task_num, Communicate com)
        {

            try
            {
                string[] str = new string[task_num];
                List<int> index_array = new List<int>();
                for (int i = 0; i < task_num; i++)
                {
                    if (mcts_array[i].queue_to_main_thread.Count != 0)
                    {
                        str[i] = mcts_array[i].queue_to_main_thread.Dequeue();
                        index_array.Add(i);
                    }
                }

                List<string> str_p = new List<string>();
                List<int> p_index_array = new List<int>();
                List<string> str_v = new List<string>();
                List<int> v_index_array = new List<int>();

                for (int i = 0; i < str.Length; i++)
                {
                    if (str[i] != "" && str[i] != null)
                    {
                        string[] s = str[i].Split(',');
                        if (s[0] == "p")
                        {
                            str_p.Add(s[1]);
                            p_index_array.Add(i);
                        }
                        else
                        {
                            str_v.Add(s[1]);
                            v_index_array.Add(i);
                        }
                    }
                }

                // ToDo: PythonにPolicy NetworkまたはValue Networkの実行を投げる処理を書く
                if (str_p.Count != 0)
                {
                    string s = "p,";
                    for (int i = 0; i < str_p.Count; i++)
                    {
                        s += str_p[i];
                        if (i != str_p.Count - 1)
                            s += ",";
                    }
                    com.ThrowRequest(s);
                    string s_ret = com.ReceiveResponse();
                    string[] s_array = s_ret.Split(":");
                    for (int i = 0; i < str_p.Count; i++)
                        mcts_array[p_index_array[i]].queue_from_main_thread.Enqueue(s_array[i]);

                    for (int i = 0; i < str_p.Count; i++)
                        while (mcts_array[p_index_array[i]].queue_from_main_thread.Count > 0) { Thread.Sleep(1); }
                }

                if (str_v.Count != 0)
                {
                    string s = "v,";
                    for (int i = 0; i < str_v.Count; i++)
                    {
                        s += str_v[i];
                        if (i != str_v.Count - 1)
                            s += ",";
                    }
                    com.ThrowRequest(s);
                    string s_ret = com.ReceiveResponse();
                    string[] s_array = s_ret.Split(",");
                    for (int i = 0; i < str_v.Count; i++)
                        mcts_array[v_index_array[i]].queue_from_main_thread.Enqueue(s_array[i]);
                    for (int i = 0; i < str_v.Count; i++)
                        while (mcts_array[v_index_array[i]].queue_from_main_thread.Count > 0) { Thread.Sleep(1); }
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
        }
    }
}
