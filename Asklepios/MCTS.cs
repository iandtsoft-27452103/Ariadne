using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.AttackBitBoard;
using static Asklepios.MakeMove;
using static Asklepios.DeclarationWin;
using static Asklepios.ShogiCommon;
using static Asklepios.ShogiCommon.Square;
using System.Diagnostics;

namespace Asklepios
{
    internal class MCTS
    {
        public BoardTree BTree;
        public List<Node> NodeList = new List<Node>();
        public int PlayOutCount;
        public float[] RootOutput;
        //public Move[] RootMoves;
        public const float value_lambda = 0.5f;
        public const int nthr = 30;
        //public const int search_time_limit = 30000;
        //start_timeを後で追加する
        //public int RootPly;
        public int TaskNumber;
        public long SearchTimeLimit;// ミリ秒で指定する
        public const long TimeBuffer = 500;
        public Stopwatch sw = new Stopwatch();
        public Queue<string> queue_from_main_thread = new Queue<string>();
        public Queue<string> queue_to_main_thread = new Queue<string>();
        //public MoveFeature mf;
        public Mate3Ply m3p = new Mate3Ply();
        public TT tt = new TT();
        public bool is_abort;
        public bool is_finished;

        public List<Move>TotalParam(MCTS[] mcts_tasks, ref List<float> f, ref List<int> t, ref List<Move> p, bool use_draw)// ToDo:勝率のリストも返すように変更する
        {
            int value_max, max_index, limit;
            List<int> trial_count_array = new List<int>();
            List<float> win_rate_array = new List<float>();
            List<Move> return_moves = new List<Move>();
            List<float> return_win_rate_array = new List<float>();
            List<int> return_trial_count_array = new List<int>();
            List<Move> pv = new List<Move>();

            switch (BTree.RootMoves.Length)
            {
                case 1:
                    limit = 1;
                    break;
                case 2:
                    limit = 2;
                    break;
                default:
                    limit = 3;
                    break;
            }

            if (mcts_tasks.Length > 1)
            {
                for (int i = 1; i < mcts_tasks.Length; i++)
                {
                    for (int j = 0; j < BTree.RootMoves.Length; j++)
                    {
                        mcts_tasks[0].NodeList[j + 1].TrialCount += mcts_tasks[i].NodeList[j + 1].TrialCount;
                        mcts_tasks[0].NodeList[j + 1].WinCount += mcts_tasks[i].NodeList[j + 1].WinCount;
                        mcts_tasks[0].NodeList[j + 1].DrawCount += mcts_tasks[i].NodeList[j + 1].DrawCount;
                        mcts_tasks[0].NodeList[j + 1].LostCount += mcts_tasks[i].NodeList[j + 1].LostCount;
                        mcts_tasks[0].NodeList[j + 1].EvalCount += mcts_tasks[i].NodeList[j + 1].EvalCount;// 2023.12.13 追加
                        mcts_tasks[0].NodeList[j + 1].WinRateSum += mcts_tasks[i].NodeList[j + 1].WinRateSum;// 2023.12.13 追加
                    }
                }
            }

            for (int i = 0; i < BTree.RootMoves.Length; i++)
            {
                trial_count_array.Add(mcts_tasks[0].NodeList[i + 1].TrialCount);
            }
            
            for (int i = 0; i < BTree.RootMoves.Length; i++)
            {
                //if (use_draw)
                {
                    if (trial_count_array[i] == 0)
                    {
                        win_rate_array.Add(0);
                    }
                    else
                    {
                        //win_rate_array.Add((mcts_tasks[0].NodeList[i + 1].WinCount + (float)(mcts_tasks[0].NodeList[i + 1].DrawCount * 0.5)) / trial_count_array[i]);
                        win_rate_array.Add((float)(mcts_tasks[0].NodeList[i + 1].WinRateSum / (float)(mcts_tasks[0].NodeList[i + 1].EvalCount)));
                    }
                }
                /*else
                {
                    if (trial_count_array[i] == 0)
                    {
                        win_rate_array[i] = 0;
                    }
                    else
                    {
                        //win_rate_array.Add(mcts_tasks[0].NodeList[i + 1].WinCount / trial_count_array[i]);
                    }
                }*/
            }

            // Value Networkの結果を使う場合
            /*for (int i = 0; i < BTree.RootMoves.Length; i++)
            {
                win_rate_array.Add(mcs_tasks[0].NodeList[i + 1].WinRateSum / mcs_tasks[0].NodeList[i + 1].EvalCount);
            }*/

            int max_node_index = 0;
            for (int i = 0; i < limit; i++)
            {
                value_max = trial_count_array.Max();
                max_index = Array.IndexOf(trial_count_array.ToArray(), value_max);
                if (i == 0)
                    max_node_index = max_index + 1;
                return_trial_count_array.Add(trial_count_array[max_index]);
                trial_count_array[max_index] = int.MinValue;
                return_moves.Add(BTree.RootMoves[max_index]);
                return_win_rate_array.Add(win_rate_array[max_index]);
                /*if (BTree.RootMoves.Length == 1)
                {
                    return_moves.Add(BTree.RootMoves[max_index]);
                    return_win_rate_array.Add(win_rate_array[max_index]);
                }*/
            }

            //Console.WriteLine("start_calc_pv");
            // 2023.11.27追加 pvの編集
            Move best_move = return_moves[0];
            pv.Add(best_move);
            Node best_move_node = mcts_tasks[0].NodeList[max_node_index];
            Node current_node = new Node();
            Node node = new Node();
            int depth = 1;
            int index, mi, mv;
            // ※デバッグ中
            if (best_move_node.ChildIndexes.Count > 0)
            {
                int[] temp_trial_count_array = new int[best_move_node.ChildIndexes.Count];
                for (int i = 0; i < best_move_node.ChildIndexes.Count; i++)
                {
                    index = best_move_node.ChildIndexes[i];
                    current_node = mcts_tasks[0].NodeList[index];
                    temp_trial_count_array[i] = current_node.TrialCount;
                }
                mv = temp_trial_count_array.Max();
                mi = Array.IndexOf(temp_trial_count_array, mv) + 1;
                //pv.Add(mcts_tasks[0].NodeList[mi].move);
                node = mcts_tasks[0].NodeList[mi];
                while (true)
                {
                    if (node.ChildIndexes.Count == 0)
                        break;
                    temp_trial_count_array = new int[node.ChildIndexes.Count];
                    for (int i = 0; i < node.ChildIndexes.Count; i++)
                    {
                        index = node.ChildIndexes[i];
                        current_node = mcts_tasks[0].NodeList[index];
                        temp_trial_count_array[i] = current_node.TrialCount;
                    }
                    mv = temp_trial_count_array.Max();
                    mi = Array.IndexOf(temp_trial_count_array, mv);
                    pv.Add(mcts_tasks[0].NodeList[node.ChildIndexes[mi]].move);
                    node = mcts_tasks[0].NodeList[node.ChildIndexes[mi]];
                    depth++;
                }
            }

            f = return_win_rate_array;
            t = return_trial_count_array;
            p = pv;
            //Console.WriteLine("end_total_param");

            return return_moves;
        }

        public void GenRootMoves()
        {
            List<Move> move_list = new List<Move>();
            if (IsAttacked(BTree, BTree.SQ_King[(int)BTree.RootColor], (int)BTree.RootColor) == 0)
            {
                GenCap.Generate(BTree, ref move_list, (int)BTree.RootColor);
                GenNoCap.Generate(BTree, ref move_list, (int)BTree.RootColor);
                GenDrop.Generate(ref BTree, ref move_list, (int)BTree.RootColor);
            }
            else
            {
                GenEvasion.Generate(ref BTree,ref move_list, (int)BTree.RootColor);
            }

            BTree.RootMoves = move_list.ToArray();// 玉を取る非合法手とDiscoverd Checkの手を除去する処理は入っていない。
        }

        public void Root()
        {
            float[] result_array = new float[BTree.RootMoves.Length];

            //Console.WriteLine("root_0");

            NodeList.Clear();

            Node root_node = new Node();
            NodeList.Add(root_node);

            //Console.WriteLine("root_a");

            for (int i = 0; i < BTree.RootMoves.Length; i++)// 要対応：RootMovesは全合法手を展開するか否か？
            {
                Node n = new Node();
                n.color = (int)BTree.RootColor;
                Move m = BTree.RootMoves[i];
                n.ParentIndex = 0;
                n.ThisIndex = i + 1;
                n.move = m;
                NodeList[0].ChildIndexes.Add(i + 1);
                n.PolicyResult = RootOutput[i];
                result_array[i] = RootOutput[i];
                NodeList.Add(n);
            }

            // Softmax関数はPython側でかけた方がよさそう
            while (true)
            {
                long elapsed = sw.ElapsedMilliseconds;
                if (SearchTimeLimit < elapsed)
                    break;

                if (is_abort)
                    break;

                int t, i;
                float[] ucb1_array = new float[BTree.RootMoves.Length];
                t = 0;
                i = 1;
                while (i < BTree.RootMoves.Length)
                    t += NodeList[i++].PlayoutCount;
                i = 1;
                while (i < BTree.RootMoves.Length)
                {
                    float u = NodeList[i].PolicyResult * (float)Math.Sqrt(t) / (NodeList[i].PlayoutCount + 1);
                    float q = 0.0F;
                    if (NodeList[i].EvalCount > 0 && NodeList[i].PlayoutCount > 0)
                        q = (float)((1 - value_lambda) * (NodeList[i].WinRateSum / NodeList[i].EvalCount) + value_lambda * (NodeList[i].WinCount / NodeList[i].PlayoutCount));
                    ucb1_array[i - 1] = u + q;
                    i++;
                }

                int max_index = Array.IndexOf(ucb1_array, ucb1_array.Max()) + 1;
                Do(ref BTree, (int)BTree.RootColor, BTree.RootMoves[max_index - 1]);

                // 指した手によって自玉がDiscovered Checkになってしまった場合
                /*if (IsAttacked(BTree, BTree.SQ_King[(int)BTree.RootColor], (int)BTree.RootColor) != 0)
                {
                    UnDo(ref BTree, (int)BTree.RootColor, BTree.RootMoves[max_index - 1]);
                    continue;
                }*/

                // 玉を取ってしまう非合法手の場合
                /*if (BTree.RootMoves[max_index - 1].CapPiece == Piece.Type.King)
                {
                    UnDo(ref BTree, (int)BTree.RootColor, BTree.RootMoves[max_index - 1]);
                    continue;
                }*/

                if (NodeList[max_index].IsLeaf)
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < (elapsed + TimeBuffer))
                        break;
                    if (is_abort)
                        break;

                    if (NodeList[max_index].TrialCount >= nthr)
                    {
                        //Console.WriteLine("root_loop0");
                        ExpandNode(((int)BTree.RootColor ^ 1), max_index, BTree.ply + 1);
                        if (is_abort)
                            break;
                        //Console.WriteLine("root_loop1");
                    }
                    else
                    {
                        //Console.WriteLine("root_loop2");
                        BoardTree bt = new BoardTree();
                        BoardTreeAlloc(ref bt);
                        bt = bt.DeepCopy(BTree, false);
                        //bool b = Test.CompareBoard(BTree, bt);
                        int result = PlayOut(ref bt, ((int)BTree.RootColor ^ 1), max_index, BTree.ply + 1);
                        //Console.WriteLine("root_loop3");
                        EvalNode(max_index, ((int)BTree.RootColor ^ 1));
                        if (is_abort)
                            break;
                        UpdateParam(max_index, result);
                        //Console.WriteLine("root_loop4");
                    }
                }
                else
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < elapsed + TimeBuffer)
                        break;
                    if (is_abort)
                        break;
                    DescendNode((int)BTree.RootColor ^ 1, max_index, nthr, BTree.ply + 1);
                }

                UnDo(ref BTree, (int)BTree.RootColor, BTree.RootMoves[max_index - 1]);
            }
        }

        // ToDo:合法手のチェックのルーチンを共通化した方が良い
        private void ExpandNode(int color, int parent_index, int ply)
        {
            // 1手詰めと宣言勝ちをチェックし、勝ちがあった場合は手番側の勝ちをNodeに設定してreturnする color = 1
            //Move[] mate_move = new Move[3];
            Move mate_move = new Move();
            //if (m3p.IsMateIn3Ply(ref BTree, color, true, ref mate_move))
            ulong ulret = IsAttacked(BTree, BTree.SQ_King[color], color);
            if (ulret == 0 && Mate1Ply.IsMateIn1Ply(ref BTree, ref mate_move, color) != 0)
            {
                NodeList[parent_index].WinRateSum = 0.0F;
                NodeList[parent_index].LostRateSum = 1.0F;
                NodeList[parent_index].EvalCount += 1;
                return;
            }
            else
            {
                if (IsAttacked(BTree, BTree.SQ_King[color ^ 1], color ^ 1) != 0)
                {
                    int iret = JudgeDeclarationWin(BTree);
                    if (iret == color + 1)
                    {
                        NodeList[parent_index].WinRateSum = 0.0F;
                        NodeList[parent_index].LostRateSum = 1.0F;
                        NodeList[parent_index].EvalCount += 1;
                        return;
                    }
                    else if (color == 0 && iret == 2 || color == 1 && iret == 1)
                    {
                        NodeList[parent_index].WinRateSum = 1.0F;
                        NodeList[parent_index].LostRateSum = 0.0F;
                        NodeList[parent_index].EvalCount += 1;
                        return;
                    }
                }
            }

            // 自玉に王手がかかっている場合
            if (ulret != 0)
            {
                List<Move> evasion_moves = new List<Move>();
                GenEvasion.Generate(ref BTree, ref evasion_moves, color);
                // 王手を逃れる手が生成できなかったら詰み
                if (evasion_moves.Count == 0)
                {
                    NodeList[parent_index].WinRateSum = 1.0F;
                    NodeList[parent_index].LostRateSum = 0.0F;
                    NodeList[parent_index].EvalCount += 1;
                    return;
                }
            }
            
            int current_index;
            bool flag = false;

            //if (!tt.PolicyMoves.ContainsKey(BTree.CurrentHash))
            {
                // トランスポジションテーブルにデータがなかった場合

                // メインスレッドにPolicy Networkのリクエストを投げる
                // ToDo:ここを修正する。ルートの方向に128手まで遡る
                string str_throw = SFEN.ToSFEN(BTree, color);
                str_throw = "p," + str_throw;
                string str_csa_moves = "";
                int index = parent_index;
                int counter = 0;
                while (index != 0)
                {
                    str_csa_moves = ":" + CSA.Move2CSA(NodeList[index].move) + str_csa_moves;
                    index = NodeList[index].ParentIndex;
                    counter++;
                    if (counter == 128)
                        break;
                }

                queue_to_main_thread.Enqueue(str_throw + str_csa_moves);

                //Console.WriteLine("expand_node0");
                //Console.WriteLine(str_throw);
                //is_finished = false;

                // メインスレッドから結果を受信するまで待機する
                while (queue_from_main_thread.Count == 0) { Thread.Sleep(1); if (is_abort) { return; } }
                //while (!is_finished) { if (is_abort) { return; } }

                //is_finished = false;

                string str_receive = queue_from_main_thread.Dequeue(); //データのフォーマットは"7776FU 0.65,2226FU 0.25,5556FU 0.1"といった感じを想定

                //Console.WriteLine("expand_node0");

                // 詰んでいた場合
                if (str_receive == "mate")
                {
                    NodeList[parent_index].WinRateSum = 1.0F;
                    NodeList[parent_index].LostRateSum = 0.0F;
                    NodeList[parent_index].EvalCount += 1;
                    return;
                }

                string[] s = str_receive.Split(',');
                List<Move> moves = new List<Move>();

                for (int i = 0; i < s.Length; i += 2)
                {
                    Node n = new Node();
                    n.color = color;
                    string[] s2 = s[i].Split(" ");
                    Move m = CSA.CSA2Move(BTree, s2[0]);

                    // Python側でたまに非合法手を生成してしまうのでチェックする。
                    // チェックルーチンは共通化した方が望ましいが…
                    if (m.From < NSquare)
                    {
                        uint idirec = ADirec[m.From, m.To];

                        // 対象の駒がPINされている場合スキップ
                        if (IsPinnedOnKing(BTree, m.From, (int)idirec, color) != 0)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        if (m.CapPiece != Piece.Type.Empty)
                        {
                            continue;
                        }
                    }

                    // 不成が非合法手になる場合スキップ→こうならないように手の生成処理を実装したはずなのだが、たまに生成してしまっている
                    if (color == 0)
                    {
                        if (m.PieceType == Piece.Type.Knight && m.To < (int)A7 && m.FlagPromo == 0)
                        {
                            continue;
                        }
                        else if ((m.PieceType == Piece.Type.Lance || m.PieceType == Piece.Type.Pawn) && m.To < (int)A8 && m.FlagPromo == 0)
                        {
                            continue;
                        }
                    }
                    else
                    {
                        if (m.PieceType == Piece.Type.Knight && m.To > (int)I3 && m.FlagPromo == 0)
                        {
                            continue;
                        }
                        else if ((m.PieceType == Piece.Type.Lance || m.PieceType == Piece.Type.Pawn) && m.To > (int)I2 && m.FlagPromo == 0)
                        {
                            continue;
                        }
                    }

                    // 玉を捕獲してしまう場合スキップ
                    if (m.CapPiece == Piece.Type.King)
                    {
                        continue;
                    }

                    Do(ref BTree, color, m);

                    // 指した手によって自玉がDiscovered Checkになってしまった場合
                    if (IsAttacked(BTree, BTree.SQ_King[color], color) != 0)
                    {
                        UnDo(ref BTree, color, m);
                        continue;
                    }              

                    n.ThisIndex = NodeList.Count;
                    n.ParentIndex = NodeList[parent_index].ThisIndex;
                    n.move = m;
                    moves.Add(m);

                    //Do(ref BTree, color, m);

                    NodeList[parent_index].ChildIndexes.Add(n.ThisIndex);
                    n.PolicyResult = float.Parse(s2[1]);// SoftMax関数はPython側でかける
                    tt.PolicyDict.TryAdd(BTree.CurrentHash, n.PolicyResult);
                    NodeList.Add(n);
                    current_index = n.ThisIndex;
                    BoardTree bt = new BoardTree();
                    BoardTreeAlloc(ref bt);
                    bt = bt.DeepCopy(BTree, false);
                    int result = PlayOut(ref bt, color ^ 1, current_index, ply + 1);
                    EvalNode(current_index, color ^ 1);
                    UpdateParam(current_index, result);

                    UnDo(ref BTree, color, m);

                    flag = true;
                }
                //tt.PolicyMoves.Add(BTree.CurrentHash, moves);
                tt.PolicyMoves.TryAdd(BTree.CurrentHash, moves);
            }

            if (flag == true)
                NodeList[parent_index].IsLeaf = false;
        }

        // 歩が成れないところに成る手（歩不成）を生成しているのを確認。歩・香・桂について非合法手チェックを入れる

        private int PlayOut(ref BoardTree bt, int start_color, int node_index, int temp_ply)
        {
            const int ply_max = 384;
            int result = 2;// 初期値は引き分け
            int ply = temp_ply;
            int color = start_color;

            //Mate3Ply m3p_local = new Mate3Ply();
            //int record_move_index = 0;
            //Move[] record_moves = new Move[512];
            try
            {
                while (ply < ply_max)
                {
                    List<Move> move_list = new List<Move>();

                    /*if(tt.PolicyMoves.TryGetValue(bt.CurrentHash, out move_list))
                    {
                        goto next;
                    }
                    else
                    {
                        move_list = new List<Move>();
                    }*/
                    /*lock (this)
                    {

                    }*/

                    if (IsAttacked(bt, bt.SQ_King[color], color) == 0)
                    {
                        //Move[] moves = new Move[3];
                        Move mate_move = new Move();

                        // 詰みありの場合
                        //if (m3p.IsMateIn3Ply(ref bt, color, true, ref moves))
                        if (Mate1Ply.IsMateIn1Ply(ref bt, ref mate_move, color) != 0)
                        {
                            if (color == (int)bt.RootColor)
                            {
                                result = 0;// root手番の勝ち
                            }
                            else
                            {
                                result = 1;// 相手の勝ち
                            }
                            break;
                        }


                        // 宣言勝ちできるかどうかの確認
                        if (IsAttacked(bt, bt.SQ_King[color ^ 1], color ^ 1) == 0)
                        {
                            // 先後両方の玉に王手がかかっていない場合のみ宣言勝ちできる
                            int iret = DeclarationWin.JudgeDeclarationWin(bt);
                            if (iret == 1 && color == (int)bt.RootColor && color == 0)
                            {
                                result = 0; // root手番の勝ち（かつroot手番は先手）
                                break;
                            }
                            else if (iret == 2 && color == (int)bt.RootColor && color == 1)
                            {
                                result = 1; // 相手の勝ち（かつroot手番は後手）
                                break;
                            }
                        }
                        GenCap.Generate(bt, ref move_list, color);
                        GenNoCap.Generate(bt, ref move_list, color);
                        GenDrop.Generate(ref bt, ref move_list, color);
                    }
                    else
                    {
                        GenEvasion.Generate(ref bt, ref move_list, color);
                    }

                next:

                    List<Move> legal_move_list = new List<Move>();
                    List<int> score_list = new List<int>();

                    // ここでは合法手のチェックを行う
                    for (int i = 0; i < move_list.Count; i++)
                    {
                        if (move_list[i].From < NSquare)
                        {
                            uint idirec = ADirec[move_list[i].From, move_list[i].To];

                            // 対象の駒がPINされている場合スキップ
                            if (IsPinnedOnKing(bt, move_list[i].From, (int)idirec, color) != 0)
                            {
                                continue;
                            }
                        }

                        // 不成が非合法手になる場合スキップ→こうならないように手の生成処理を実装したはずなのだが、たまに生成してしまっている
                        if (color == 0)
                        {
                            if (move_list[i].PieceType == Piece.Type.Knight && move_list[i].To < (int)A7 && move_list[i].FlagPromo == 0)
                            {
                                continue;
                            }
                            else if ((move_list[i].PieceType == Piece.Type.Lance || move_list[i].PieceType == Piece.Type.Pawn) && move_list[i].To < (int)A8 && move_list[i].FlagPromo == 0)
                            {
                                continue;
                            }
                        }
                        else
                        {
                            if (move_list[i].PieceType == Piece.Type.Knight && move_list[i].To > (int)I3 && move_list[i].FlagPromo == 0)
                            {
                                continue;
                            }
                            else if ((move_list[i].PieceType == Piece.Type.Lance || move_list[i].PieceType == Piece.Type.Pawn) && move_list[i].To > (int)I2 && move_list[i].FlagPromo == 0)
                            {
                                continue;
                            }
                        }

                        // 玉を捕獲してしまう場合スキップ
                        if (move_list[i].CapPiece == Piece.Type.King)
                        {
                            continue;
                        }

                        legal_move_list.Add(move_list[i]);
                    }

                    // 合法手がなかった場合は終局
                    if (legal_move_list.Count == 0)
                    {
                        if (color != (int)bt.RootColor)
                        {
                            result = 0;// root手番の勝ち
                        }
                        else
                        {
                            result = 1;// 相手の勝ち
                        }
                        break;
                    }

                    /*for (int i = 0; i < legal_move_list.Count; i++)
                    {
                        uint hash = (mf.progress_table[ply] << 23) | (legal_move_list[i].Value);
                        score_list.Add((int)mf.move_feature[hash]);//
                    }

                    int candidate_size;
                    if (score_list.Count < 4)
                    {
                        candidate_size = score_list.Count;
                    }
                    else
                    {
                        candidate_size = 4;
                    }

                    Move[] candidate_moves = new Move[candidate_size];
                    int[] candidate_indexes = new int[candidate_size];
                    for (int i = 0; i < candidate_size; i++)
                    {
                        int max_index = Array.IndexOf(score_list.ToArray(), score_list.Max());
                        candidate_indexes[i] = max_index;
                        score_list[max_index] = int.MinValue;//
                        candidate_moves[i] = legal_move_list[candidate_indexes[i]];
                    }*/

                    Random r = new Random();
                    //int n = r.Next(candidate_size);
                    int n = r.Next(legal_move_list.Count);

                    //Do(ref bt, color, candidate_moves[n]);
                    Do(ref bt, color, legal_move_list[n]);

                    //record_moves[record_move_index++] = candidate_moves[n];

                    color ^= 1;
                    ply++;
                }

                NodeList[node_index].PlayoutCount++;
                PlayOutCount++;
            }
            catch (Exception e)
            {
                result = 2;
            }

            return result;
        }

        private void EvalNode(int node_index, int color)
        {
            if (tt.ValueDict.TryGetValue(BTree.CurrentHash, out float f))
            {
                // トランスポジションテーブルにデータがあった場合
                NodeList[node_index].WinRateSum = 1 - f;
                NodeList[node_index].LostRateSum = f;
                NodeList[node_index].EvalCount++;
                return;
            }

            // メインスレッドにValue Networkのリクエストを投げる
            string str_throw = SFEN.ToSFEN(BTree, color);
            str_throw = "v," + str_throw;
            queue_to_main_thread.Enqueue(str_throw);

            //Console.WriteLine("eval_node0");

            //is_finished = false;

            // メインスレッドから結果を受信するまで待機する
            while (queue_from_main_thread.Count == 0) { Thread.Sleep(1); if (is_abort) { return; } }

            //while (!is_finished) { if (is_abort) { return; } }

            //is_finished = false;

            string str_receive = queue_from_main_thread.Dequeue(); //データのフォーマットは"0.65"といった感じを想定

            //Console.WriteLine("eval_node1");

            // たまにこのケースがあるようだ
            /*if (str_receive == "")
            {
                NodeList[node_index].WinRateSum = 0.5F;
                NodeList[node_index].LostRateSum = 0.5F;
                NodeList[node_index].EvalCount += 1;
                return;
            }*/
            float v = float.Parse(str_receive);

            // 手番側の勝率で学習させてある。1手指した後の手番の勝率が返ってくるので反転する。
            NodeList[node_index].WinRateSum = 1 - v;
            NodeList[node_index].LostRateSum = v;
            NodeList[node_index].EvalCount++;// ToDo:要るかどうか分からないので要精査

            // トランスポジションテーブルに保存する
            tt.ValueDict.TryAdd(BTree.CurrentHash, v);
        }

        private void UpdateParam(int node_index, int result)
        {
            NodeList[node_index].TrialCount += 1;
            if (result == 0)
            {
                if (NodeList[node_index].color == (int)BTree.RootColor)
                {
                    NodeList[node_index].WinCount += 1;
                }
                else
                {
                    NodeList[node_index].LostCount += 1;
                }
            }
            else if (result == 1)
            {
                if (NodeList[node_index].color == (int)BTree.RootColor)
                {
                    NodeList[node_index].LostCount += 1;
                }
                else
                {
                    NodeList[node_index].WinCount += 1;
                }
            }
            else
            {
                NodeList[node_index].DrawCount += 1;
            }
            Node current_node = NodeList[node_index];
            float delta;
            float delta2;
            // 手番側の勝率で学習させてある
            if (current_node.color == (int)BTree.RootColor)
            {
                delta = NodeList[node_index].WinRateSum;
                delta2 = NodeList[node_index].LostRateSum;
            }
            else
            {
                delta = NodeList[node_index].LostRateSum;
                delta2 = NodeList[node_index].WinRateSum;
            }
            if (current_node.ParentIndex == 0)
                return;
            while (true)
            {
                int index = current_node.ParentIndex;
                current_node = NodeList[index];
                NodeList[index].TrialCount += 1;
                NodeList[index].PlayoutCount += 1;
                if (result == 0)
                {
                    if (NodeList[index].color == (int)BTree.RootColor)
                    {
                        NodeList[index].WinCount += 1;
                    }
                    else
                    {
                        NodeList[index].LostCount += 1;
                    }
                }
                else if (result == 1)
                {
                    if (NodeList[index].color == (int)BTree.RootColor)
                    {
                        NodeList[index].LostCount += 1;
                    }
                    else
                    {
                        NodeList[index].WinCount += 1;
                    }
                }
                else
                {
                    NodeList[index].DrawCount += 1;
                }

                if (current_node.color == (int)BTree.RootColor)
                {
                    NodeList[index].WinRateSum += delta;
                    NodeList[index].LostRateSum += delta2;
                }
                else
                {
                    NodeList[index].WinRateSum += delta2;
                    NodeList[index].LostRateSum += delta;
                }

                NodeList[index].EvalCount += 1;
                if (current_node.ParentIndex == 0)
                    break;
            }
        }

        void DescendNode(int color, int node_index, int nthr, int ply)
        {
            if (NodeList[node_index].ChildIndexes.Count == 0)
                return;

            int idx = new int();
            bool flag = false;
            while (true)
            {
                long elapsed = sw.ElapsedMilliseconds;
                if (SearchTimeLimit < elapsed + TimeBuffer)
                    break;
                if (is_abort)
                    break;
                int t, i;
                float[] ucb1_array = new float[NodeList[node_index].ChildIndexes.Count];
                t = 0;
                i = 0;
                while (i < NodeList[node_index].ChildIndexes.Count)
                {
                    idx = NodeList[node_index].ChildIndexes[i];
                    t += NodeList[idx].PlayoutCount;
                    i++;
                }

                i = 0;
                while (i < NodeList[node_index].ChildIndexes.Count)
                {
                    idx = NodeList[node_index].ChildIndexes[i];
                    float u = NodeList[idx].PolicyResult * (float)Math.Sqrt(t) / (NodeList[idx].PlayoutCount + 1);
                    float q = 0.0F;
                    if (NodeList[idx].EvalCount > 0 && NodeList[idx].PlayoutCount > 0)
                        q = (float)((1 - value_lambda) * (NodeList[idx].WinRateSum / NodeList[idx].EvalCount) + value_lambda * (NodeList[idx].WinCount / NodeList[idx].PlayoutCount));
                    ucb1_array[i++] = u + q;
                }

                int max_index = Array.IndexOf(ucb1_array, ucb1_array.Max());
                idx = NodeList[node_index].ChildIndexes[max_index];
                Do(ref BTree, color, NodeList[idx].move);

                if (NodeList[idx].IsLeaf)
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < elapsed + TimeBuffer)
                    {
                        flag = true;
                        goto end;
                    }

                    if (is_abort)
                        break;

                    if (NodeList[idx].TrialCount >= nthr)
                    {
                        ExpandNode(color ^ 1, idx, ply + 1);
                        if (is_abort)
                            break;
                    }
                    else
                    {
                        BoardTree bt = new BoardTree();
                        BoardTreeAlloc(ref bt);
                        bt = bt.DeepCopy(BTree, false);
                        int result = PlayOut(ref bt, color ^ 1, idx, ply + 1);
                        EvalNode(idx, color ^ 1);
                        if (is_abort)
                            break;
                        UnDo(ref BTree, color, NodeList[idx].move);
                        AscendNode(color ^ 1, idx, result);
                        return;
                    }
                }
                else
                {
                    elapsed = sw.ElapsedMilliseconds;
                    if (SearchTimeLimit < elapsed + TimeBuffer)
                    {
                        flag = true;
                        goto end;
                    }

                    if (is_abort)
                        break;

                    DescendNode(color ^ 1, idx, nthr, ply + 1);
                    return;
                }

                UnDo(ref BTree, color, NodeList[idx].move);

            end:

                if (flag)
                {
                    Node current_node = NodeList[idx];
                    int temp_color = color ^ 1;
                    while (true)
                    {
                        int index = current_node.ParentIndex;
                        current_node = NodeList[index];
                        if (current_node.ParentIndex == 0)
                            break;
                        UnDo(ref BTree, temp_color, current_node.move);
                        temp_color ^= 1;
                    }
                    break;
                }
            }
        }

        void AscendNode(int color, int node_index, int result)// パラメータ plyは要らないか？
        {
            NodeList[node_index].TrialCount += 1;
            if (result == 0)
            {
                if (NodeList[node_index].color == (int)BTree.RootColor)
                {
                    NodeList[node_index].WinCount += 1;
                }
                else
                {
                    NodeList[node_index].LostCount += 1;
                }
            }
            else if (result == 1)
            {
                if (NodeList[node_index].color == (int)BTree.RootColor)
                {
                    NodeList[node_index].LostCount += 1;
                }
                else
                {
                    NodeList[node_index].WinCount += 1;
                }
            }
            else
            {
                NodeList[node_index].DrawCount += 1;
            }
            Node current_node = NodeList[node_index];
            float delta;
            float delta2;
            // 手番側の勝率で学習させてある
            if (current_node.color == (int)BTree.RootColor)
            {
                delta = NodeList[node_index].WinRateSum;
                delta2 = NodeList[node_index].LostRateSum;
            }
            else
            {
                delta = NodeList[node_index].LostRateSum;
                delta2 = NodeList[node_index].WinRateSum;
            }
            if (current_node.ParentIndex == 0)
                return;
            int temp_color = color;
            //int temp_ply = ply - 1;
            while (true)
            {
                int index = current_node.ParentIndex;
                current_node = NodeList[index];
                NodeList[index].TrialCount += 1;
                NodeList[index].PlayoutCount += 1;
                if (result == 0)
                {
                    if (NodeList[index].color == (int)BTree.RootColor)
                    {
                        NodeList[index].WinCount += 1;
                    }
                    else
                    {
                        NodeList[index].LostCount += 1;
                    }
                }
                else if (result == 1)
                {
                    if (NodeList[index].color == (int)BTree.RootColor)
                    {
                        NodeList[index].LostCount += 1;
                    }
                    else
                    {
                        NodeList[index].WinCount += 1;
                    }
                }
                else
                {
                    NodeList[index].DrawCount += 1;
                }

                if (current_node.color == (int)BTree.RootColor)
                {
                    NodeList[index].WinRateSum += delta;
                    NodeList[index].LostRateSum += delta2;
                }
                else
                {
                    NodeList[index].WinRateSum += delta2;
                    NodeList[index].LostRateSum += delta;
                }

                NodeList[index].EvalCount += 1;
                if (current_node.ParentIndex == 0)
                    break;
                UnDo(ref BTree, temp_color, current_node.move);
                temp_color ^= 1;
            }
        }
    }
}
