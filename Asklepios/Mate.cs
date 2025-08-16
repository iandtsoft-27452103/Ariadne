using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.MakeMove;
using static Asklepios.AttackBitBoard;

namespace Asklepios
{
    internal class Mate
    {
        public Move[] move_cur;
        public List<List<Move>> mate_proc = new List<List<Move>>();
        public List<List<Move>> no_mate_proc = new List<List<Move>>();
        public Move first_move = new Move();
        public Move second_move = new Move();
        public int max_ply = 0;
        public bool is_abort = false;
        public bool is_mate_root = false;
        public BoardTree BTree;
        public List<CheckMove> RootCheckMoves = new List<CheckMove>();
        public string root_str_pv = "";
        //public int color;

        public List<CheckMove> GenRootCheckMoves(ref BoardTree bt)
        {
            List<CheckMove> checkMoves = new List<CheckMove>();

            if (bt.RootColor == 0)
            {
                GenCheck.BGenChecks(ref bt, ref checkMoves);
            }
            else
            {
                GenCheck.WGenChecks(ref bt, ref checkMoves);
            }

            return checkMoves;
        }

        public void MateSearchWrapper()
        {
            const int depth_max = 17;// 後で変える
            int rest_depth = 1;

            while (rest_depth < depth_max)
            {
                max_ply = rest_depth;
                move_cur = new Move[max_ply + 1];
                for (int i = 0; i < max_ply + 1; i++)
                    move_cur[i] = new Move();

                mate_proc.Clear();
                no_mate_proc.Clear();
                first_move = new Move();
                second_move = new Move();

                is_mate_root = Offend(ref BTree, (int)BTree.RootColor, rest_depth, 1);

                if (is_mate_root)
                    break;

                if (is_abort)
                {
                    is_mate_root = false;
                    break;
                }

                rest_depth += 2;
            }

            if (is_mate_root && !is_abort)
            {
                Console.WriteLine("詰みあり");
                root_str_pv = OutResult(rest_depth);
            }
        }

        private string OutResult(int rest_depth)
        {
            int i, j, k;

            List<List<Move>> l = mate_proc;
            List<List<Move>> nl = no_mate_proc;
            
            bool b;
            int color;
            string str_pv = "";
            string[] str_color = new string[2];

            str_color[0] = "+";
            str_color[1] = "-";

            /*if (BTree.RootColor == 0)
            {
                str_color[0] = "+";
                str_color[1] = "-";
            }
            else
            {
                str_color[0] = "-";
                str_color[1] = "+";
            }*/

            b = false;
            List<int> idxes = new List<int>();
            //int a = 0;
            for (i = 0; i < l.Count; i++)
            {
                //if (is_abort)
                    //return "";
                string s = (i + 1).ToString() + " / " + l.Count.ToString();
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
                        /*else
                        {
                            int z = 0;
                        }*/
                    }
                    if (!b)
                    {
                        idxes.Add(i);
                    }
                }
            }

            for (i = 0; i < l.Count; i++)
            {
                //if (is_abort)
                    //return "";
                if (idxes.Contains(i))
                    continue;
                str_pv = "";
                color = (int)BTree.RootColor;
                for (j = 0; j < rest_depth; j++)
                {
                    //str_pv += CSA.Move2CSA(l[index][i]);
                    str_pv += str_color[color];
                    str_pv += CSA.Move2CSA(l[i][j]);
                    if (j != rest_depth - 1)
                        str_pv += ", ";
                    color ^= 1;
                }
                Console.WriteLine(str_pv);
            }

            return str_pv;
        }

        public bool Offend(ref BoardTree bt, int color, int rest_depth, int ply)// PrevMoveは要るか？
        {
            bool is_mate;
            List<CheckMove> checkMoves = new List<CheckMove>();

            if (is_abort)
                return false;

            // 王手を生成する
            if (ply != 1)
            {
                if (color == 0)
                {
                    GenCheck.BGenChecks(ref bt, ref checkMoves);
                }
                else
                {
                    GenCheck.WGenChecks(ref bt, ref checkMoves);
                }
            }
            else
            {
                checkMoves = RootCheckMoves;
            }

            // 王手が生成されなかったら詰まない
            if (checkMoves.Count == 0)
            {
                return false;
            }        

            for (int i = 0; i < checkMoves.Count; i++)
            {
                //CheckMove current_check_move = checkMoves[i];
                if (is_abort)
                    return false;

                move_cur[ply] = checkMoves[i];

                if (move_cur[ply].PieceType == Piece.Type.Empty)// たまにこのような状態になる。原因は特定できていない。
                    continue;

                Do(ref bt, color, move_cur[ply]);

                // // たまにこのような状態になる。原因は特定できていない。
                if (IsAttacked(BTree, BTree.SQ_King[color ^ 1], color ^ 1) == 0)
                {
                    UnDo(ref bt, color, move_cur[ply]);
                    continue;
                }

                // Discoverd Checkになってしまった場合
                if (IsAttacked(BTree, BTree.SQ_King[color], color) != 0)
                {
                    UnDo(ref bt, color, move_cur[ply]);
                    continue;
                }

                is_mate = Defend(ref bt, color ^ 1, rest_depth - 1, ply + 1);

                //if (is_mate && rest_depth == 1 && move_cur[ply].From == )

                /*if (ply == max_ply)
                {
                    List<Move> moves = new List<Move>();
                    for (int j = 1; j < ply + 1; j++)
                        moves.Add(move_cur[j]);
                    mate_proc.Add(moves);
                }*/

                if (is_mate)
                {
                    if (ply == max_ply)
                    {
                        List<Move> moves = new List<Move>();
                        for (int j = 1; j < ply + 1; j++)
                            moves.Add(move_cur[j]);
                        mate_proc.Add(moves);
                    }
                    UnDo(ref bt, color, move_cur[ply]);
                    return true;
                }             

                UnDo(ref bt, color, move_cur[ply]);
            }

            return false;
        }

        // ToDo: 詰みの手順を返す処理を入れる
        public bool Defend(ref BoardTree bt, int color, int rest_depth, int ply)// PrevMoveは要るか？
        {
            bool is_mate;
            int mate_count = 0;
            List<Move> evasionMoves = new List<Move>();

            if (is_abort)
                return false;

            //Move save_evasion_move = new Move();
            //int save_pv_length = pv_length;

            // 王手を避ける手を生成する
            GenEvasion.Generate(ref bt, ref evasionMoves, color);

            // 残り深さが1で王手を避ける手が生成されたら詰みではない
            if (rest_depth == 0 && evasionMoves.Count > 0)
            {
                return false;
            }

            // 王手を避ける手が生成されなかったら詰みである
            if (evasionMoves.Count == 0)
            {
                //pv_length = ply - 1;
                return true;
            }

            //Move current_evasion_move;

            for (int i = 0;i < evasionMoves.Count; i++)
            {
                if (is_abort)
                    return false;
                //current_evasion_move = evasionMoves[i];

                move_cur[ply] = evasionMoves[i];

                Do(ref bt, color, move_cur[ply]);

                is_mate = Offend(ref bt, color ^ 1, rest_depth - 1, ply + 1);

                if (!is_mate)
                {
                    //if (rest_depth == 2)
                    {
                        /*while (mate_count > 0)
                        {
                            mate_proc.Remove(mate_proc[mate_proc.Count - 1]);
                            mate_count--;
                        }*/
                        List<Move> moves = new List<Move>();
                        for (int j = 0; j < ply - 1; j++)
                        {
                            moves.Add(move_cur[j + 1]);
                        }
                        no_mate_proc.Add(moves);
                    }
                    // ※守備側で不詰みがあった場合、1つ上の攻撃側の手までの手順は全部不詰みとなる。
                    UnDo(ref bt, color, move_cur[ply]);
                    return false;
                }
                else
                {
                    mate_count++;
                    /*if (pv_length >= save_pv_length)
                    {
                        save_evasion_move = current_evasion_move;
                    }*/
                }

                UnDo(ref bt, color, move_cur[ply]);
            }

            if (ply == 2 && mate_count == evasionMoves.Count)
            {
                first_move = move_cur[1];
                second_move = move_cur[2];
            }

            return true;// どの手を指しても詰みだった場合
        }
    }
}
