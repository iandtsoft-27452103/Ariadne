using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.ShogiCommon;
using static Asklepios.Color;
using static Asklepios.Piece;
using static Asklepios.BitOperation;
using static Asklepios.AttackBitBoard;
using static Asklepios.MakeMove;
using System.Reflection;

namespace Asklepios
{
    internal class Board
    {
        public struct BoardTree
        {
            public BitBoard[,] BB_Piece;
            public BitBoard[] BB_BH;
            public BitBoard[] BB_RD;
            public BitBoard[] BB_HDK;
            public BitBoard[] BB_Total_Gold;
            public BitBoard[] BB_Occupied;
            public BitBoard[] BB_PawnAttacks;
            public int[] SQ_King;
            public int[,] Hand;
            public int[] ShogiBoard;
            public Color.Type RootColor;
            public Move[] RootMoves;
            public ulong[] Hash;
            public ulong CurrentHash;
            public ulong PrevHash;
            public int ply;
            public BoardTree Copy()
            {
                return (BoardTree)MemberwiseClone();
            }
            public BoardTree DeepCopy(BoardTree bt, bool flag)
            {
                BoardTree bt_base = new BoardTree();
                BoardTreeAlloc(ref bt_base);
                for (int i = 0; i < ColorNum; i++)
                {
                    bt_base.BB_BH[i] = BitBoard.Copy(bt.BB_BH[i]);
                    bt_base.BB_RD[i] = BitBoard.Copy(bt.BB_RD[i]);
                    bt_base.BB_HDK[i] = BitBoard.Copy(bt.BB_HDK[i]);
                    bt_base.BB_Total_Gold[i] = BitBoard.Copy(bt.BB_Total_Gold[i]);
                    bt_base.BB_Occupied[i] = BitBoard.Copy(bt.BB_Occupied[i]);
                    bt_base.BB_PawnAttacks[i] = BitBoard.Copy(bt.BB_PawnAttacks[i]);
                    bt_base.SQ_King[i] = bt.SQ_King[i];
                    for (int j = 0; j < PieceNum; j++)
                    {
                        bt_base.BB_Piece[i, j] = BitBoard.Copy(bt.BB_Piece[i, j]);
                    }
                    for (int j = 0; j < NHand; j++)
                    {
                        bt_base.Hand[i, j] = bt.Hand[i, j];
                    }
                }

                bt_base.RootColor = bt.RootColor;
                bt_base.ply = bt.ply;
                bt_base.CurrentHash = bt.CurrentHash;
                bt_base.PrevHash = bt.PrevHash;

                for (int i = 0; i < NSquare; i++)
                {
                    bt_base.ShogiBoard[i] = bt.ShogiBoard[i];
                }

                for (int i = 0; i < PlyMax + 1; i++)
                {
                    if (i != 0 && bt.Hash[i] == 0) { break; }
                    bt_base.Hash[i] = bt.Hash[i];
                }

                if (flag)
                {
                    bt_base.RootMoves = new Move[bt.RootMoves.Length];
                    for(int i = 0; i < bt.RootMoves.Length; i++)
                    {
                        Move move = new Move();
                        move.Push(bt.RootMoves[i].From, bt.RootMoves[i].To, bt.RootMoves[i].PieceType, bt.RootMoves[i].CapPiece, bt.RootMoves[i].FlagPromo);
                        bt_base.RootMoves[i] = move;
                    }
                }

                return bt_base;
            }
        }

        public static void BoardTreeAlloc(ref BoardTree BTree)
        {
            BTree.BB_Piece = new BitBoard[ColorNum, Piece.PieceNum];
            BTree.BB_BH = new BitBoard[ColorNum];
            BTree.BB_RD = new BitBoard[ColorNum];
            BTree.BB_HDK = new BitBoard[ColorNum];
            BTree.BB_Total_Gold = new BitBoard[ColorNum];
            BTree.BB_Occupied = new BitBoard[ColorNum];
            BTree.BB_PawnAttacks = new BitBoard[ColorNum];
            BTree.SQ_King = new int[ColorNum];
            BTree.Hand = new int[ColorNum, NHand];
            BTree.ShogiBoard = new int[NSquare];
            BTree.RootColor = Color.Type.Black;
            // RootMovesは初期化しない
            BTree.Hash = new ulong[PlyMax + 1];
            BTree.ply = 1;

            for (int i = 0; i < ColorNum; i++)
            {
                BTree.BB_BH[i] = new BitBoard();
                BTree.BB_RD[i] = new BitBoard();
                BTree.BB_HDK[i] = new BitBoard();
                BTree.BB_Total_Gold[i] = new BitBoard();
                BTree.BB_Occupied[i] = new BitBoard();
                BTree.BB_PawnAttacks[i] = new BitBoard();
                BTree.SQ_King[i] = new int();
                for (int j = 0; j < Piece.PieceNum; j++)
                {
                    BTree.BB_Piece[i, j] = new BitBoard();
                }
                for (int j = 0; j < NHand; j++)
                {
                    BTree.Hand[i, j] = new int();
                }
            }
        }

        public static void BoardIni(ref BoardTree BTree)
        {
            for (int i = 0; i < ColorNum; i++)
            {
                for (int j = 0; j < PieceNum; j++)
                {
                    BBIni(ref BTree.BB_Piece[i, j]);
                    if (j < NHand)
                    {
                        BTree.Hand[i, j] = 0;
                    }
                }
                BBIni(ref BTree.BB_BH[i]);
                BBIni(ref BTree.BB_RD[i]);
                BBIni(ref BTree.BB_HDK[i]);
                BBIni(ref BTree.BB_Total_Gold[i]);
                BBIni(ref BTree.BB_Occupied[i]);
                BBIni(ref BTree.BB_PawnAttacks[i]);
            }
            BTree.SQ_King[0] = (int)Square.E1;
            BTree.SQ_King[1] = (int)Square.E9;
            BTree.RootColor = Color.Type.Black;

            for (int i = 0; i < NSquare; i++)
            {
                Color.Type c;
                BTree.ShogiBoard[i] = StartPosition[i];
                int pc = BTree.ShogiBoard[i];
                if (pc > 0)
                {
                    c = Color.Type.Black;
                }
                else
                {
                    c = Color.Type.White;
                    pc = Math.Abs(pc);
                }

                if (pc != (int)Piece.Type.Empty)
                {
                    Xor(i, ref BTree.BB_Piece[(int)c, pc]);
                    Xor(i, ref BTree.BB_Occupied[(int)c]);
                    switch (pc)
                    {
                        case (int)Piece.Type.Pawn:
                            if (c == Color.Type.Black)
                            {
                                Xor(i - 9, ref BTree.BB_PawnAttacks[(int)c]);
                            }
                            else
                            {
                                Xor(i + 9, ref BTree.BB_PawnAttacks[(int)c]);
                            }
                            break;
                        case (int)Piece.Type.Gold:
                        case (int)Piece.Type.Pro_Pawn:
                        case (int)Piece.Type.Pro_Lance:
                        case (int)Piece.Type.Pro_Knight:
                        case (int)Piece.Type.Pro_Silver:
                            Xor(i, ref BTree.BB_Total_Gold[(int)c]);
                            break;
                        case (int)Piece.Type.Bishop:
                            Xor(i, ref BTree.BB_BH[(int)c]);
                            break;
                        case (int)Piece.Type.Rook:
                            Xor(i, ref BTree.BB_RD[(int)c]);
                            break;
                        case (int)Piece.Type.King:
                            Xor(i, ref BTree.BB_HDK[(int)c]);
                            break;
                        case (int)Piece.Type.Horse:
                            Xor(i, ref BTree.BB_BH[(int)c]);
                            Xor(i, ref BTree.BB_HDK[(int)c]);
                            break;
                        case (int)Piece.Type.Dragon:
                            Xor(i, ref BTree.BB_RD[(int)c]);
                            Xor(i, ref BTree.BB_HDK[(int)c]);
                            break;
                    }
                }
            }
            BTree.PrevHash = 0;
            BTree.Hash[1] = BTree.CurrentHash = Hash.HashFunc(BTree);
            BTree.ply = 1;
        }

        public static void SetRootPos(ref BoardTree bt, string str_response, int color, ref float[] policy_output)
        {
            string[] s0 = str_response.Split(',');
            //bt.RootMoves = new Move[s0.Length];
            List<Move> moves = new List<Move>();
            policy_output = new float[s0.Length];
            for (int i = 0; i < s0.Length; i++)
            {
                string[] s1 = s0[i].Split(" ");
                //bt.RootMoves[i] = CSA.CSA2Move(bt, s1[0]);
                Move temp_move = CSA.CSA2Move(bt, s1[0]);
                Do(ref bt, color, temp_move);

                // 指した手によって自玉がDiscovered Checkになってしまった場合
                if (IsAttacked(bt, bt.SQ_King[color], color) != 0)
                {
                    UnDo(ref bt, color, temp_move);
                    continue;
                }
                    

                // 玉を取ってしまう非合法手の場合
                if (temp_move.CapPiece == Piece.Type.King)
                {
                    UnDo(ref bt, color, temp_move);
                    continue;
                }

                moves.Add(temp_move);

                UnDo(ref bt, color, temp_move);
                
                policy_output[i] = float.Parse(s1[1]);
            }

            bt.RootMoves = new Move[moves.Count];
            for (int i = 0; i < moves.Count; i++)
                bt.RootMoves[i] = moves[i];

            bt.RootColor = (Color.Type)color;
        }
    }
}
