using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.AttackBitBoard;
using static Asklepios.BitOperation;
using static Asklepios.ShogiCommon;
using static Asklepios.Hash;

namespace Asklepios
{
    internal class MakeMove
    {
        // パラメータのplyは一旦削除した
        public static void Do(ref BoardTree BTree, int color, Move move)
        {
            int sign = -1;
            if (color == (int)Color.Type.White)
                sign = 1;

            BTree.PrevHash = BTree.CurrentHash;

            if (move.From == NSquare)
            {
                switch (move.PieceType)
                {
                    case Piece.Type.Pawn:
                        Xor(move.To + sign * NFile, ref BTree.BB_PawnAttacks[color]);
                        break;
                    case Piece.Type.Gold:
                        Xor(move.To, ref BTree.BB_Total_Gold[color]);
                        break;
                    case Piece.Type.Bishop:
                        Xor(move.To, ref BTree.BB_BH[color]);
                        break;
                    case Piece.Type.Rook:
                        Xor(move.To, ref BTree.BB_RD[color]);
                        break;
                }

                Xor(move.To, ref BTree.BB_Piece[color, (int)move.PieceType]);
                BTree.CurrentHash ^= Hash.PieceRand[color, (int)move.PieceType, move.To];
                BTree.Hand[color, (int)move.PieceType] -= 1;
                BTree.ShogiBoard[move.To] = -sign * (int)move.PieceType;
                Xor(move.To, ref BTree.BB_Occupied[color]);
            }
            else
            {
                BitBoard bb_set_clear = new BitBoard();
                BBOr(ref bb_set_clear, AbbMask[move.From], AbbMask[move.To]);
                BBXor(ref BTree.BB_Occupied[color], BTree.BB_Occupied[color], bb_set_clear);
                BTree.ShogiBoard[move.From] = (int)Piece.Type.Empty;
                if (move.FlagPromo == 1)
                {
                    switch(move.PieceType)
                    {
                        case Piece.Type.Pawn:
                            Xor(move.To, ref BTree.BB_PawnAttacks[color]);
                            Xor(move.To, ref BTree.BB_Total_Gold[color]);
                            break;
                        case Piece.Type.Lance:
                        case Piece.Type.Knight:
                        case Piece.Type.Silver:
                            Xor(move.To, ref BTree.BB_Total_Gold[color]);
                            break;
                        case Piece.Type.Bishop:
                            Xor(move.To, ref BTree.BB_HDK[color]);
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            break;
                        case Piece.Type.Rook:
                            Xor(move.To, ref BTree.BB_HDK[color]);
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            break;
                    }
                    Xor(move.From, ref BTree.BB_Piece[color, (int)move.PieceType]);
                    Xor(move.To, ref BTree.BB_Piece[color, (int)(move.PieceType + Piece.PromoteNum)]);
                    BTree.CurrentHash ^= PieceRand[color, (int)(move.PieceType + Piece.PromoteNum), move.To] ^ PieceRand[color, (int)move.PieceType, move.From];
                    BTree.ShogiBoard[move.To] = -sign * ((int)move.PieceType + Piece.PromoteNum);
                }
                else
                {
                    switch (move.PieceType)
                    {
                        case Piece.Type.Pawn:
                            Xor(move.To + sign * NFile, ref BTree.BB_PawnAttacks[color]);
                            Xor(move.To, ref BTree.BB_PawnAttacks[color]);
                            break;
                        case Piece.Type.Gold:
                        case Piece.Type.Pro_Pawn:
                        case Piece.Type.Pro_Lance:
                        case Piece.Type.Pro_Knight:
                        case Piece.Type.Pro_Silver:
                            BBXor(ref BTree.BB_Total_Gold[color], BTree.BB_Total_Gold[color], bb_set_clear);
                            break;
                        case Piece.Type.Bishop:
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            break;
                        case Piece.Type.Rook:
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            break;
                        case Piece.Type.King:
                            BTree.SQ_King[color] = move.To;
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                        case Piece.Type.Horse:
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                        case Piece.Type.Dragon:
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                    }
                    BBXor(ref BTree.BB_Piece[color, (int)move.PieceType], BTree.BB_Piece[color, (int)move.PieceType], bb_set_clear);
                    BTree.CurrentHash ^= PieceRand[color, (int)move.PieceType, move.To] ^ PieceRand[color, (int)move.PieceType, move.From];
                    BTree.ShogiBoard[move.To] = -sign * (int)move.PieceType;
                }
            }

            if ((int)move.CapPiece != 0)
            {
                switch (move.CapPiece)
                {
                    case Piece.Type.Pawn:
                        Xor(move.To - sign * NFile, ref BTree.BB_PawnAttacks[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Lance:
                    case Piece.Type.Knight:
                    case Piece.Type.Silver:
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Gold:
                        Xor(move.To, ref BTree.BB_Total_Gold[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Bishop:
                        Xor(move.To, ref BTree.BB_BH[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Rook:
                        Xor(move.To, ref BTree.BB_RD[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Pro_Pawn:
                    case Piece.Type.Pro_Lance:
                    case Piece.Type.Pro_Knight:
                    case Piece.Type.Pro_Silver:
                        Xor(move.To, ref BTree.BB_Total_Gold[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] += 1;
                        break;
                    case Piece.Type.Horse:
                        Xor(move.To, ref BTree.BB_HDK[color ^ 1]);
                        Xor(move.To, ref BTree.BB_BH[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] += 1;
                        break;
                    case Piece.Type.Dragon:
                        Xor(move.To, ref BTree.BB_HDK[color ^ 1]);
                        Xor(move.To, ref BTree.BB_RD[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] += 1;
                        break;
                }
                Xor(move.To, ref BTree.BB_Piece[color ^ 1, (int)move.CapPiece]);
                BTree.CurrentHash ^= PieceRand[color ^ 1, (int)move.CapPiece, move.To];
                Xor(move.To, ref BTree.BB_Occupied[color ^ 1]);
            }

            BTree.Hash[++BTree.ply] = BTree.CurrentHash;
        }

        public static void UnDo(ref BoardTree BTree, int color, Move move)
        {
            int sign = -1;
            if (color == (int)Color.Type.White)
                sign = 1;

            BTree.CurrentHash = BTree.PrevHash;

            if (move.From == NSquare)
            {
                switch (move.PieceType)
                {
                    case Piece.Type.Pawn:
                        Xor(move.To + sign * NFile, ref BTree.BB_PawnAttacks[color]);
                        break;
                    case Piece.Type.Gold:
                        Xor(move.To, ref BTree.BB_Total_Gold[color]);
                        break;
                    case Piece.Type.Bishop:
                        Xor(move.To, ref BTree.BB_BH[color]);
                        break;
                    case Piece.Type.Rook:
                        Xor(move.To, ref BTree.BB_RD[color]);
                        break;
                }
                Xor(move.To, ref BTree.BB_Piece[color, (int)move.PieceType]);
                BTree.Hand[color, (int)move.PieceType] += 1;
                BTree.ShogiBoard[move.To] = (int)Piece.Type.Empty;
                Xor(move.To, ref BTree.BB_Occupied[color]);
            }
            else
            {
                BitBoard bb_set_clear = new BitBoard();
                BBOr(ref bb_set_clear, AbbMask[move.From], AbbMask[move.To]);
                BBXor(ref BTree.BB_Occupied[color], BTree.BB_Occupied[color], bb_set_clear);
                BTree.ShogiBoard[move.From] = (int)Piece.Type.Empty;
                if (move.FlagPromo == 1)
                {
                    switch (move.PieceType)
                    {
                        case Piece.Type.Pawn:
                            Xor(move.To, ref BTree.BB_PawnAttacks[color]);
                            Xor(move.To, ref BTree.BB_Total_Gold[color]);
                            break;
                        case Piece.Type.Lance:
                        case Piece.Type.Knight:
                        case Piece.Type.Silver:
                            Xor(move.To, ref BTree.BB_Total_Gold[color]);
                            break;
                        case Piece.Type.Bishop:
                            Xor(move.To, ref BTree.BB_HDK[color]);
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            break;
                        case Piece.Type.Rook:
                            Xor(move.To, ref BTree.BB_HDK[color]);
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            break;
                    }
                    Xor(move.From, ref BTree.BB_Piece[color, (int)move.PieceType]);
                    Xor(move.To, ref BTree.BB_Piece[color, (int)(move.PieceType + Piece.PromoteNum)]);
                    BTree.ShogiBoard[move.From] = -sign * (int)move.PieceType;
                }
                else
                {
                    switch (move.PieceType)
                    {
                        case Piece.Type.Pawn:
                            Xor(move.To + sign * NFile, ref BTree.BB_PawnAttacks[color]);
                            Xor(move.To, ref BTree.BB_PawnAttacks[color]);
                            break;
                        case Piece.Type.Gold:
                        case Piece.Type.Pro_Pawn:
                        case Piece.Type.Pro_Lance:
                        case Piece.Type.Pro_Knight:
                        case Piece.Type.Pro_Silver:
                            BBXor(ref BTree.BB_Total_Gold[color], BTree.BB_Total_Gold[color], bb_set_clear);
                            break;
                        case Piece.Type.Bishop:
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            break;
                        case Piece.Type.Rook:
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            break;
                        case Piece.Type.King:
                            BTree.SQ_King[color] = move.From;
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                        case Piece.Type.Horse:
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                        case Piece.Type.Dragon:
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                    }
                    BBXor(ref BTree.BB_Piece[color, (int)move.PieceType], BTree.BB_Piece[color, (int)move.PieceType], bb_set_clear);
                    BTree.ShogiBoard[move.From] = -sign * (int)move.PieceType;
                }
            }
            if ((int)move.CapPiece != 0)
            {
                switch (move.CapPiece)
                {
                    case Piece.Type.Pawn:
                        Xor(move.To - sign * NFile, ref BTree.BB_PawnAttacks[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] -= 1;
                        break;
                    case Piece.Type.Lance:
                    case Piece.Type.Knight:
                    case Piece.Type.Silver:
                        BTree.Hand[color, (int)move.CapPiece] -= 1;
                        break;
                    case Piece.Type.Gold:
                        Xor(move.To, ref BTree.BB_Total_Gold[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] -= 1;
                        break;
                    case Piece.Type.Bishop:
                        Xor(move.To, ref BTree.BB_BH[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] -= 1;
                        break;
                    case Piece.Type.Rook:
                        Xor(move.To, ref BTree.BB_RD[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] -= 1;
                        break;
                    case Piece.Type.Pro_Pawn:
                    case Piece.Type.Pro_Lance:
                    case Piece.Type.Pro_Knight:
                    case Piece.Type.Pro_Silver:
                        Xor(move.To, ref BTree.BB_Total_Gold[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] -= 1;
                        break;
                    case Piece.Type.Horse:
                        Xor(move.To, ref BTree.BB_HDK[color ^ 1]);
                        Xor(move.To, ref BTree.BB_BH[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] -= 1;
                        break;
                    case Piece.Type.Dragon:
                        Xor(move.To, ref BTree.BB_HDK[color ^ 1]);
                        Xor(move.To, ref BTree.BB_RD[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] -= 1;
                        break;
                }
                Xor(move.To, ref BTree.BB_Piece[color ^ 1, (int)move.CapPiece]);
                Xor(move.To, ref BTree.BB_Occupied[color ^ 1]);
                BTree.ShogiBoard[move.To] = sign * (int)move.CapPiece;
            }
            else
            {
                BTree.ShogiBoard[move.To] = (int)Piece.Type.Empty;
            }

            BTree.Hash[BTree.ply] = 0;
            BTree.PrevHash = BTree.Hash[--BTree.ply - 1];
        }

        // MCSのプレイアウトで3手詰めのハッシュテーブルを使わない場合、この方が少し速くなる
        // ※バグがある
        public static void DoNoHash(ref BoardTree BTree, int color, Move move)
        {
            int sign = -1;
            if (color == (int)Color.Type.White)
                sign = 1;

            //BTree.PrevHash = BTree.CurrentHash;

            if (move.From == NSquare)
            {
                switch (move.PieceType)
                {
                    case Piece.Type.Pawn:
                        Xor(move.To + sign * NFile, ref BTree.BB_PawnAttacks[color]);
                        break;
                    case Piece.Type.Gold:
                        Xor(move.To, ref BTree.BB_Total_Gold[color]);
                        break;
                    case Piece.Type.Bishop:
                        Xor(move.To, ref BTree.BB_BH[color]);
                        break;
                    case Piece.Type.Rook:
                        Xor(move.To, ref BTree.BB_RD[color]);
                        break;
                }

                Xor(move.To, ref BTree.BB_Piece[color, (int)move.PieceType]);
                //BTree.CurrentHash ^= Hash.PieceRand[color, (int)move.PieceType, move.To];
                BTree.Hand[color, (int)move.PieceType] -= 1;
                BTree.ShogiBoard[move.To] = -sign * (int)move.PieceType;
                Xor(move.To, ref BTree.BB_Occupied[color]);
            }
            else
            {
                BitBoard bb_set_clear = new BitBoard();
                BBOr(ref bb_set_clear, AbbMask[move.From], AbbMask[move.To]);
                BBXor(ref BTree.BB_Occupied[color], BTree.BB_Occupied[color], bb_set_clear);
                BTree.ShogiBoard[move.From] = (int)Piece.Type.Empty;
                if (move.FlagPromo == 1)
                {
                    switch (move.PieceType)
                    {
                        case Piece.Type.Pawn:
                            Xor(move.To, ref BTree.BB_PawnAttacks[color]);
                            Xor(move.To, ref BTree.BB_Total_Gold[color]);
                            break;
                        case Piece.Type.Lance:
                        case Piece.Type.Knight:
                        case Piece.Type.Silver:
                            Xor(move.To, ref BTree.BB_Total_Gold[color]);
                            break;
                        case Piece.Type.Bishop:
                            Xor(move.To, ref BTree.BB_HDK[color]);
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            break;
                        case Piece.Type.Rook:
                            Xor(move.To, ref BTree.BB_HDK[color]);
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            break;
                    }
                    Xor(move.From, ref BTree.BB_Piece[color, (int)move.PieceType]);
                    Xor(move.To, ref BTree.BB_Piece[color, (int)(move.PieceType + Piece.PromoteNum)]);
                    //BTree.CurrentHash ^= PieceRand[color, (int)(move.PieceType + Piece.PromoteNum), move.To] ^ PieceRand[color, (int)move.PieceType, move.From];
                    BTree.ShogiBoard[move.To] = -sign * ((int)move.PieceType + Piece.PromoteNum);
                }
                else
                {
                    switch (move.PieceType)
                    {
                        case Piece.Type.Pawn:
                            Xor(move.To + sign * NFile, ref BTree.BB_PawnAttacks[color]);
                            Xor(move.To, ref BTree.BB_PawnAttacks[color]);
                            break;
                        case Piece.Type.Gold:
                        case Piece.Type.Pro_Pawn:
                        case Piece.Type.Pro_Lance:
                        case Piece.Type.Pro_Knight:
                        case Piece.Type.Pro_Silver:
                            BBXor(ref BTree.BB_Total_Gold[color], BTree.BB_Total_Gold[color], bb_set_clear);
                            break;
                        case Piece.Type.Bishop:
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            break;
                        case Piece.Type.Rook:
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            break;
                        case Piece.Type.King:
                            BTree.SQ_King[color] = move.To;
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                        case Piece.Type.Horse:
                            BBXor(ref BTree.BB_BH[color], BTree.BB_BH[color], bb_set_clear);
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                        case Piece.Type.Dragon:
                            BBXor(ref BTree.BB_RD[color], BTree.BB_RD[color], bb_set_clear);
                            BBXor(ref BTree.BB_HDK[color], BTree.BB_HDK[color], bb_set_clear);
                            break;
                    }
                    BBXor(ref BTree.BB_Piece[color, (int)move.PieceType], BTree.BB_Piece[color, (int)move.PieceType], bb_set_clear);
                    //BTree.CurrentHash ^= PieceRand[color, (int)move.PieceType, move.To] ^ PieceRand[color, (int)move.PieceType, move.From];
                    BTree.ShogiBoard[move.To] = -sign * (int)move.PieceType;
                }
            }

            if ((int)move.CapPiece != 0)
            {
                switch (move.CapPiece)
                {
                    case Piece.Type.Pawn:
                        Xor(move.To - sign * NFile, ref BTree.BB_PawnAttacks[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Lance:
                    case Piece.Type.Knight:
                    case Piece.Type.Silver:
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Gold:
                        Xor(move.To, ref BTree.BB_Total_Gold[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Bishop:
                        Xor(move.To, ref BTree.BB_BH[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Rook:
                        Xor(move.To, ref BTree.BB_RD[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece] += 1;
                        break;
                    case Piece.Type.Pro_Pawn:
                    case Piece.Type.Pro_Lance:
                    case Piece.Type.Pro_Knight:
                    case Piece.Type.Pro_Silver:
                        Xor(move.To, ref BTree.BB_Total_Gold[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] += 1;
                        break;
                    case Piece.Type.Horse:
                        Xor(move.To, ref BTree.BB_HDK[color ^ 1]);
                        Xor(move.To, ref BTree.BB_BH[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] += 1;
                        break;
                    case Piece.Type.Dragon:
                        Xor(move.To, ref BTree.BB_HDK[color ^ 1]);
                        Xor(move.To, ref BTree.BB_RD[color ^ 1]);
                        BTree.Hand[color, (int)move.CapPiece - Piece.PromoteNum] += 1;
                        break;
                }
                Xor(move.To, ref BTree.BB_Piece[color ^ 1, (int)move.CapPiece]);
                //BTree.CurrentHash ^= PieceRand[color ^ 1, (int)move.CapPiece, move.To];
                Xor(move.To, ref BTree.BB_Occupied[color ^ 1]);
            }

            //BTree.Hash[++BTree.ply] = BTree.CurrentHash;
        }
    }
}
