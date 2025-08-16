using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Net.NetworkInformation;
using System.Text;
using System.Threading.Tasks;
using static Asklepios.Board;
using static Asklepios.ShogiCommon;
using static Asklepios.Color;
using static Asklepios.Piece;
using static Asklepios.Direction;
using static Asklepios.BitOperation;

namespace Asklepios
{
    internal class AttackBitBoard
    {
        private const int HashLength = 128;
        private const int ShiftTableLength = 2;
        public static BitBoard[,,] AbbPieceAttacks = new BitBoard[ColorNum, PieceNum, NSquare];
        public static BitBoard[,] AbbRankAttacks = new BitBoard[NSquare, HashLength]; // '-'
        public static BitBoard[,] AbbDiag1Attacks = new BitBoard[NSquare, HashLength]; // '/'
        public static BitBoard[,] AbbDiag2Attacks = new BitBoard[NSquare, HashLength]; // '\'
        public static BitBoard[,] AbbFileAttacks = new BitBoard[NSquare, HashLength]; // '|'
        public static BitBoard[] AbbRankMaskEx = new BitBoard[NSquare]; // '-'
        public static BitBoard[] AbbDiag1MaskEx = new BitBoard[NSquare]; // '/'
        public static BitBoard[] AbbDiag2MaskEx = new BitBoard[NSquare]; // '\'
        public static BitBoard[] AbbFileMaskEx = new BitBoard[NSquare];// '|'
        public static BitBoard[,] AbbPlusMinusRays = new BitBoard[ColorNum, NSquare];
        public static BitBoard[] AbbMask = new BitBoard[NSquare];
        public static uint[,] Diag1ShiftTable = new uint[NSquare, ShiftTableLength];
        public static uint[,] Diag2ShiftTable = new uint[NSquare, ShiftTableLength];
        public static uint[,] FileShiftTable = new uint[NSquare, ShiftTableLength];
        public static BitBoard[,] AbbObstacle = new BitBoard[NSquare, NSquare];
        public static uint[,] ADirec = new uint[NSquare, NSquare];

        public static void Xor(int sq, ref BitBoard BB)
        {
            BB.p[0] ^= AbbMask[sq].p[0];
            BB.p[1] ^= AbbMask[sq].p[1];
            BB.p[2] ^= AbbMask[sq].p[2];
        }

        public static void IniTables()
        {
            for (int isquare = 0; isquare < NSquare; isquare++)
            {
                AbbMask[isquare] = BBSetMask(isquare);
            }

            for (int c = 0; c < ColorNum; c++)
            {
                for (int pc = 0; pc < PieceNum; pc++)
                {
                    for (int sq = 0; sq < NSquare; sq++)
                    {
                        int irank = (int)AiRank[sq];
                        int ifile = (int)AiFile[sq];
                        BitBoard bb = new BitBoard();
                        BBIni(ref bb);
                        if (c == 0)
                        {
                            switch (pc)
                            {
                                case (int)Piece.Type.Pawn:
                                    SetAttacks(irank - 1, ifile, ref bb);
                                    break;
                                case (int)Piece.Type.Knight:
                                    SetAttacks(irank - 2, ifile - 1, ref bb);
                                    SetAttacks(irank - 2, ifile + 1, ref bb);
                                    break;
                                case (int)Piece.Type.Silver:
                                    SetAttacks(irank + 1, ifile - 1, ref bb);
                                    SetAttacks(irank + 1, ifile + 1, ref bb);
                                    SetAttacks(irank - 1, ifile, ref bb);
                                    SetAttacks(irank - 1, ifile - 1, ref bb);
                                    SetAttacks(irank - 1, ifile + 1, ref bb);
                                    break;
                                case (int)Piece.Type.Gold:
                                case (int)Piece.Type.Pro_Pawn:
                                case (int)Piece.Type.Pro_Lance:
                                case (int)Piece.Type.Pro_Knight:
                                case (int)Piece.Type.Pro_Silver:
                                    SetAttacks(irank + 1, ifile, ref bb);
                                    SetAttacks(irank - 1, ifile - 1, ref bb);
                                    SetAttacks(irank - 1, ifile + 1, ref bb);
                                    SetAttacks(irank - 1, ifile, ref bb);
                                    SetAttacks(irank, ifile - 1, ref bb);
                                    SetAttacks(irank, ifile + 1, ref bb);
                                    break;
                                case (int)Piece.Type.King:
                                    SetAttacks(irank + 1, ifile - 1, ref bb);
                                    SetAttacks(irank + 1, ifile + 1, ref bb);
                                    SetAttacks(irank + 1, ifile, ref bb);
                                    SetAttacks(irank - 1, ifile - 1, ref bb);
                                    SetAttacks(irank - 1, ifile + 1, ref bb);
                                    SetAttacks(irank - 1, ifile, ref bb);
                                    SetAttacks(irank, ifile - 1, ref bb);
                                    SetAttacks(irank, ifile + 1, ref bb);
                                    break;
                            }
                        }
                        else
                        {
                            switch (pc)
                            {
                                case (int)Piece.Type.Pawn:
                                    SetAttacks(irank + 1, ifile, ref bb);
                                    break;
                                case (int)Piece.Type.Knight:
                                    SetAttacks(irank + 2, ifile - 1, ref bb);
                                    SetAttacks(irank + 2, ifile + 1, ref bb);
                                    break;
                                case (int)Piece.Type.Silver:
                                    SetAttacks(irank + 1, ifile - 1, ref bb);
                                    SetAttacks(irank + 1, ifile + 1, ref bb);
                                    SetAttacks(irank + 1, ifile, ref bb);
                                    SetAttacks(irank - 1, ifile - 1, ref bb);
                                    SetAttacks(irank - 1, ifile + 1, ref bb);
                                    break;
                                case (int)Piece.Type.Gold:
                                case (int)Piece.Type.Pro_Pawn:
                                case (int)Piece.Type.Pro_Lance:
                                case (int)Piece.Type.Pro_Knight:
                                case (int)Piece.Type.Pro_Silver:
                                    SetAttacks(irank + 1, ifile - 1, ref bb);
                                    SetAttacks(irank + 1, ifile + 1, ref bb);
                                    SetAttacks(irank + 1, ifile, ref bb);
                                    SetAttacks(irank - 1, ifile, ref bb);
                                    SetAttacks(irank, ifile - 1, ref bb);
                                    SetAttacks(irank, ifile + 1, ref bb);
                                    break;
                                case (int)Piece.Type.King:
                                    SetAttacks(irank + 1, ifile - 1, ref bb);
                                    SetAttacks(irank + 1, ifile + 1, ref bb);
                                    SetAttacks(irank + 1, ifile, ref bb);
                                    SetAttacks(irank - 1, ifile - 1, ref bb);
                                    SetAttacks(irank - 1, ifile + 1, ref bb);
                                    SetAttacks(irank - 1, ifile, ref bb);
                                    SetAttacks(irank, ifile - 1, ref bb);
                                    SetAttacks(irank, ifile + 1, ref bb);
                                    break;
                            }
                        }
                        AbbPieceAttacks[c, pc, sq] = bb;
                    }
                }
            }
        }

        public static void IniLongAttacks()
        {
            string AppPath = AppDomain.CurrentDomain.SetupInformation.ApplicationBase;
            string FilePath = AppPath + "\\abb_rank_mask_ex.txt";
            string line;
            StreamReader sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[1]);
                bb.p[1] = ulong.Parse(s[2]);
                bb.p[2] = ulong.Parse(s[3]);
                AbbRankMaskEx[sq] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_diag1_mask_ex.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[1]);
                bb.p[1] = ulong.Parse(s[2]);
                bb.p[2] = ulong.Parse(s[3]);
                AbbDiag1MaskEx[sq] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_diag2_mask_ex.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[1]);
                bb.p[1] = ulong.Parse(s[2]);
                bb.p[2] = ulong.Parse(s[3]);
                AbbDiag2MaskEx[sq] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_file_mask_ex.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[1]);
                bb.p[1] = ulong.Parse(s[2]);
                bb.p[2] = ulong.Parse(s[3]);
                AbbFileMaskEx[sq] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_rank_attacks.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                int hash_value = int.Parse(s[1]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[2]);
                bb.p[1] = ulong.Parse(s[3]);
                bb.p[2] = ulong.Parse(s[4]);
                AbbRankAttacks[sq, hash_value] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_diag1_attacks.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                int hash_value = int.Parse(s[1]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[2]);
                bb.p[1] = ulong.Parse(s[3]);
                bb.p[2] = ulong.Parse(s[4]);
                AbbDiag1Attacks[sq, hash_value] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_diag2_attacks.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                int hash_value = int.Parse(s[1]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[2]);
                bb.p[1] = ulong.Parse(s[3]);
                bb.p[2] = ulong.Parse(s[4]);
                AbbDiag2Attacks[sq, hash_value] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_file_attacks.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                int hash_value = int.Parse(s[1]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[2]);
                bb.p[1] = ulong.Parse(s[3]);
                bb.p[2] = ulong.Parse(s[4]);
                AbbFileAttacks[sq, hash_value] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_plus_rays.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[1]);
                bb.p[1] = ulong.Parse(s[2]);
                bb.p[2] = ulong.Parse(s[3]);
                AbbPlusMinusRays[(int)Color.Type.White, sq] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\abb_minus_rays.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                BitBoard bb = new BitBoard();
                BBIni(ref bb);
                bb.p[0] = ulong.Parse(s[1]);
                bb.p[1] = ulong.Parse(s[2]);
                bb.p[2] = ulong.Parse(s[3]);
                AbbPlusMinusRays[(int)Color.Type.Black, sq] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\diag1_shift_table.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                int length = int.Parse(s[1]);
                Diag1ShiftTable[sq, length] = uint.Parse(s[2]);
            }
            sr.Close();

            FilePath = AppPath + "\\diag2_shift_table.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                int length = int.Parse(s[1]);
                Diag2ShiftTable[sq, length] = uint.Parse(s[2]);
            }
            sr.Close();

            FilePath = AppPath + "\\file_shift_table.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq = int.Parse(s[0]);
                int length = int.Parse(s[1]);
                FileShiftTable[sq, length] = uint.Parse(s[2]);
            }
            sr.Close();

            FilePath = AppPath + "\\abb_obstacle.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq0 = int.Parse(s[0]);
                int sq1 = int.Parse(s[1]);
                BitBoard bb = new BitBoard();
                bb.p[0] = ulong.Parse(s[2]);
                bb.p[1] = ulong.Parse(s[3]);
                bb.p[2] = ulong.Parse(s[4]);
                AbbObstacle[sq0, sq1] = bb;
            }
            sr.Close();

            FilePath = AppPath + "\\adirec.txt";
            sr = new StreamReader(FilePath, Encoding.UTF8);
            while ((line = sr.ReadLine()) != null)
            {
                string[] s = line.Split(',');
                int sq0 = int.Parse(s[0]);
                int sq1 = int.Parse(s[1]);
                uint Value = uint.Parse(s[2]);
                ADirec[sq0, sq1] = Value;
            }
            sr.Close();
        }

        private static void SetAttacks(int irank, int ifile, ref BitBoard BB)
        {
            if (irank >= (int)Rank.rank1 && irank <= (int)Rank.rank9 &&
                ifile >= (int)ShogiCommon.File.file1 && ifile <= (int)ShogiCommon.File.file9)
            {
                Xor(irank * NFile + ifile, ref BB);
            }
        }

        private static BitBoard BBSetMask(int sq)
        {
            BitBoard bb = new BitBoard();

            BBIni(ref bb);

            if (sq > 53) 
            { 
                bb.p[2] = 1U << (80 - sq); 
            }
            else if (sq > 26) 
            { 
                bb.p[1] = 1U << (53 - sq); 
            }
            else 
            {
                bb.p[0] = 1U << (26 - sq); 
            }

            return bb;
        }

        public static BitBoard GetFileAttacks(BitBoard bb_occupied, int sq)
        {
            ulong hash;
            ulong[] temp_hash = new ulong[3];

            temp_hash[0] = PextU64(bb_occupied.p[0], AbbFileMaskEx[sq].p[0]);
            temp_hash[1] = PextU64(bb_occupied.p[1], AbbFileMaskEx[sq].p[1]);
            temp_hash[2] = PextU64(bb_occupied.p[2], AbbFileMaskEx[sq].p[2]);

            hash = (temp_hash[0] << (int)FileShiftTable[sq, 0]) | (temp_hash[1] << (int)FileShiftTable[sq, 1]) | (temp_hash[2]);

            return AbbFileAttacks[sq, hash];
        }

        public static BitBoard GetRankAttacks(BitBoard bb_occupied, int sq)
        {
            ulong hash = new ulong();

            switch (AiRank[sq])
            {
                case (uint)Rank.rank1:
                case (uint)Rank.rank2:
                case (uint)Rank.rank3:
                    hash = PextU64(bb_occupied.p[0], AbbRankMaskEx[sq].p[0]);
                    break;
                case (uint)Rank.rank4:
                case (uint)Rank.rank5:
                case (uint)Rank.rank6:
                    hash = PextU64(bb_occupied.p[1], AbbRankMaskEx[sq].p[1]);
                    break;
                case (uint)Rank.rank7:
                case (uint)Rank.rank8:
                case (uint)Rank.rank9:
                    hash = PextU64(bb_occupied.p[2], AbbRankMaskEx[sq].p[2]);
                    break;
            }

            return AbbRankAttacks[sq, hash];
        }

        public static BitBoard GetRookAttacks(BitBoard bb_occupied, int sq)
        {
            BitBoard bb = new BitBoard();
            BBOr(ref bb, GetRankAttacks(bb_occupied, sq), GetFileAttacks(bb_occupied, sq));
            return bb;
        }

        public static BitBoard GetDragonAttacks(BitBoard bb_occupied, int sq)
        {
            BitBoard bb = GetRookAttacks(bb_occupied, sq);
            BBOr(ref bb, bb, AbbPieceAttacks[0, (int)Piece.Type.King, sq]);// 玉の利きは先後同一
            return bb;
        }

        public static BitBoard GetDiag1Attacks(BitBoard bb_occupied, int sq)
        {
            ulong hash;
            ulong[] temp_hash = new ulong[3];

            temp_hash[0] = PextU64(bb_occupied.p[0], AbbDiag1MaskEx[sq].p[0]);
            temp_hash[1] = PextU64(bb_occupied.p[1], AbbDiag1MaskEx[sq].p[1]);
            temp_hash[2] = PextU64(bb_occupied.p[2], AbbDiag1MaskEx[sq].p[2]);

            hash = (temp_hash[0] << (int)Diag1ShiftTable[sq, 0]) | (temp_hash[1] << (int)Diag1ShiftTable[sq, 1]) | (temp_hash[2]);

            return AbbDiag1Attacks[sq, hash];
        }

        public static BitBoard GetDiag2Attacks(BitBoard bb_occupied, int sq)
        {
            ulong hash;
            ulong[] temp_hash = new ulong[3];

            temp_hash[0] = PextU64(bb_occupied.p[0], AbbDiag2MaskEx[sq].p[0]);
            temp_hash[1] = PextU64(bb_occupied.p[1], AbbDiag2MaskEx[sq].p[1]);
            temp_hash[2] = PextU64(bb_occupied.p[2], AbbDiag2MaskEx[sq].p[2]);

            hash = (temp_hash[0] << (int)Diag2ShiftTable[sq, 0]) | (temp_hash[1] << (int)Diag2ShiftTable[sq, 1]) | (temp_hash[2]);

            return AbbDiag2Attacks[sq, hash];
        }

        public static BitBoard GetBishopAttacks(BitBoard bb_occupied, int sq)
        {
            BitBoard bb = new BitBoard();
            BBOr(ref bb, GetDiag1Attacks(bb_occupied, sq), GetDiag2Attacks(bb_occupied, sq));
            return bb;
        }

        public static BitBoard GetHorseAttacks(BitBoard bb_occupied, int sq)
        {
            BitBoard bb = GetBishopAttacks(bb_occupied, sq);
            BBOr(ref bb, bb, AbbPieceAttacks[0, (int)Piece.Type.King, sq]);// 玉の利きは先後同一
            return bb;
        }

        public static ulong IsPinnedOnKing(BoardTree BTree, int isquare, int idirec, int color)
        {
            BitBoard bb_occupied = new BitBoard();
            BitBoard bb_attacks = new BitBoard();
            BBOr(ref bb_occupied, BTree.BB_Occupied[0], BTree.BB_Occupied[1]);
            switch(idirec)
            {
                case (int)DirectionType.Rank:
                    bb_attacks = BitBoard.Copy(GetRankAttacks(bb_occupied, isquare));
                    if (BBContract(bb_attacks, BTree.BB_Piece[color, (int)Piece.Type.King]) != 0)
                    {
                        return BBContract(bb_attacks, BTree.BB_RD[color ^ 1]);
                    }
                    break;
                case (int)DirectionType.File:
                    bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, isquare));
                    if (BBContract(bb_attacks, BTree.BB_Piece[color, (int)Piece.Type.King]) != 0)
                    {
                        BitBoard bb_attacker = new BitBoard();
                        BBAnd(ref bb_attacker, BTree.BB_Piece[color ^ 1, (int)Piece.Type.Lance], AbbPlusMinusRays[color, isquare]);
                        BBOr(ref bb_attacker, bb_attacker, BTree.BB_RD[color ^ 1]);
                        return BBContract(bb_attacks, bb_attacker);
                    }
                    break;
                case (int)DirectionType.Diag1:
                    bb_attacks = BitBoard.Copy(GetDiag1Attacks(bb_occupied, isquare));
                    if (BBContract(bb_attacks, BTree.BB_Piece[color, (int)Piece.Type.King]) != 0)
                    {
                        return BBContract(bb_attacks, BTree.BB_BH[color ^ 1]);
                    }
                    break;
                case (int)DirectionType.Diag2:
                    bb_attacks = BitBoard.Copy(GetDiag2Attacks(bb_occupied, isquare));
                    if (BBContract(bb_attacks, BTree.BB_Piece[color, (int)Piece.Type.King]) != 0)
                    {
                        return BBContract(bb_attacks, BTree.BB_BH[color ^ 1]);
                    }
                    break;
            }
            return 0;
        }

        /*public static BitBoard AttacksToPiece(BoardTree BTree, int sq, int color)
        {
            int delta = -9;
            if (color == 0) { delta = -delta; }
            BitBoard bb_occupied = new BitBoard();
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            BitBoard bb_ret, bb_attacks, bb;
            bb_ret = new BitBoard();
            if (sq + delta >= 0 && sq + delta < NSquare)
                BBAnd(ref bb_ret, BTree.BB_Piece[color, (int)Piece.Type.Pawn], AbbMask[sq + delta]);
            BBAndOr(ref bb_ret, BTree.BB_Piece[color, (int)Piece.Type.Knight], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq]);
            BBAndOr(ref bb_ret, BTree.BB_Piece[color, (int)Piece.Type.Silver], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq]);
            BBAndOr(ref bb_ret, BTree.BB_Total_Gold[color], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq]);
            BBAndOr(ref bb_ret, BTree.BB_HDK[color], AbbPieceAttacks[color ^ 1, (int)Piece.Type.King, sq]);
            bb_attacks = BitBoard.Copy(GetBishopAttacks(bb_occupied, sq));
            BBAndOr(ref bb_ret, BTree.BB_BH[color], bb_attacks);
            bb = BitBoard.Copy(BTree.BB_RD[color]);
            bb_attacks = BitBoard.Copy(GetRankAttacks(bb_occupied, sq));
            BBAndOr(ref bb_ret, bb, bb_attacks);
            BBAndOr(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Lance], AbbPlusMinusRays[color ^ 1, sq]);
            bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, sq));
            BBAndOr(ref bb_ret, bb, bb_attacks);
            return bb_ret;
        }*/

        public static BitBoard AttacksToPiece(BoardTree BTree, int sq, int color)
        {
            int delta = -9;
            if (color == 0) { delta = -delta; }
            BitBoard bb_occupied = new BitBoard();
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            BitBoard bb_ret, bb_attacks, bb;
            bb_ret = new BitBoard();
            BBAnd(ref bb_ret, BTree.BB_Piece[color, (int)Piece.Type.Pawn], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Pawn, sq]);
            BBAndOr(ref bb_ret, BTree.BB_Piece[color, (int)Piece.Type.Knight], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq]);
            BBAndOr(ref bb_ret, BTree.BB_Piece[color, (int)Piece.Type.Silver], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq]);
            BBAndOr(ref bb_ret, BTree.BB_Total_Gold[color], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq]);
            BBAndOr(ref bb_ret, BTree.BB_HDK[color], AbbPieceAttacks[color ^ 1, (int)Piece.Type.King, sq]);
            bb_attacks = BitBoard.Copy(GetBishopAttacks(bb_occupied, sq));
            BBAndOr(ref bb_ret, BTree.BB_BH[color], bb_attacks);
            bb = BitBoard.Copy(BTree.BB_RD[color]);
            bb_attacks = BitBoard.Copy(GetRankAttacks(bb_occupied, sq));
            BBAndOr(ref bb_ret, bb, bb_attacks);
            BBAndOr(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Lance], AbbPlusMinusRays[color ^ 1, sq]);
            bb_attacks = BitBoard.Copy(GetFileAttacks(bb_occupied, sq));
            BBAndOr(ref bb_ret, bb, bb_attacks);
            return bb_ret;
        }

        public static ulong IsAttacked(BoardTree BTree, int sq, int color)
        {
            BitBoard bb = new BitBoard();
            BitBoard bb_occupied = new BitBoard();
            BitBoard bb_atk;
            BitBoard bb1;
            BBAnd(ref bb, BTree.BB_PawnAttacks[color ^ 1], AbbMask[sq]);
            BBAndOr(ref bb, BTree.BB_Piece[color ^ 1, (int)Piece.Type.Knight], AbbPieceAttacks[color, (int)Piece.Type.Knight, sq]);
            BBAndOr(ref bb, BTree.BB_Piece[color ^ 1, (int)Piece.Type.Silver], AbbPieceAttacks[color, (int)Piece.Type.Silver, sq]);
            BBAndOr(ref bb, BTree.BB_Total_Gold[color ^ 1], AbbPieceAttacks[color, (int)Piece.Type.Gold, sq]);
            BBAndOr(ref bb, BTree.BB_HDK[color ^ 1], AbbPieceAttacks[color, (int)Piece.Type.King, sq]);
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            bb_atk = BitBoard.Copy(GetBishopAttacks(bb_occupied, sq));
            BBAndOr(ref bb, BTree.BB_BH[color ^ 1], bb_atk);
            bb1 = BitBoard.Copy(BTree.BB_RD[color ^ 1]);
            BBAndOr(ref bb1, BTree.BB_Piece[color ^ 1, (int)Piece.Type.Lance], AbbPlusMinusRays[color, sq]);
            bb_atk = BitBoard.Copy(GetFileAttacks(bb_occupied, sq));
            BBAndOr(ref bb, bb1, bb_atk);
            bb_atk = BitBoard.Copy(GetRankAttacks(bb_occupied, sq));
            BBAndOr(ref bb, BTree.BB_RD[color ^ 1], bb_atk);
            return BBTest(bb);
        }

        public static BitBoard TempIsAttacked(BoardTree BTree, int sq, int color)
        {
            BitBoard bb = new BitBoard();
            BitBoard bb_occupied = new BitBoard();
            BitBoard bb_atk;
            BitBoard bb1;
            BBAnd(ref bb, BTree.BB_PawnAttacks[color ^ 1], AbbMask[sq]);
            BBAndOr(ref bb, BTree.BB_Piece[color ^ 1, (int)Piece.Type.Knight], AbbPieceAttacks[color, (int)Piece.Type.Knight, sq]);
            BBAndOr(ref bb, BTree.BB_Piece[color ^ 1, (int)Piece.Type.Silver], AbbPieceAttacks[color, (int)Piece.Type.Silver, sq]);
            BBAndOr(ref bb, BTree.BB_Total_Gold[color ^ 1], AbbPieceAttacks[color, (int)Piece.Type.Gold, sq]);
            BBAndOr(ref bb, BTree.BB_HDK[color ^ 1], AbbPieceAttacks[color, (int)Piece.Type.King, sq]);
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            bb_atk = BitBoard.Copy(GetBishopAttacks(bb_occupied, sq));
            BBAndOr(ref bb, BTree.BB_BH[color ^ 1], bb_atk);
            bb1 = BitBoard.Copy(BTree.BB_RD[color ^ 1]);
            BBAndOr(ref bb1, BTree.BB_Piece[color ^ 1, (int)Piece.Type.Lance], AbbPlusMinusRays[color, sq]);
            bb_atk = BitBoard.Copy(GetFileAttacks(bb_occupied, sq));
            BBAndOr(ref bb, bb1, bb_atk);
            bb_atk = BitBoard.Copy(GetRankAttacks(bb_occupied, sq));
            BBAndOr(ref bb, BTree.BB_RD[color ^ 1], bb_atk);
            return bb;
        }

        public static int IsDiscoverKing(BoardTree BTree, int ifrom, int ito, int color)
        {
            int idirec = (int)ADirec[BTree.SQ_King[color], ifrom];
            if (idirec != 0 && idirec != ADirec[BTree.SQ_King[color], ito] && IsPinnedOnKing(BTree, ifrom, idirec, color) != 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        public static int IsMatePawnDrop(ref BoardTree BTree, int sq_drop, int color)
        {
            if (color == (int)Color.Type.White)
            {
                if (sq_drop - 9 > 0 && BTree.ShogiBoard[sq_drop - 9] != -(int)Piece.Type.King)
                    return 0;
            }
            else
            {
                if (sq_drop + 9 < NSquare && BTree.ShogiBoard[sq_drop + 9] != (int)Piece.Type.King)
                    return 0;
            }

            BitBoard bb_sum = new BitBoard();
            BitBoard bb;
            BitBoard bb_occupied = new BitBoard();

            BBAnd(ref bb_sum, BTree.BB_Piece[color, (int)Piece.Type.Knight], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq_drop]);
            BBAndOr(ref bb_sum, BTree.BB_Piece[color, (int)Piece.Type.Silver], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_drop]);
            BBAndOr(ref bb_sum, BTree.BB_Total_Gold[color], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_drop]);
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            bb = BitBoard.Copy(GetBishopAttacks(bb_occupied, sq_drop));
            BBAndOr(ref bb_sum, BTree.BB_BH[color], bb);
            bb = BitBoard.Copy(GetRookAttacks(bb_occupied, sq_drop));
            BBAndOr(ref bb_sum, BTree.BB_RD[color], bb);
            BBOr(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Horse], BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            BBAndOr(ref bb_sum, bb, AbbPieceAttacks[color, (int)Piece.Type.King, sq_drop]);
            while(BBTest(bb_sum) != 0)
            {
                int ifrom = LastOne012(bb_sum.p[0], bb_sum.p[1], bb_sum.p[2]);
                Xor(ifrom, ref bb_sum);
                if (IsDiscoverKing(BTree, ifrom, sq_drop, color) != 0)
                    continue;
                return 0;
            }

            int iking = BTree.SQ_King[color];
            int iret = 1;
            Xor(sq_drop, ref BTree.BB_Occupied[color ^ 1]);
            BitBoard bb_move = new BitBoard();
            BBNotAnd(ref bb_move, AbbPieceAttacks[color, (int)Piece.Type.King, iking], BTree.BB_Occupied[color]);
            while (BBTest(bb_move) != 0)
            {
                int ito = LastOne012(bb_move.p[0], bb_move.p[1], bb_move.p[2]);
                if (IsAttacked(BTree, ito, color) == 0)
                {
                    iret = 0;
                    break;
                }
                Xor(ito, ref bb_move);
            }
            Xor(sq_drop, ref BTree.BB_Occupied[color ^ 1]);
            return iret;
        }

        /*public static int IsMatePawnDrop(ref BoardTree BTree, int sq_drop, int color)
        {
            if (color == (int)Color.Type.Black)
            {
                if (BTree.ShogiBoard[sq_drop - 9] != -(int)Piece.Type.King)
                    return 1;
            }
            else
            {
                if (BTree.ShogiBoard[sq_drop + 9] != (int)Piece.Type.King)
                    return 1;
            }

            BitBoard bb_sum = new BitBoard();
            BitBoard bb;
            BitBoard bb_occupied = new BitBoard();

            BBAnd(ref bb_sum, BTree.BB_Piece[color, (int)Piece.Type.Knight], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Knight, sq_drop]);
            BBAndOr(ref bb_sum, BTree.BB_Piece[color, (int)Piece.Type.Silver], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Silver, sq_drop]);
            BBAndOr(ref bb_sum, BTree.BB_Total_Gold[color], AbbPieceAttacks[color ^ 1, (int)Piece.Type.Gold, sq_drop]);
            BBOr(ref bb_occupied, BTree.BB_Occupied[color], BTree.BB_Occupied[color ^ 1]);
            bb = BitBoard.Copy(GetBishopAttacks(bb_occupied, sq_drop));
            BBAndOr(ref bb_sum, BTree.BB_BH[color], bb);
            bb = BitBoard.Copy(GetRookAttacks(bb_occupied, sq_drop));
            BBAndOr(ref bb_sum, BTree.BB_RD[color], bb);
            BBOr(ref bb, BTree.BB_Piece[color, (int)Piece.Type.Horse], BTree.BB_Piece[color, (int)Piece.Type.Dragon]);
            BBAndOr(ref bb_sum, bb, AbbPieceAttacks[color, (int)Piece.Type.King, sq_drop]);
            while (BBTest(bb_sum) != 0)
            {
                int ifrom = LastOne012(bb_sum.p[0], bb_sum.p[1], bb_sum.p[2]);
                Xor(ifrom, ref bb_sum);
                if (IsDiscoverKing(BTree, ifrom, sq_drop, color) != 0)
                    continue;
                return 0;
            }

            int iking = BTree.SQ_King[color];
            int iret = 1;
            Xor(sq_drop, ref BTree.BB_Occupied[color ^ 1]);
            BitBoard bb_move = new BitBoard();
            BBNotAnd(ref bb_move, AbbPieceAttacks[color, (int)Piece.Type.King, iking], BTree.BB_Occupied[color]);
            while (BBTest(bb_move) != 0)
            {
                int ito = LastOne012(bb_move.p[0], bb_move.p[1], bb_move.p[2]);
                if (IsAttacked(BTree, ito, color) == 0)
                {
                    iret = 0;
                    break;
                }
                Xor(ito, ref bb_move);
            }
            Xor(sq_drop, ref BTree.BB_Occupied[color ^ 1]);
            return iret;
        }*/
    }
}
