using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class ShogiCommon
    {
        public const int NSquare = 81;
        public const int NRank = 9;
        public const int NFile = 9;
        public const int NHand = 8;
        public const int PlyMax = 1024;
        public const int MoveMax = 700;

        public static ulong[] MaskFile = { (1U << 18 | 1U << 9 | 1U) << 8, 
                                           (1U << 18 | 1U << 9 | 1U) << 7, 
                                           (1U << 18 | 1U << 9 | 1U) << 6, 
                                           (1U << 18 | 1U << 9 | 1U) << 5, 
                                           (1U << 18 | 1U << 9 | 1U) << 4, 
                                           (1U << 18 | 1U << 9 | 1U) << 3, 
                                           (1U << 18 | 1U << 9 | 1U) << 2, 
                                           (1U << 18 | 1U << 9 | 1U) << 1, 
                                            1U << 18 | 1U << 9 | 1U };
        public static ulong[] MaskRank = { 0x7FC0000, 0x3FE00, 0x1FF, 0x7FC0000, 0x3FE00, 0x1FF, 0x7FC0000, 0x3FE00, 0x1FF };
        public enum Square {
            A9 = 0, B9, C9, D9, E9, F9, G9, H9, I9,
            A8, B8, C8, D8, E8, F8, G8, H8, I8,
            A7, B7, C7, D7, E7, F7, G7, H7, I7,
            A6, B6, C6, D6, E6, F6, G6, H6, I6,
            A5, B5, C5, D5, E5, F5, G5, H5, I5,
            A4, B4, C4, D4, E4, F4, G4, H4, I4,
            A3, B3, C3, D3, E3, F3, G3, H3, I3,
            A2, B2, C2, D2, E2, F2, G2, H2, I2,
            A1, B1, C1, D1, E1, F1, G1, H1, I1
        };
        public enum Rank
        {
            rank1 = 0, rank2, rank3, rank4, rank5, rank6, rank7, rank8, rank9
        };

        public enum File
        {
            file1 = 0, file2, file3, file4, file5, file6, file7, file8, file9
        };

        public enum EnumBit
        {
            b0000, b0001, b0010, b0011, b0100, b0101, b0110, b0111,
            b1000, b1001, b1010, b1011, b1100, b1101, b1110, b1111
        };

        public enum Direction
        {
            direc_misc = EnumBit.b0000,
            direc_file = EnumBit.b0010, /* | */
            direc_rank = EnumBit.b0011, /* - */
            direc_diag1 = EnumBit.b0100, /* / */
            direc_diag2 = EnumBit.b0101, /* \ */
            flag_cross = EnumBit.b0010,
            flag_diag = EnumBit.b0100
        };

        public static uint[] AiRank = { (uint)Rank.rank1, (uint)Rank.rank1, (uint)Rank.rank1, (uint)Rank.rank1, (uint)Rank.rank1, (uint)Rank.rank1, (uint)Rank.rank1, (uint)Rank.rank1, (uint)Rank.rank1,
                          (uint)Rank.rank2, (uint)Rank.rank2, (uint)Rank.rank2, (uint)Rank.rank2, (uint)Rank.rank2, (uint)Rank.rank2, (uint)Rank.rank2, (uint)Rank.rank2, (uint)Rank.rank2,
                          (uint)Rank.rank3, (uint)Rank.rank3, (uint)Rank.rank3, (uint)Rank.rank3, (uint)Rank.rank3, (uint)Rank.rank3, (uint)Rank.rank3, (uint)Rank.rank3, (uint)Rank.rank3,
                          (uint)Rank.rank4, (uint)Rank.rank4, (uint)Rank.rank4, (uint)Rank.rank4, (uint)Rank.rank4, (uint)Rank.rank4, (uint)Rank.rank4, (uint)Rank.rank4, (uint)Rank.rank4,
                          (uint)Rank.rank5, (uint)Rank.rank5, (uint)Rank.rank5, (uint)Rank.rank5, (uint)Rank.rank5, (uint)Rank.rank5, (uint)Rank.rank5, (uint)Rank.rank5, (uint)Rank.rank5,
                          (uint)Rank.rank6, (uint)Rank.rank6, (uint)Rank.rank6, (uint)Rank.rank6, (uint)Rank.rank6, (uint)Rank.rank6, (uint)Rank.rank6, (uint)Rank.rank6, (uint)Rank.rank6,
                          (uint)Rank.rank7, (uint)Rank.rank7, (uint)Rank.rank7, (uint)Rank.rank7, (uint)Rank.rank7, (uint)Rank.rank7, (uint)Rank.rank7, (uint)Rank.rank7, (uint)Rank.rank7,
                          (uint)Rank.rank8, (uint)Rank.rank8, (uint)Rank.rank8, (uint)Rank.rank8, (uint)Rank.rank8, (uint)Rank.rank8, (uint)Rank.rank8, (uint)Rank.rank8, (uint)Rank.rank8,
                          (uint)Rank.rank9, (uint)Rank.rank9, (uint)Rank.rank9, (uint)Rank.rank9, (uint)Rank.rank9, (uint)Rank.rank9, (uint)Rank.rank9, (uint)Rank.rank9, (uint)Rank.rank9,
                        };

        public static uint[] AiFile = { (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9,
                          (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9,
                          (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9,
                          (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9,
                          (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9,
                          (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9,
                          (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9,
                          (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9,
                          (uint)File.file1, (uint)File.file2, (uint)File.file3, (uint)File.file4, (uint)File.file5, (uint)File.file6, (uint)File.file7, (uint)File.file8, (uint)File.file9
                        };

        public static string[] StrUSI = { "9a", "8a", "7a", "6a", "5a", "4a", "3a", "2a", "1a",
                                          "9b", "8b", "7b", "6b", "5b", "4b", "3b", "2b", "1b",
                                          "9c", "8c", "7c", "6c", "5c", "4c", "3c", "2c", "1c",
                                          "9d", "8d", "7d", "6d", "5d", "4d", "3d", "2d", "1d",
                                          "9e", "8e", "7e", "6e", "5e", "4e", "3e", "2e", "1e",
                                          "9f", "8f", "7f", "6f", "5f", "4f", "3f", "2f", "1f",
                                          "9g", "8g", "7g", "6g", "5g", "4g", "3g", "2g", "1g",
                                          "9h", "8h", "7h", "6h", "5h", "4h", "3h", "2h", "1h",
                                          "9i", "8i", "7i", "6i", "5i", "4i", "3i", "2i", "1i"
                                        };

        public static string[] StrCSA = { "91", "81", "71", "61", "51", "41", "31", "21", "11",
                                          "92", "82", "72", "62", "52", "42", "32", "22", "12",
                                          "93", "83", "73", "63", "53", "43", "33", "23", "13",
                                          "94", "84", "74", "64", "54", "44", "34", "24", "14",
                                          "95", "85", "75", "65", "55", "45", "35", "25", "15",
                                          "96", "86", "76", "66", "56", "46", "36", "26", "16",
                                          "97", "87", "77", "67", "57", "47", "37", "27", "17",
                                          "98", "88", "78", "68", "58", "48", "38", "28", "18",
                                          "99", "89", "79", "69", "59", "49", "39", "29", "19",
                                          "00" 
                                        };

        public static string[] StrPiece = { "", "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU", "TO", "NY", "NK", "NG", "", "UM", "RY" };

        public static int[] StartPosition = { -(int)Piece.Type.Lance, -(int)Piece.Type.Knight, -(int)Piece.Type.Silver, -(int)Piece.Type.Gold, -(int)Piece.Type.King, -(int)Piece.Type.Gold, -(int)Piece.Type.Silver, -(int)Piece.Type.Knight, -(int)Piece.Type.Lance,
                                               (int)Piece.Type.Empty, -(int)Piece.Type.Rook,  (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, -(int)Piece.Type.Bishop, (int)Piece.Type.Empty,
                                              -(int)Piece.Type.Pawn, -(int)Piece.Type.Pawn, -(int)Piece.Type.Pawn, -(int)Piece.Type.Pawn, -(int)Piece.Type.Pawn, -(int)Piece.Type.Pawn, -(int)Piece.Type.Pawn, -(int)Piece.Type.Pawn, -(int)Piece.Type.Pawn,
                                               (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty,
                                               (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty,
                                               (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty,
                                               (int)Piece.Type.Pawn, (int)Piece.Type.Pawn, (int)Piece.Type.Pawn, (int)Piece.Type.Pawn, (int)Piece.Type.Pawn, (int)Piece.Type.Pawn, (int)Piece.Type.Pawn, (int)Piece.Type.Pawn, (int)Piece.Type.Pawn,
                                               (int)Piece.Type.Empty, (int)Piece.Type.Bishop, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Empty, (int)Piece.Type.Rook, (int)Piece.Type.Empty,
                                               (int)Piece.Type.Lance, (int)Piece.Type.Knight, (int)Piece.Type.Silver, (int)Piece.Type.Gold, (int)Piece.Type.King, (int)Piece.Type.Gold, (int)Piece.Type.Silver, (int)Piece.Type.Knight, (int)Piece.Type.Lance
                                             };
        public struct CheckTable { public BitBoard gold, silver, knight, lance; }
    }
}