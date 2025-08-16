using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class BitOperation
    {
        private static int[] BitIndex = {  0, 47,  1, 56, 48, 27,  2, 60,
                                          57, 49, 41, 37, 28, 16,  3, 61,
                                          54, 58, 35, 52, 50, 42, 21, 44,
                                          38, 32, 29, 23, 17, 11,  4, 62,
                                          46, 55, 26, 59, 40, 36, 15, 53,
                                          34, 51, 20, 43, 31, 22, 10, 45,
                                          25, 39, 14, 33, 19, 30,  9, 24,
                                          13, 18,  8, 12,  7,  6,  5, 63
                                        };

        // C++のPextと同等の関数を自作
        // Chess Programing Wikiを参照
        // Bitboardはulong3枚なので、
        // 1枚ずつ渡す
        public static ulong PextU64(ulong Val, ulong Mask)
        {
            ulong BB;
            ulong Res = 0UL;

            for (BB = 1; Mask != 0UL; BB += BB)
            {
                if ((Val & Mask & (~Mask + 1)) != 0UL)
                {
                    Res |= BB;
                }

                Mask &= (Mask - 1);
            }

            return Res;
        }

        // C++のBitScanForwardと同等の関数を自作
        // Chess Programing Wikiを参照
        // Bitboardはulong3枚なので、
        // 1枚ずつ渡す
        public static int BitScanForward(ulong bb_p)
        {
            const ulong Magic = 0x03f79d71b4cb0a89;
            return BitIndex[(bb_p ^ (bb_p - 1)) * Magic >> 58];
        }

        // こちらは計算が重いので、使わない方が
        // ベターかもしれない
        // Bitboardはulong3枚なので、
        // 1枚ずつ渡す
        public static int BitScanReverse(ulong bb_p)
        {
            const ulong Magic = 0x03f79d71b4cb0a89;
            bb_p |= bb_p >> 1;
            bb_p |= bb_p >> 2;
            bb_p |= bb_p >> 4;
            bb_p |= bb_p >> 8;
            bb_p |= bb_p >> 16;
            bb_p |= bb_p >> 32;
            return BitIndex[(bb_p * Magic) >> 58];
        }

        // 立っているビットの数を数える
        // Bitboardはulong3枚なので、
        // 1枚ずつ渡す
        public static int PopuCount(ulong bb_p)
        {
            int counter = 0;

            while (bb_p != 0)
            {
                counter++;
                bb_p &= bb_p - 1UL;
            }

            return counter;
        }

        public static int FirstOne012(ulong bb_p0, ulong bb_p1, ulong bb_p2)
        {
            int index;

            if (bb_p0 != 0)
            {
                index = BitScanReverse(bb_p0);
                return 26 - index;
            }

            if (bb_p1 != 0)
            {
                index = BitScanReverse(bb_p1);
                return 53 - index;
            }

            index = BitScanReverse(bb_p2);
            return 80 - index;
        }

        public static int FirstOne01(ulong bb_p0, ulong bb_p1)
        {
            int index;

            if (bb_p0 != 0) 
            { 
                index = BitScanReverse(bb_p0);
                return 26 - index;
            }
            index = BitScanReverse(bb_p1);
            return 53 - index;
        }

        public static int FirstOne12(ulong bb_p1, ulong bb_p2)
        {
            int index;

            if (bb_p1 != 0)
            {
                index = BitScanReverse(bb_p1);
                return 53 - index;
            }
            index = BitScanReverse(bb_p2);
            return 80 - index;
        }

        public static int FirstOne0(ulong bb_p0)
        {
            int index = BitScanReverse(bb_p0);

            return 26 - index;
        }

        public static int FirstOne1(ulong bb_p1)
        {
            int index = BitScanReverse(bb_p1);

            return 53 - index;
        }

        public static int FirstOne2(ulong bb_p2)
        {
            int index = BitScanReverse(bb_p2);

            return 80 - index;
        }

        public static int LastOne012(ulong bb_p0, ulong bb_p1, ulong bb_p2)
        {
            int index;

            if (bb_p2 != 0)
            {
                index = BitScanForward(bb_p2);
                return 80 - index;
            }
            if (bb_p1 != 0)
            {
                index = BitScanForward(bb_p1);
                return 53 - index;
            }
            index = BitScanForward(bb_p0);
            return 26 - index;
        }

        /*public static int LastOne2Pcs(ulong a, int a_sq, ulong b, int b_sq)
        {
            int index;

            if (b != 0)
            {
                index = BitScanForward(b);
                return b_sq - index;
            }
            index = BitScanForward(a);
            return a_sq - index;
        }*/
        public static int LastOne01(ulong bb_p0, ulong bb_p1)
        {
            int index;

            if (bb_p1 != 0)
            {
                index = BitScanForward(bb_p1);
                return 53 - index;
            }
            index = BitScanForward(bb_p0);
            return 26 - index;
        }

        public static int LastOne12(ulong bb_p1, ulong bb_p2)
        {
            int index;

            if (bb_p2 != 0)
            {
                index = BitScanForward(bb_p2);
                return 80 - index;
            }
            index = BitScanForward(bb_p1);
            return 53 - index;
        }

        public static int LastOne0(ulong bb_p0)
        {
            int index = BitScanForward(bb_p0);

            return 26 - index;
        }

        public static int LastOne1(ulong bb_p1)
        {
            int index = BitScanForward(bb_p1);

            return 53 - index;
        }

        public static int LastOne2(ulong bb_p2)
        {
            int index = BitScanForward(bb_p2);

            return 80 - index;
        }

        public static int LastOneN(ulong bb_p0, int num)
        {
            int index = BitScanForward(bb_p0);

            return num - index;
        }

        public static int LastOneAB(ulong bb_p_a, int num_a, ulong bb_p_b, int num_b)
        {
            int index;

            if (bb_p_b != 0)
            {
                index = BitScanForward(bb_p_b);
                return num_b - index;
            }
            index = BitScanForward(bb_p_a);
            return num_a - index;
        }

        public static void BBIni(ref BitBoard BB)
        {
            BB.p[0] = BB.p[1] = BB.p[2] = 0UL;
        }

        public static void BBNot(ref BitBoard BB0, BitBoard BB1)
        {
            BB0.p[0] = ~BB1.p[0];
            BB0.p[1] = ~BB1.p[1];
            BB0.p[2] = ~BB1.p[2];
        }

        public static void BBAnd(ref BitBoard BB0, BitBoard BB1, BitBoard BB2)
        {
            BB0.p[0] = BB1.p[0] & BB2.p[0];
            BB0.p[1] = BB1.p[1] & BB2.p[1];
            BB0.p[2] = BB1.p[2] & BB2.p[2];
        }

        public static void BBOr(ref BitBoard BB0, BitBoard BB1, BitBoard BB2)
        {
            BB0.p[0] = BB1.p[0] | BB2.p[0];
            BB0.p[1] = BB1.p[1] | BB2.p[1];
            BB0.p[2] = BB1.p[2] | BB2.p[2];
        }

        public static void BBNotAnd(ref BitBoard BB0, BitBoard BB1, BitBoard BB2)
        {
            BB0.p[0] = BB1.p[0] & ~BB2.p[0];
            BB0.p[1] = BB1.p[1] & ~BB2.p[1];
            BB0.p[2] = BB1.p[2] & ~BB2.p[2];
        }

        public static void BBSetClear(ref BitBoard BB0, BitBoard BBSetClear)
        {
            BB0.p[0] ^= BBSetClear.p[0];
            BB0.p[1] ^= BBSetClear.p[1];
            BB0.p[2] ^= BBSetClear.p[2];
        }

        public static void BBXor(ref BitBoard BB0, BitBoard BB1, BitBoard BB2)
        {
            BB0.p[0] = BB1.p[0] ^ BB2.p[0];
            BB0.p[1] = BB1.p[1] ^ BB2.p[1];
            BB0.p[2] = BB1.p[2] ^ BB2.p[2];
        }

        public static void BBAndOr(ref BitBoard BB0, BitBoard BB1, BitBoard BB2)
        {
            BB0.p[0] |= BB1.p[0] & BB2.p[0];
            BB0.p[1] |= BB1.p[1] & BB2.p[1];
            BB0.p[2] |= BB1.p[2] & BB2.p[2];
        }

        public static ulong BBContract(BitBoard BB0, BitBoard BB1)
        {
            return (BB0.p[0] & BB1.p[0] | BB0.p[1] & BB1.p[1] | BB0.p[2] & BB1.p[2]);
        }

        public static ulong BBTest(BitBoard BB)
        {
            return (BB.p[0] | BB.p[1] | BB.p[2]);
        }
    }
}
