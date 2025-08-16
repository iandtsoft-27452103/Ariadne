using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class Hash
    {
        /*
         *ハッシュ用クラス
         * Bonanzaのrand.cとhash.cを移植
         * PRNG based on Mersenne Twister ( M.Matsumoto and T.Nishimura, 1998 ).
        */

        private const uint RandM = 397;
        private const int RandN = 624;
        private const uint MaskU = 0x80000000U;
        private const uint MaskL = 0x7fffffffU;
        private const uint Mask32 = 0xffffffffU;

        public static ulong[,,] PieceRand = new ulong[Color.ColorNum, Piece.PieceNum, ShogiCommon.NSquare];

        public struct RandWorkT
        {
            public int count;
            public uint[] cnst;
            public uint[] vec;
        }

        public static RandWorkT rand_work;

        public static void IniRand(uint u)
        {
            rand_work.cnst = new uint[2];
            rand_work.vec = new uint[RandN];

            rand_work.count = RandN;
            rand_work.cnst[0] = 0;
            rand_work.cnst[1] = 0x9908b0dfU;

            for (int i = 1; i < RandN; i++)
            {
                u = (uint)(i + 1812433253U * (u ^ (u >> 30)));
                u &= Mask32;
                rand_work.vec[i] = u;
            }
        }

        public static uint Rand32()
        {
            uint u, u0, u1, u2;
            int i;

            if (rand_work.count == RandN)
            {
                rand_work.count = 0;

                for (i = 0; i < RandN - RandM; i++)
                {
                    u = rand_work.vec[i] & MaskU;
                    u |= rand_work.vec[i + 1] & MaskL;

                    u0 = rand_work.vec[i + RandM];
                    u1 = u >> 1;
                    u2 = rand_work.cnst[u & 1];

                    rand_work.vec[i] = u0 ^ u1 ^ u2;
                }

                for (; i < RandN - 1; i++)
                {
                    u = rand_work.vec[i] & MaskU;
                    u |= rand_work.vec[i + 1] & MaskL;

                    u0 = rand_work.vec[i + RandM - RandN];
                    u1 = u >> 1;
                    u2 = rand_work.cnst[u & 1];

                    rand_work.vec[i] = u0 ^ u1 ^ u2;
                }

                u = rand_work.vec[RandN - 1] & MaskU;
                u |= rand_work.vec[0] & MaskL;

                u0 = rand_work.vec[RandM - 1];
                u1 = u >> 1;
                u2 = rand_work.cnst[u & 1];

                rand_work.vec[RandN - 1] = u0 ^ u1 ^ u2;
            }

            u = rand_work.vec[rand_work.count++];
            u ^= (u >> 11);
            u ^= (u << 7) & 0x9d2c5680U;
            u ^= (u << 15) & 0xefc60000U;
            u ^= (u >> 18);

            return u;
        }

        public static ulong Rand64()
        {
            ulong h = Rand32();
            ulong l = Rand32();

            return l | (h << 32);
        }

        public static void IniRandomTable()
        {
            for (int c = 0; c < Color.ColorNum; c++)
            {
                for (int pc = 0; pc < Piece.PieceNum; pc++)
                {
                    for (int sq = 0; sq < ShogiCommon.NSquare; sq++)
                    {
                        PieceRand[c, pc, sq] = Rand64();
                    }
                }
            }
        }

        public static ulong HashFunc(Board.BoardTree BTree)
        {
            BitBoard bb = new BitBoard();
            int sq;
            ulong key = 0;

            for (int c = 0; c < Color.ColorNum; c++)
            {
                for (int pc = 1; pc < Piece.PieceNum; pc++)
                {
                    bb = BitBoard.Copy(BTree.BB_Piece[c, pc]);
                    while(BitOperation.BBTest(bb) != 0)
                    {
                        sq = BitOperation.LastOne012(bb.p[0], bb.p[1], bb.p[2]);
                        AttackBitBoard.Xor(sq, ref bb);
                        key ^= PieceRand[c, pc, sq];
                    }
                }
            }

            return key;
        }
    }
}
