using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class BitBoard
    {
        // メモリ的にはuint3枚にすべきだが、ビット演算の都合で
        // ulong3枚にしている
        public ulong[] p;// privateにしていない
        public BitBoard()
        {
            p = new ulong[3];
        }

        public static BitBoard Copy(BitBoard bb)
        {
            BitBoard bb_copy = new BitBoard();
            bb_copy.p[0] = bb.p[0];
            bb_copy.p[1] = bb.p[1];
            bb_copy.p[2] = bb.p[2];
            return bb_copy;
        }

        public static BitBoard SetFull()
        {
            BitBoard bb_copy = new BitBoard();
            bb_copy.p[0] = bb_copy.p[1] = bb_copy.p[2] = 0x7FFFFFF;
            return bb_copy;
        }
    }
}
