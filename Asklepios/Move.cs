using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class Move
    {
        private uint _move;
        public Move() 
        {
            _move = 0U;
        }
        public uint Value
        {
            set { _move = value; }
            get { return _move; }
        }

        public int To
        {
            get
            {
                return (int)_move & 0x007f;
            }
        }

        public int From
        {
            get
            {
                return (int)(_move >> 7) & 0x007f;
            }
        }

        public int FlagPromo
        {
            get
            {
                return (int)(_move >> 14) & 1;
            }
        }

        public Piece.Type PieceType
        {
            get 
            {
                return (Piece.Type)((_move >> 15) & 0x000f);
            }
        }

        public Piece.Type CapPiece
        {
            get
            {
                return (Piece.Type)((_move >> 19) & 0x000f);
            }
        }

        /*
        xxxxxxxx xxxxxxxx x1111111 To位置
        xxxxxxxx xx111111 1xxxxxxx From位置
        xxxxxxxx x1xxxxxx xxxxxxxx 成る手かどうか
        xxxxx111 1xxxxxxx xxxxxxxx 動かした駒の種類
        x1111xxx xxxxxxxx xxxxxxxx 捕獲した駒
        */

        public void Push(int from, int to, Piece.Type pc, Piece.Type cap_pc, int flag_promo)
        {
            _move = ((uint)cap_pc << 19) | ((uint)pc << 15) | ((uint)flag_promo << 14) | ((uint)from << 7) | (uint)to;
        }
    }

    internal class CheckMove:Move
    {
        // check_type 0: check_normal
        //            1: check_clear
        //            2: check_set
        public int check_type;
    }

    internal class NextMove:Move
    {
        public int next_phase;
        //public int remaining;
    }
}
