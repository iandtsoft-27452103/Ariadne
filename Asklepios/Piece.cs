using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class Piece
    {
        public const int PieceNum = 16;
        public const int PromoteNum = 8;

        public enum Type
        {
            Empty, Pawn, Lance, Knight, Silver, Gold, Bishop, Rook, King, Pro_Pawn, Pro_Lance, Pro_Knight, Pro_Silver, None, Horse, Dragon
        }
    }
}
