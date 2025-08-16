using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class TT
    {
        //public Dictionary<ulong, int> PolicyMoveCount = new Dictionary<ulong, int>();
        public Dictionary<ulong, List<Move>> PolicyMoves = new Dictionary<ulong, List<Move>>();
        public Dictionary<ulong, float> PolicyDict = new Dictionary<ulong, float>();
        public Dictionary<ulong, float> ValueDict = new Dictionary<ulong, float>();

        public TT()
        {
            //PolicyMoveCount.Clear();
            PolicyDict.Clear();
            ValueDict.Clear();
        }
    }
}
