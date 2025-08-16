using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class MateNode
    {
        public int color;
        public int ParentIndex;
        public int ThisIndex;
        public bool IsLeaf;
        public List<int> ChildIndexes;
        public Move move;
        public bool IsMate;

        public MateNode()
        {
            color = 0;
            ParentIndex = int.MaxValue;
            ThisIndex = int.MaxValue;
            IsLeaf = true;
            ChildIndexes = new List<int>();
            ChildIndexes.Clear();
            IsMate = false;
        }
    }
}
