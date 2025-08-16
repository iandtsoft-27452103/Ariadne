using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Asklepios
{
    internal class Direction
    {
        public enum DirectionType
        {
            Misc = 0, File = 2, Rank = 3, Diag1 = 4, Diag2 = 5
        }

        public enum DirectionFlag
        {
            Cross = 2, Diag = 4
        }
    }
}
