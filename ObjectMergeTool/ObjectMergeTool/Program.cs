using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace ObjectMergeTool
{
    class Program
    {
        static void Main(string[] args)
        {
            SegmentationResult reminder = new SegmentationResult(); 

            SegmentationResult objectToMerge = new SegmentationResult();
            objectToMerge.Merge(out SegmentationResult reminder);
        }
    }
}
