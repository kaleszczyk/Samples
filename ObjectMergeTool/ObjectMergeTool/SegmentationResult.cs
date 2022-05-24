using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace ObjectMergeTool
{
    public class SegmentationResult
    {
        public int FrameID { get; set; } //? czy potrzebne chyba nie
        public List<BoundingBox> BoundingBoxes {get; set;}
        public Bitmap OriginalImage { get; set; }
    }
}
