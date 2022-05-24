using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectMergeTool
{
    public static class SegmentationResultExtension
    {
        public static MergeTool mergeTool = new MergeTool(); 
        public static SegmentationResult Merge(this SegmentationResult objectToMerge, out SegmentationResult reminder)
        {
            reminder = new SegmentationResult();


        
            
        }

     

    }

    public class MergeTool
    {
        private SegmentationResult _reminder = null; 
        
        public MergeTool()
        {

        }

        public SegmentationResult Merge(SegmentationResult objectToMerge, int objectID)
        {
            if(_reminder == null)
            {
                if(IsOnTopBorder(objectToMerge))
                _reminder = objectToMerge; 
            }
            else
            {
                if(Math.Abs(objectToMerge.FrameID - _reminder.FrameID) ==1 )
                {
                    //sprawdzenie przystawania do górnej lub dolnej krawędzi w zaleznosci od kierunku 
                }
            }

        }


        private bool IsOnTopBorder(SegmentationResult segmentationResult, int objectID)
        {
            var filtered = segmentationResult.BoundingBoxes.Where(x => x.ObjectID == objectID).ToList(); 

            foreach(var label in filtered)
            {
                if(label)
            }
        }

        //check bottom border
    }
}
