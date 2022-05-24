using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YOLODataSetConverter
{
    class Program
    {
        static void Main(string[] args)
        {
            string saveDirectory = @"D:\YOLO\darknet2\darknet\data\graw\segmentacja\images\train\"; // ścieżka do wszystkich obrazów w data secie (one muszą być scalone, nie podzielone na train i val)
            string readDirectory = @"D:\YOLO\darknet2\darknet\data\graw\segmentacja\labels\train"; // ścieżka do labeli 

            ToYOLOConverter converter = new ToYOLOConverter(saveDirectory, readDirectory);
            converter.Convert();
        }
    }
}
