using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YOLODataSetConverter
{
    public class Labels
    { 
        public List<Label> LabelList { get; set; }

        public Labels(string mode = "segmentation")
        {
            if (mode != "segmentation") throw new NotImplementedException("Labels list was implemented only for segmentation objects.");
            Initialize(); 
            
        }

        public void Initialize()
        {
            LabelList = new List<Label>();

            LabelList.Add(new Label(0, "tlo", 255, 255, 255));
            LabelList.Add(new Label(1, "object01", 87, 27, 126));
            LabelList.Add(new Label(2, "object02", 202, 34, 107));
            LabelList.Add(new Label(3, "object03", 244, 51, 255));
            LabelList.Add(new Label(4, "object04", 62, 160, 85));
            LabelList.Add(new Label(5, "object05", 135, 247, 23));
            LabelList.Add(new Label(6, "object06", 0, 0, 255));
            LabelList.Add(new Label(7, "object07", 255, 0, 0));
            LabelList.Add(new Label(8, "object08", 255, 128, 0));
            LabelList.Add(new Label(9, "object09", 198, 226, 255));
            LabelList.Add(new Label(10, "object10", 156, 102, 31));
            LabelList.Add(new Label(11, "object11", 255, 0, 255));
            LabelList.Add(new Label(12, "object12", 0, 255, 255));
            LabelList.Add(new Label(13, "object13", 255, 255, 0));
            LabelList.Add(new Label(14, "object14", 191, 255, 0));
            LabelList.Add(new Label(15, "object15", 0, 255, 191));
            LabelList.Add(new Label(16, "object16", 230, 230, 230));
            LabelList.Add(new Label(17, "object17", 255, 204, 255));
            LabelList.Add(new Label(18, "object18", 77, 0, 77));
            LabelList.Add(new Label(19, "object19", 153, 102, 51));
            LabelList.Add(new Label(20, "object20", 115, 128, 77));
            LabelList.Add(new Label(21, "object21", 0, 0, 102));
            LabelList.Add(new Label(22, "object22", 255, 255, 230));
            LabelList.Add(new Label(23, "object23", 179, 77, 102));
            LabelList.Add(new Label(24, "object24", 179, 102, 255));
            LabelList.Add(new Label(25, "object25", 0, 128, 255));
            LabelList.Add(new Label(26, "object26", 53, 0, 200));
            LabelList.Add(new Label(27, "szyna przejazdu", 170, 40, 0));
            LabelList.Add(new Label(28, "object28", 45, 179, 0));
            LabelList.Add(new Label(29, "object29", 4, 90, 190));
            LabelList.Add(new Label(30, "object30", 51, 250, 0));
            LabelList.Add(new Label(31, "object31", 251, 20, 100));
            LabelList.Add(new Label(32, "object32", 4, 50, 180));
            LabelList.Add(new Label(33, "object33", 11, 210, 210));
            LabelList.Add(new Label(34, "object34", 44, 200, 250));
            LabelList.Add(new Label(35, "object35", 232, 172, 227));

        } 

    }

    public class Label
    {
        public Label(int id, string name, byte R, byte G, byte B)
        {
            this.Id = id;
            this.Name = name;
            this.R = R;
            this.G = G;
            this.B = B; 
           
        }
        public int Id { get; set; }
        public string Name { get; set; }
        public byte R { get; set; }
        public byte G { get; set; }
        public byte B { get; set; }
    }
}
