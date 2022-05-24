using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.IO;
using System.Drawing; 
using AForge.Imaging.Filters;
using AForge;
using AForge.Imaging;

namespace YOLODataSetConverter
{
    public class ToYOLOConverter
    {
        string _saveDirectory;
        string _readDirectory;
        Labels labels;

        public ToYOLOConverter(string saveDirectory, string readDirectory)
        {
            _saveDirectory = saveDirectory;
            _readDirectory = readDirectory;
            labels = new Labels();
            Convert(); 

        }

        public void Convert()
        {
            var labelsDirectories = Directory.GetFiles(_readDirectory); 

            foreach(var directory in labelsDirectories)
            {
                //check if corresponding image exists
                string fileName = Path.GetFileName(directory);
                if (!IsCorrespondingImageExists(fileName)) continue;

                //read image from file               
                var image = System.Drawing.Image.FromFile(directory);
                Bitmap label = new Bitmap(image);

                //exclude all blobs from labels image
                var blobs = ExcludeAllObjects(label);

                //prepare records to save 
                var recordsToSave = PrepareListOfRecordsToSave(blobs, label.Size);

                //image disposing
                label.Dispose();

                //save blobs to txt file 
                Save(recordsToSave, Path.Combine(_saveDirectory, fileName.Replace("label", "image").Replace(".png", ".txt")));
            }
        }

        private bool IsCorrespondingImageExists(string fileName)
        {
            bool exists = false;

            var imagePath = Path.Combine(_saveDirectory, fileName.Replace("label", "image"));

            exists = File.Exists(imagePath);
                

            return exists; 
        }

        private List<Tuple<int, Blob>> ExcludeAllObjects(Bitmap label)
        {
            List<Tuple<int, Blob>> result = new List<Tuple<int, Blob>>();

            ConnectedComponentsLabeling connectedComponentsLabeling;
            connectedComponentsLabeling = new ConnectedComponentsLabeling();
            connectedComponentsLabeling.FilterBlobs = false;

            foreach(var labelObject in labels.LabelList)
            {
                if (labelObject.Name == "tlo") continue;
                var blobs = GetBlobsForSpecificLabel(labelObject, label, connectedComponentsLabeling);
                if(blobs.Count> 0)
                {
                    result.AddRange(blobs);
                }
                
            }

            

            return result;
        }
        
        private List<Tuple<int, Blob>> GetBlobsForSpecificLabel(Label labelObject, Bitmap labelImage, ConnectedComponentsLabeling connectedComponentsLabeling)
        {
            List<Tuple<int, Blob>> result = new List<Tuple<int, Blob>>();
            var colorFiltering = new ColorFiltering(new IntRange(labelObject.R, labelObject.R), new IntRange(labelObject.G, labelObject.G), new IntRange(labelObject.B, labelObject.B)); 
            var separateColorBitmap = colorFiltering.Apply(labelImage);

            connectedComponentsLabeling.Apply(separateColorBitmap);
            var blobs = connectedComponentsLabeling.BlobCounter.GetObjects(labelImage, true);
            if (blobs.Count() > 0)
            {
                foreach(var blob in blobs)
                {
                    result.Add(new Tuple<int, Blob>(labelObject.Id, blob));
                }

            }
            return result; 

        }

        private List<RecordToSave> PrepareListOfRecordsToSave(List<Tuple<int,Blob>> blobs, Size size)
        {
            int id;
            float normalizedX;
            float normalizedY;
            float normalizedWidth;
            float normalizedHeight;

            List<RecordToSave> result = new List<RecordToSave>(); 

            
            foreach (var blob in blobs)
            {
                id = blob.Item1;
                //normalize rectangle values 
                normalizedX = (float)blob.Item2.CenterOfGravity.X / (float)size.Width;
                normalizedY = (float)blob.Item2.CenterOfGravity.Y / (float)size.Height;
                normalizedWidth = (float)blob.Item2.Rectangle.Width / (float)size.Width;
                normalizedHeight = (float)blob.Item2.Rectangle.Height / (float)size.Height;

                result.Add(new RecordToSave()
                {
                    ObjectNo = id,
                    X = normalizedX,
                    Y = normalizedY,
                    Width = normalizedWidth,
                    Height = normalizedHeight
                });
            }

            return result; 

        }

        private void Save(List<RecordToSave> records, string directory)
        {
            using (StreamWriter sw = new StreamWriter(directory))
            {
                foreach(var record in records)
                {
                    sw.WriteLine(String.Format("{0} {1:0.000000} {2:0.000000} {3:0.000000} {4:0.000000}",
                        record.ObjectNo,
                        record.X,
                        record.Y,
                        record.Width,
                        record.Height));
                }
                
            }
        }

    }
}

