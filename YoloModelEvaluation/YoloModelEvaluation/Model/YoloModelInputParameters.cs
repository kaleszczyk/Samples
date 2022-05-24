using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloModelEvaluation.Model
{
    public class YoloModelInputParameters
    {
        private string modelDataFilePath;
        private string modelConfigFilePath;
        private string modelWeightsFilePath;

        public string ModelDataFilePath
        {
            get
            {
                return modelDataFilePath;
            }
            set
            {
                if (value.IslDataFileCorrect())
                {
                    modelDataFilePath = value; 
                }
                else
                {
                    modelDataFilePath = string.Empty;
                }
            }

        }
        public string ModelConfigFilePath
        {
            get
            {
                return modelConfigFilePath;
            }
            set
            {
                if(value.IsConfigFileCorrect())
                {
                    modelConfigFilePath = value;
                }
                else
                {
                    md
                }
            }
        }

        public string ModelWeightsFilePath
        {
            get
            {
                return modelWeightsFilePath; 
                   
            }
            set
            {
                modelWeightsFilePath = value;
            }
        }


    }
}
