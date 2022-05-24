using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloModelEvaluation.Utils;

namespace YoloModelEvaluation.Model
{
    public class InputParameters
    {

        //parametry wejściowe pochodące od uzytkownika 
        private string validationDataDirectory;

        private YoloModelInputParameters yoloModelInputParameters; 

        public string ValidationDataDirectory
        {
            get
            {
                return validationDataDirectory;
            }
            set
            {
                 validationDataDirectory = value;
            }
        }
    }
}
