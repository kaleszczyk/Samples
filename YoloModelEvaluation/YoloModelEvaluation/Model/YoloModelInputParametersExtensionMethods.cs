using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloModelEvaluation.Model
{
    internal static class YoloModelInputParametersExtensionMethods
    {
        internal static bool IslDataFileCorrect(this string modelDataFilePath)
        {
            //czy istnieje?
            //czy ma dobre rozszerzenie?
            //czy ścieżki ze środka istnieją i posiadają jakieś dane 
            return true;
        }

        internal static bool IsConfigFileCorrect(this string modelDataConfigPath)
        {
            //czy istnieje?
            //czy ma dobre rozszerzenie?
            return true;
        }
        internal static bool IsWeightsFileCorrect(this string modelDataFilePath)
        {
            //czy istnieje?
            //czy ma dobre rozszerzenie?
            return true;
        }
    }
}
