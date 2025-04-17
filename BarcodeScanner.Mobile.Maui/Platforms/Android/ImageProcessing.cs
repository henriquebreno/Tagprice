using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace BarcodeScanner.Mobile
{

    public class ImageProcessing
    {
       
        static ImageProcessing()
        {
            if (DeviceInfo.Platform == DevicePlatform.Android)
            {
                NativeLibrary.Load("image_processing");
            }
        }

        [DllImport("image_processing", EntryPoint = "argb2yolo")]
        public static extern void Argb2Yolo(int[] srcArray, IntPtr destBuffer, int width, int height);

    }

}
