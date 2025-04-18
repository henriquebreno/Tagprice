﻿namespace BarcodeScanner.Mobile
{
    public class OnDetectedEventArg : EventArgs
    {
        public List<BarcodeResult> BarcodeResults { get; set; }
        public DetectionResults DetectionResults { get; set; }

        public OCRResult OCRResult { get; set; }
        public byte[] ImageData { get; set; }
        public OnDetectedEventArg()
        {
            ImageData = new byte[0];
            BarcodeResults = new List<BarcodeResult>();
            OCRResult = new OCRResult();
            DetectionResults = new DetectionResults();
        }
    }
}
