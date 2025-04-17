using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BarcodeScanner.Mobile
{
    public class DetectedObject
    {
        public float Confidence { get; }
        public RectF BoundingBox { get; }
        public int ClassIndex { get; }
        public string Label { get; }

        public DetectedObject(float confidence, RectF boundingBox, int classIndex, string label)
        {
            Confidence = confidence;
            BoundingBox = boundingBox;
            ClassIndex = classIndex;
            Label = label;
        }
        public DetectedObject()
        {
                
        }
    }
}
