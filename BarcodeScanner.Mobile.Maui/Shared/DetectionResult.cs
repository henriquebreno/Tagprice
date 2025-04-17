using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BarcodeScanner.Mobile
{
    public class DetectionResults : List<DetectedObject>
    {
        public object Image { get; set; }
        public List<ObjectDetection> Detections { get; set; }
        public object Info { get; set; }
    }

    public class ObjectDetection
    {
        public RectF BoundingBox { get; set; }
        public Category Category { get; set; }
    }
    public class Category
    {
        public string Label { get; set; }
        public float Confidence { get; set; }
    }
}
