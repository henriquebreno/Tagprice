using Android.Content;
using Java.IO;
using Java.Nio.Channels;
using Java.Nio;
using System;
using System.Collections.Generic;

using Android.Graphics;

using Color = Android.Graphics.Color;
using TensorFlow.Lite.Support.Image;
using TensorFlow.Lite.Support.Common.Ops;
using TensorFlow.Lite.Support.TensorBuffer;
using TensorFlow.Lite.Support.Common;
using static KotlinX.Serialization.Descriptors.PrimitiveKind;
using TensorFlow.Lite.Support.Image.Ops;
using System.Drawing;
using TensorFlow.Lite.Support.Label;
using Java.Lang;
using Android.Runtime;
using Math = Java.Lang.Math;
using System.Text.Json.Serialization;
using System.Diagnostics;
using TensorFlow.Lite.Task.Vision.Detector;
using Xamarin.TensorFlow.Lite;


namespace BarcodeScanner.Mobile
{
    public class Stats
    {
        public float ImageSetupTime;
        public float InferenceTime;
        public float PostProcessTime;
    }
    public class TensorflowDetector
    {
        private string modelPath;
        private string labelPath;
        private Xamarin.TensorFlow.Lite.Interpreter interpreter;
        private IList<string> labels = new List<string>();
        private static int NUM_BYTES_PER_CHANNEL = 4;
        private int tensorWidth;
        private int tensorHeight;
        private int numChannel;
        private int numElements;
        private ImageProcessor PreImageProcessor;
        private ImageProcessor PosImageProcessor;
        private TensorImage TensorImage;
        private NormalizeOp preProcessNormalizeOp;
        private NormalizeOp posProcessNormalizeOp;

        private int[] intValues;
        private float[][] output;
        private int outputShape2;
        private int outputShape3;
        private readonly Context Context;
        private ImageProcessing ImageProcessing;
        private byte[] bytes;
        private ByteBuffer imgData;
        private ByteBuffer outData;
        private Java.Lang.Object[] inputArray;
        private JavaDictionary<Integer, Java.Lang.Object> outputMap;
        public Stats stats = new Stats();
        private double confidenceThreshold = 0.9f;
        private double iouThreshold = 0.45f;
        private int numItemsThreshold = 30;
        private int numClasses;




        public TensorflowDetector(Context context)
        {
            this.Context = context;
            posProcessNormalizeOp = new NormalizeOp(ModelConfig.IMAGE_MEAN, ModelConfig.IMAGE_STD);
            preProcessNormalizeOp = new NormalizeOp(ModelConfig.PROBABILITY_MEAN, ModelConfig.PROBABILITY_STD);
            ImageProcessing = new ImageProcessing();


        }

        public void Setup()
        {
            var modelBuffer = LoadModel(this.Context, ModelConfig.MODEL_FILENAME);
            labels = FileUtil.LoadLabels(this.Context, ModelConfig.LABEL_FILENAME);
            numClasses = labels.Count();
            var options = new Xamarin.TensorFlow.Lite.Interpreter.Options();
            options.SetNumThreads(4);
            options.SetUseNNAPI(true);

            interpreter = new Xamarin.TensorFlow.Lite.Interpreter(modelBuffer, options);
            interpreter.AllocateTensors();
            var shape = interpreter.GetOutputTensor(0).Shape();
            outputShape2 = shape[1];
            outputShape3 = shape[2];
            output = new float[outputShape2][];
            for (int i = 0; i < outputShape2; i++)
                output[i] = new float[outputShape3];

            //int[] inputShape = interpreter.GetInputTensor(0).Shape();
            //int[] outputShape = interpreter.GetOutputTensor(0).Shape();

            //tensorWidth = inputShape[1];
            //tensorHeight = inputShape[2];
            //numChannel = outputShape[1];
            //numElements = outputShape[2];

            //TensorImage = new TensorImage(Xamarin.TensorFlow.Lite.DataType.Float32);
            //LoadLabels(labelPath);
        }

        private TensorImage LoadImage(Bitmap bitmap, int sensorOrientation)
        {
            //TensorImage.Load(bitmap);
            // Creates processor for the TensorImage.
            //int cropSize = Math.Min(bitmap.Width, bitmap.Height);
            var numRotation = -sensorOrientation / 90;

            var builder = new ImageProcessor.Builder();
            //builder.Add(new ResizeWithCropOrPadOp(500, 500));
            var image = builder
                .Add(new ResizeOp(ModelConfig.INPUT_SIZE, ModelConfig.INPUT_SIZE, ResizeOp.ResizeMethod.Bilinear))
                .Add(new Rot90Op(numRotation))
                //.Add(preProcessNormalizeOp)
                .Add(new CastOp(DataType.Float32))
                .Build();
            return image.Process(TensorImage.FromBitmap(bitmap));
        }

        public static Bitmap CropCenterSquare(Bitmap bitmap, int size)
        {
            int x = (bitmap.Width - size) / 2;
            int y = (bitmap.Height - size) / 2;
            return Bitmap.CreateBitmap(bitmap, x, y, size, size);
        }

        private MappedByteBuffer LoadModel(Context context, string modelPath)
        {
            using var fileDescriptor = context.Assets.OpenFd(modelPath);
            using var inputStream = new FileInputStream(fileDescriptor.FileDescriptor);
            return inputStream.Channel.Map(FileChannel.MapMode.ReadOnly, fileDescriptor.StartOffset, fileDescriptor.DeclaredLength);
        }

        private void LoadLabels(string labelPath)
        {
            foreach (var line in System.IO.File.ReadLines(labelPath))
            {
                labels.Add(line);
            }
        }

        public void Clear()
        {
            interpreter?.Dispose();
            interpreter = null;
        }

        public DetectionResults Detect(Bitmap bitmap, int sensorOrientation )
        {
            if (interpreter == null )
                return default;

        
            TensorImage = LoadImage(bitmap, sensorOrientation);
            //var PreProcessImage = PreProcess(TensorImage.Bitmap);
            SetInputOptim(TensorImage.Bitmap);
            var detectedObjects = RunInference(); 
            System.Console.WriteLine($"BestBox Result:{System.Text.Json.JsonSerializer.Serialize(detectedObjects)}");

            int imgW, imgH;

            if (sensorOrientation == 90 || sensorOrientation == 270)
            {
                imgH = TensorImage.Bitmap.Height;
                imgW = imgH * 3 / 4;
            }
            else
            {
                imgW = TensorImage.Bitmap.Width;
                imgH = imgW * 3 / 4;
            }

            // Converter resultados
            var detections = new List<ObjectDetection>();
            foreach (var detectedObject in detectedObjects)
            {
                var category = new Category() {
                    Label = detectedObject.Label, 
                    Confidence = detectedObject.Confidence
                };


                var box = detectedObject.BoundingBox;
                var bbox = new RectF
                (
                    left: box.Left * imgW,
                    top: box.Top * imgH,
                    right: box.Right * imgW,
                    bottom: box.Bottom * imgH
                );

                var detection = new ObjectDetection
                {
                    BoundingBox = bbox,
                    Category = category
                };

                detections.Add(detection);
            }

            var detectionResult = new DetectionResults
            {
                Image = TensorImage.Bitmap,
                Detections = detections,
                Info = stats
            };
            System.Console.WriteLine($"Inference Result:{System.Text.Json.JsonSerializer.Serialize(stats.InferenceTime)}");

            return detectionResult;
           
        }

        public Bitmap PreProcess(Bitmap bitmap)
        {
            Bitmap resizedBitmap = Bitmap.CreateScaledBitmap(bitmap,ModelConfig.INPUT_SIZE, ModelConfig.INPUT_SIZE, true);
            return resizedBitmap;
        }
        public void SetInputOptim(Bitmap bitmap)
        {
            int width = bitmap.Width;
            int height = bitmap.Height;

            if (intValues == null)
            {
                intValues = new int[ModelConfig.INPUT_SIZE * ModelConfig.INPUT_SIZE];
                bytes = new byte[width * height * 3];

                int batchSize = 1;
                int rgb = 3;
                int numPixels = ModelConfig.INPUT_SIZE * ModelConfig.INPUT_SIZE;
                int bufferSize = batchSize * rgb * numPixels * NUM_BYTES_PER_CHANNEL;
                imgData = ByteBuffer.AllocateDirect(bufferSize);
                imgData.Order(ByteOrder.NativeOrder());

                outData = ByteBuffer.AllocateDirect(outputShape2 * outputShape3 * NUM_BYTES_PER_CHANNEL);
                outData.Order(ByteOrder.NativeOrder());
            }

            // Pega os pixels do bitmap no array intValues
            bitmap.GetPixels(intValues, 0, width, 0, 0, width, height);

            // Chama o método nativo
            ImageProcessing.Argb2Yolo(intValues, imgData.GetDirectBufferAddress(), width, height);

            imgData.Rewind();
            inputArray = new Java.Lang.Object[] { imgData };
            outputMap = new JavaDictionary<Integer, Java.Lang.Object>();
            outData.Rewind();
            outputMap[0] = outData;
        }

        public List<DetectedObject> RunInference()
        {
            if (interpreter != null)
            {
                // Obtém o shape do tensor de saída
                int[] outputShape = interpreter.GetOutputTensor(0).Shape();

                outputShape2 = outputShape[1];
                outputShape3 = outputShape[2];
                output = new float[outputShape2][];

                for (int i = 0; i < outputShape2; i++)
                {
                    output[i] = new float[outputShape3];
                }

                var stopwatch = Stopwatch.StartNew();

                // Executa a inferência
                interpreter.RunForMultipleInputsOutputs(inputArray, (IDictionary<Integer, Java.Lang.Object>) outputMap);

                stopwatch.Stop();
                stats.InferenceTime = stopwatch.ElapsedMilliseconds;

                if (outputMap.TryGetValue((Integer)0, out var outputBufferObj) && outputBufferObj is ByteBuffer byteBuffer)
                {
                    byteBuffer.Rewind();

                    for (int j = 0; j < outputShape2; ++j)
                    {
                        for (int k = 0; k < outputShape3; ++k)
                        {
                            output[j][k] = byteBuffer.Float;
                        }
                    }

                    stopwatch.Restart();

                    var result = Postprocess(
                        output,
                        outputShape3,
                        outputShape2,
                        (float)confidenceThreshold,
                        (float)iouThreshold,
                        numItemsThreshold,
                        numClasses,
                        labels
                    );

                    stopwatch.Stop();
                    stats.PostProcessTime = stopwatch.ElapsedMilliseconds;
                    System.Console.WriteLine($"PostProcessTime Result:{System.Text.Json.JsonSerializer.Serialize(stats.PostProcessTime)}");

                    return result;
                }
            }

            return new List<DetectedObject>();
        }



        public IDictionary<string, Float> GetLabeledProbability(IList<string> labels, TensorBuffer outputProbabilityBuffer)
        {
            TensorLabel tensorLabel = new TensorLabel(labels, outputProbabilityBuffer);
            return tensorLabel.MapWithFloatValue;
        }
        private const int MAX_RESULTS = 10; // Assuming MAX_RESULTS is 10, adjust if needed
        public List<Result> GetTopKProbability(IDictionary<string, Float> labelProb)
        {
            var pq = new SortedSet<Result>(Comparer<Result>.Create((lhs, rhs) => rhs.Confidence.CompareTo(lhs.Confidence)));

            foreach (var pair in labelProb)
            {
                pq.Add(new Result(pair.Key, pair.Key, pair.Value, RectangleF.Empty)); // Assuming default RectF.Empty
            }

            var recognitions = new List<Result>();
            int recognizeSize = Math.Min(pq.Count, MAX_RESULTS);

            for (int i = 0; i < recognizeSize; i++)
            {
                recognitions.Add(pq.Min);
                pq.Remove(pq.Min);
            }

            return recognitions;
        }

       

        public static List<int> ApplyNms(List<DetectedObject> boxes, float iouThreshold)
        {
            var picked = new List<int>();
            var used = new bool[boxes.Count];

            for (int i = 0; i < boxes.Count; i++)
            {
                if (used[i]) continue;

                picked.Add(i);
                var a = boxes[i].BoundingBox;

                for (int j = i + 1; j < boxes.Count; j++)
                {
                    if (used[j]) continue;

                    var b = boxes[j].BoundingBox;
                    float iou = CalculateIoU(a, b);

                    if (iou > iouThreshold)
                    {
                        used[j] = true;
                    }
                }
            }

            return picked;
        }

        public static float CalculateIoU(RectF a, RectF b)
        {
            float interLeft = Math.Max(a.Left, b.Left);
            float interTop = Math.Max(a.Top, b.Top);
            float interRight = Math.Min(a.Right, b.Right);
            float interBottom = Math.Min(a.Bottom, b.Bottom);

            float interArea = Math.Max(0, interRight - interLeft) * Math.Max(0, interBottom - interTop);

            float areaA = (a.Right - a.Left) * (a.Bottom - a.Top);
            float areaB = (b.Right - b.Left) * (b.Bottom - b.Top);

            float unionArea = areaA + areaB - interArea;

            return unionArea > 0 ? interArea / unionArea : 0;
        }

        public static List<DetectedObject> Postprocess(
            float[][] recognitions,
            int w, int h,
            float confidenceThreshold,
            float iouThreshold,
            int numItemsThreshold,
            int numClasses,
            IList<string> labels
        )
        {
            var proposals = new List<DetectedObject>();
            var objects = new List<DetectedObject>();

            // Process recognitions
            for (int i = 0; i < w; i++)
            {
                float maxScore = float.MinValue;
                int classIndex = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    if (recognitions[c + 4][i] > maxScore)
                    {
                        maxScore = recognitions[c + 4][i];
                        classIndex = c;
                    }
                }

                if (maxScore > confidenceThreshold)
                {
                    float dx = recognitions[0][i];
                    float dy = recognitions[1][i];
                    float dw = recognitions[2][i];
                    float dh = recognitions[3][i];

                    var rect = new RectF(
                        dx - dw / 2,
                        dy - dh / 2,
                        dx + dw / 2,
                        dy + dh / 2
                    );

                    string label = (labels != null && classIndex < labels.Count) ? labels[classIndex] : "Unknown";

                    var obj = new DetectedObject(maxScore, rect, classIndex, label);
                    proposals.Add(obj);
                }
            }

            // Sort proposals by confidence descending
            proposals.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

            // Apply NMS
            var picked = ApplyNms(proposals, iouThreshold);

            int count = Math.Min(picked.Count, numItemsThreshold);
            for (int i = 0; i < count; i++)
            {
                objects.Add(proposals[picked[i]]);
            }

            // Clamp coordinates and return results
            var result = new List<DetectedObject>();
            foreach (var obj in objects)
            {
                float left = Math.Max(0, obj.BoundingBox.Left);
                float top = Math.Max(0, obj.BoundingBox.Top);
                float right = Math.Min(1, obj.BoundingBox.Right);
                float bottom = Math.Min(1, obj.BoundingBox.Bottom);

                var boundingBox = new RectF(left, top, right, bottom);

                result.Add(new DetectedObject(
                    obj.Confidence,
                    boundingBox,
                    obj.ClassIndex,
                    obj.Label
                ));
            }

            return result;
        }


        public static ByteBuffer ConvertBitmapToByteBuffer(Bitmap bitmap, int tensorWidth, int tensorHeight)
        {
            Bitmap resizedBitmap = Bitmap.CreateScaledBitmap(bitmap, tensorWidth, tensorHeight, true);

            int batchSize = 1;
            int numChannels = 3; // RGB
            ByteBuffer inputBuffer = ByteBuffer.AllocateDirect(batchSize * tensorWidth * tensorHeight * numChannels * 4);
            inputBuffer.Order(ByteOrder.NativeOrder());

            for (int y = 0; y < tensorHeight; y++)
            {
                for (int x = 0; x < tensorWidth; x++)
                {
                    int pixel = resizedBitmap.GetPixel(x, y);
                    float r = Color.GetRedComponent(pixel) / 255.0f;
                    float g = Color.GetGreenComponent(pixel) / 255.0f;
                    float b = Color.GetBlueComponent(pixel) / 255.0f;

                    inputBuffer.PutFloat(r);
                    inputBuffer.PutFloat(g);
                    inputBuffer.PutFloat(b);
                }
            }

            return inputBuffer;
        }

    }

    public class BoundingBox
    {
        public float X1, Y1, X2, Y2, Cx, Cy, Width, Height, Confidence;
        public int ClassIndex;
        public string ClassName;

        public BoundingBox(float x1, float y1, float x2, float y2, float cx, float cy, float w, float h, float conf, int classIdx, string className)
        {
            X1 = x1; Y1 = y1; X2 = x2; Y2 = y2; Cx = cx; Cy = cy; Width = w; Height = h; Confidence = conf;
            ClassIndex = classIdx; ClassName = className;
        }
    }
    public static class ModelConfig
    {
        public const float INPUT_MEAN = 0f;
        public const float INPUT_STANDARD_DEVIATION = 255f;
        public static readonly Xamarin.TensorFlow.Lite.DataType INPUT_IMAGE_TYPE = Xamarin.TensorFlow.Lite.DataType.Float32;
        public static readonly Xamarin.TensorFlow.Lite.DataType OUTPUT_IMAGE_TYPE = Xamarin.TensorFlow.Lite.DataType.Float32;
        public const float CONFIDENCE_THRESHOLD = 0.7f;
        public const float IOU_THRESHOLD = 0.5f;
        public const float IMAGE_MEAN = 127.0f;
        public const float IMAGE_STD = 128.0f;
        public const float PROBABILITY_MEAN = 0.0F;
        public const float PROBABILITY_STD = 1.0f;
        public const string LABEL_FILENAME = "labels.txt";
        public const string MODEL_FILENAME = "yolo11n_float32.tflite";
        public static int INPUT_SIZE = 320;

    }
    public class Result
    {
        public string Id { get; set; }
        public string Title { get; set; }
        public Float Confidence { get; set; }
        public RectangleF Location { get; set; }

        public Result(string id, string title, Float confidence, RectangleF location)
        {
            Id = id;
            Title = title;
            Confidence = confidence;
            Location = location;
        }

        public override string ToString()
        {
            string resultString = "";
            if (Id != null) resultString += $"[{Id}] ";
            if (Title != null) resultString += $"{Title} ";
            if (Confidence != null) resultString += $"{Confidence.FloatValue() * 100.0f:F1}% ";
            if (Location != null) resultString += Location.ToString() + " ";
            return resultString.Trim();
        }
    }

    public class RectF
    {
        public float Left, Top, Right, Bottom;

        public RectF(float left, float top, float right, float bottom)
        {
            Left = left;
            Top = top;
            Right = right;
            Bottom = bottom;
        }

        public float Width => Right - Left;
        public float Height => Bottom - Top;
    }
}