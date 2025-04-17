using System.Collections.Generic;
using System.Text.Json.Serialization;
using Android.Graphics;
using BarcodeScanner.Mobile;
using Microsoft.Maui.Controls.PlatformConfiguration;
using Microsoft.Maui.Controls.PlatformConfiguration.iOSSpecific;
using Newtonsoft.Json;
using SkiaSharp;
using SkiaSharp.Views.Maui;

namespace SampleApp.Maui;

public partial class Page1 : ContentPage
{
    //readonly SKPaint paint = new SKPaint
    //{
    //    Style = SKPaintStyle.Stroke,
    //    Color = Color.BlueViolet.ToSKColor(),
    //    StrokeWidth = 4
    //};

    private DetectionResults DetectionResults = new DetectionResults();

    public Page1()
    {
        InitializeComponent();
        BarcodeScanner.Mobile.Methods.SetSupportBarcodeFormat(BarcodeFormats.Code39 | BarcodeFormats.QRCode | BarcodeFormats.Code128);
        On<iOS>().SetUseSafeArea(true);
    }


    private async void CancelButton_Clicked(object sender, EventArgs e)
    {
        await Navigation.PopModalAsync();
    }

    private void FlashlightButton_Clicked(object sender, EventArgs e)
    {
        Camera.TorchOn = !Camera.TorchOn;
    }

    private void SwitchCameraButton_Clicked(object sender, EventArgs e)
    {
        Camera.CameraFacing = Camera.CameraFacing == CameraFacing.Back
                                  ? CameraFacing.Front
                                  : CameraFacing.Back;
    }

    private void CameraView_OnDetected(object sender, OnDetectedEventArg e)
    {
        DetectionResults = e.DetectionResults;
        if (DetectionResults == null || DetectionResults.Detections.Count == 0) return;

        var bitmap = (Bitmap)DetectionResults.Image;
        int imageWidth = bitmap.Width;
        int imageHeight = bitmap.Height;
        SetDetections(DetectionResults.Detections, imageWidth, imageHeight);
        //string result = string.Empty;
        //for (int i = 0; i < obj.Count; i++)
        //{
        //    result += $"Type : {obj[i].BarcodeType}, Value : {obj[i].DisplayValue}{Environment.NewLine}";
        //}

        //this.Dispatcher.Dispatch(async () =>
        //{
        //    await DisplayAlert("Result", result, "OK");
        //    Camera.IsScanning = true;
        //});

        //Canvas.InvalidateSurface();

    }
    private List<ObjectDetection> _detections = new();
    private float _scaleFactor = 1f;
    private int _imageWidth = 1, _imageHeight = 1;

    public void SetDetections(List<ObjectDetection> detections, int imageWidth, int imageHeight)
    {
        _detections = detections;
        _imageWidth = imageWidth;
        _imageHeight = imageHeight;
        Canvas.InvalidateSurface();
        Console.WriteLine(JsonConvert.SerializeObject(_detections));
    }
    void SKCanvasView_PaintSurface(object sender, SKPaintSurfaceEventArgs e)
    {
        var canvas = e.Surface.Canvas;
        canvas.Clear();

        var info = e.Info;

        _scaleFactor = Math.Max(info.Width / (float)_imageWidth, info.Height / (float)_imageHeight);

        using var boxPaint = new SKPaint
        {
            Color = SKColors.LimeGreen,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = 4
        };

        using var textPaint = new SKPaint
        {
            Color = SKColors.Yellow,
            TextSize = 32,
            IsAntialias = true
        };

        using var bgPaint = new SKPaint
        {
            Color = SKColors.Black,
            Style = SKPaintStyle.Fill
        };

        foreach (var det in _detections)
        {
            var box = det.BoundingBox;

            float left = box.Left * _scaleFactor;
            float top = box.Top * _scaleFactor;
            float right = box.Right * _scaleFactor;
            float bottom = box.Bottom * _scaleFactor;

            canvas.DrawRect(SKRect.Create(left, top, right - left, bottom - top), boxPaint);

            var label = $"{det.Category.Label} {det.Category.Confidence:P1}";
            var textBounds = new SKRect();
            textPaint.MeasureText(label, ref textBounds);

            // Fundo do texto
            canvas.DrawRect(new SKRect(left, top - textBounds.Height, left + textBounds.Width + 10, top), bgPaint);

            // Texto
            canvas.DrawText(label, left + 5, top - 5, textPaint);
        }
        Camera.IsScanning = true;
    }

}