﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;
using BarcodeScanner.Mobile;
using System.Windows.Input;
using Xamarin.Forms;

namespace SampleApp.XF.Mvvm
{
    public class MvvmDemoViewModel : INotifyPropertyChanged
    {
        private bool _vibrationOnDetected { get; set; }
        public bool VibrationOnDetected
        {
            get { return this._vibrationOnDetected; }
            set
            {
                _vibrationOnDetected = value;
                OnPropertyChanged(nameof(VibrationOnDetected));
            }
        }

        private bool _isScanning { get; set; }
        public bool IsScanning
        {
            get { return _isScanning; }
            set
            {
                _isScanning = value;
                OnPropertyChanged(nameof(IsScanning));
            }
        }

        private ICommand _onDetectCommand { get; set; }
        public ICommand OnDetectCommand
        {
            get { return _onDetectCommand; }
            set
            {
                _onDetectCommand = value;
                OnPropertyChanged(nameof(OnDetectCommand));
            }
        }
        private ICommand _handleVirbationCommand { get; set; }
        public ICommand HandleVirbationCommand
        {
            get { return _handleVirbationCommand; }
            set
            {
                _handleVirbationCommand = value;
                OnPropertyChanged(nameof(HandleVirbationCommand));
            }
        }
        private ICommand _handleFlashlightCommand { get; set; }
        public ICommand HandleFlashlightCommand
        {
            get { return _handleFlashlightCommand; }
            set
            {
                _handleFlashlightCommand = value;
                OnPropertyChanged(nameof(HandleFlashlightCommand));
            }
        }

        private ICommand _handleIsScanningCommand { get; set; }
        public ICommand HandleIsScanningCommand
        {
            get { return _handleIsScanningCommand; }
            set
            {
                _handleIsScanningCommand = value;
                OnPropertyChanged(nameof(HandleIsScanningCommand));
            }
        }




        private bool _torchOn { get; set; }
        public bool TorchOn
        {
            get { return _torchOn; }
            set
            {
                _torchOn = value;
                OnPropertyChanged(nameof(TorchOn));
            }
        }


        private int _scanInterval { get; set; }
        public int ScanInterval
        {
            get { return _scanInterval; }
            set
            {
                _scanInterval = value;
                OnPropertyChanged(nameof(ScanInterval));
            }
        }
        private string result { get; set; }
        public string Result
        {
            get { return result; }
            set
            {
                result = value;
                OnPropertyChanged(nameof(Result));
            }
        }

        private float _zoom { get; set; }
        public float Zoom
        {
            get { return _zoom; }
            set
            {
                _zoom = value;
                OnPropertyChanged(nameof(Zoom));
            }
        }

        public MvvmDemoViewModel()
        {
            this.TorchOn = true;
            this.VibrationOnDetected = true;
            this.ScanInterval = 1000;
            this.IsScanning = true;
            this.OnDetectCommand = new Command<OnDetectedEventArg>(ExecuteOnDetectedCommand);
            this.HandleFlashlightCommand = new Command(() =>
            {
                this.TorchOn = !this.TorchOn;
            });
            this.HandleVirbationCommand = new Command(() => {
                this.VibrationOnDetected = !this.VibrationOnDetected;
            });
            this.HandleIsScanningCommand = new Command(() =>
            {
                this.IsScanning = !this.IsScanning;
            });
            this.Result = string.Empty;
        }
        public void ExecuteOnDetectedCommand(OnDetectedEventArg arg)
        {
            List<BarcodeScanner.Mobile.BarcodeResult> obj = arg.BarcodeResults;

            string result = string.Empty;
            for (int i = 0; i < obj.Count; i++)
            {
                result += $"Type : {obj[i].BarcodeType}, Value : {obj[i].DisplayValue}{Environment.NewLine}";
            }
            Device.BeginInvokeOnMainThread(async () =>
            {
                this.Result = result;
                //await Navigation.PopModalAsync();
            });
        }

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this,
            new PropertyChangedEventArgs(propertyName));
        }
    }
}
