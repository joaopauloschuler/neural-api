# Neural Networks with OpenCL
This example is very similar to the other
[simple image classifier example](https://github.com/joaopauloschuler/neural-api/new/master/examples/SimpleImageClassifier)
except that this example has OpenCL. The first step is detecting OpenCL platforms such as NVIDIA, AMD or Intel Graphics. 
This is done with:
```
    EasyOpenCL: TEasyOpenCL;
    ...
    EasyOpenCL := TEasyOpenCL.Create();
    if EasyOpenCL.GetPlatformCount() = 0 then
    begin
      WriteLn('No OpenCL capable platform has been found.');
      exit;
    end;
    WriteLn('Setting platform to: ', EasyOpenCL.PlatformNames[0]);
    EasyOpenCL.SetCurrentPlatform(EasyOpenCL.PlatformIds[0]);
```

`PlatformNames` and `PlatformIds` are arrays with detected platforms. This example sets the platform to the
first platform in the array. After setting the platform, we can get devices with:
```
    if EasyOpenCL.GetDeviceCount() = 0 then
    begin
      WriteLn('No OpenCL capable device has been found for platform ',EasyOpenCL.PlatformNames[0]);
      exit;
    end;
    EasyOpenCL.SetCurrentDevice(EasyOpenCL.Devices[0]);
    WriteLn('Setting device to: ', EasyOpenCL.DeviceNames[0]);
```
Then, the optimizer needs to be told to use OpenCL:
```
NeuralFit.EnableOpenCL(EasyOpenCL.PlatformIds[0], EasyOpenCL.Devices[0]);
```
In principle, CAI should work with any OpenCL capable device that supports *on the fly* compilation.
In the case that you wold like to look at a more complex example, there is a much bigger and more complex
[OpenCL image classification example](https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/visualCifar10OpenCL/).
