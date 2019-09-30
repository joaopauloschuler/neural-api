# Identity Shortcut Connection
The **identity shortcut connection** is a connection that skips few layers and then is summed to the output of a following 
layer. You can find more about it in the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
and [here](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035).

The main point of attention is the **summation** of outputs. In CAI, this is done via the `TNNetSum` class. This class gets an array
of layers as an input and sums all inputs. For this summation to work, the shape of each input must be the same otherwise you'll
get a run time error. The current example shows this:
```
GlueLayer := NN.AddLayer(TNNetReLU.Create());

NN.AddLayer(TNNetConvolutionReLU.Create({NumFeatures=}64 ,{featureSize=}3, {padding=}1, {stride=}1, {SuppressBias=}0));
NN.AddLayer(TNNetConvolutionLinear.Create({NumFeatures=}64 ,{featureSize=}3, {padding=}1, {stride=}1, {SuppressBias=}0));
NN.AddLayer(TNNetSum.Create([NN.GetLastLayer(), GlueLayer]));
NN.AddLayer(TNNetReLU.Create());
```

This example also detects your OpenCL capable device and falls back to CPU with:
```
EasyOpenCL := TEasyOpenCL.Create();
if EasyOpenCL.GetPlatformCount() > 0 then
begin
  WriteLn('Setting platform to: ', EasyOpenCL.PlatformNames[0]);
  EasyOpenCL.SetCurrentPlatform(EasyOpenCL.PlatformIds[0]);
  if EasyOpenCL.GetDeviceCount() > 0 then
  begin
    EasyOpenCL.SetCurrentDevice(EasyOpenCL.Devices[0]);
    WriteLn('Setting device to: ', EasyOpenCL.DeviceNames[0]);
    NeuralFit.EnableOpenCL(EasyOpenCL.PlatformIds[0], EasyOpenCL.Devices[0]);
  end
  else
  begin
    WriteLn('No OpenCL capable device has been found for platform ',EasyOpenCL.PlatformNames[0]);
    WriteLn('Falling back to CPU.');
  end;
end
else
begin
  WriteLn('No OpenCL platform has been found. Falling back to CPU.');
end;
```
