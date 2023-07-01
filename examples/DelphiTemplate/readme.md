In Delphi, in project options:
* At compiler, search path (-U), you'll  add the "neural" folder: ..\..\neural\
* Still at the compiler, set the final output directory (-E) to: ..\..\bin\x86_64-win64\bin\
* In "generate console application", set it to true.

In your "uses" section, include:
  neuralnetwork, neuralvolume, neuraldatasets, neuralfit, neuralthread;

