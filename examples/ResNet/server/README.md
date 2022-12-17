# Test server

After training, take the `SimpleImageClassifier-xxxxx.nn` file and the `batches.meta.txt` from the source CIFAR and run:

```bash
$ ResNetServer 9080 SimpleImageClassifier-xxxxx.nn batches.meta.txt
```

On http://localhost:9080 you will see a form where you can upload an image file and a _Classify_ button. The output is a JSON file with the scores for the 10 categories.

