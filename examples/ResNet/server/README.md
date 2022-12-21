# Test Web Server

After training, take the `SimpleImageClassifier-xxxxx.nn` file and the `classes.txt` from this folder and run:

```bash
$ ResNetServer 9080 SimpleImageClassifier-xxxxx.nn classes.txt
```

On http://localhost:9080, you will find a form where you can upload an image file and a _Classify_ button. The output is either HTML or JSON encoded. The output contains the scores for the categories. You can update the file `classes.txt` according to your own project.

