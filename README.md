## BUILD TENSORRT TEST:
```
mkdir build
cd build
cmake ..
make
```

## RUN TENSORRT:
```
./tensorrt_test -h
```

## RUN ACCURACY TEST:
From https://www.cityscapes-dataset.com/downloads/ download: 
* leftImg8bit_trainvaltest.zip (11GB)
* gtFine_trainvaltest.zip (241MB)

Download and extract to `datasets/Cityscapes`
```
cd accuracy-test
python trt_inference.py
```

## CONVERT SWIFTNET MODEL TO ONNX:
```
cd model
nano swiftnet_to_onnx.py 
# change pre-trained model path and input resolution (must be the same as in models/util.py)
python swiftnet_to_onnx.py
```

## RUN TORCH EVALUATION:
```
cd model
nano torch_timer_eval.py
# change pre-trained model path and input resolution (must be the same as in models/util.py)
python torch_timer_eval.py
```

## DOWNLOAD WEIGHTS:
wget http://elbereth.zemris.fer.hr/swiftnet/swiftnet_ss_cs.pt -P weights/

## SWIFTNET MODEL PUBLIC REPOSITORY:
https://github.com/orsic/swiftnet/
