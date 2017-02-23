# Tensorflow Tutorials in Hylang

```clojure
➜  ~ hy
hy 0.12.1+24.g45b7a4a using CPython(default) 2.7.13 on Darwin
=> (import tensorflow)
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.dylib locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.dylib locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.dylib locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.1.dylib locally
I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.dylib locally
=> (import numpy)
=> (setv x_data (.astype (numpy.random.rand 100) numpy.float32))
=>
=> (setv y_data (+ (* x_data 0.1) 0.3))
=>
=> (setv Weights (tensorflow.Variable (tensorflow.random_uniform [1] -1.0 1.0)))
=> (setv biases (tensorflow.Variable (tensorflow.zeros [1])))
=> (setv y (+ (* x_data Weights) biases))
=> (setv loss (tensorflow.reduce_mean (tensorflow.square (- y y_data))))
=> (setv optimizer (tensorflow.train.GradientDescentOptimizer 0.5))
=> (setv train (optimizer.minimize loss))
=> (setv sess (tensorflow.Session))
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:901] OS X does not support NUMA - returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_device.cc:885] Found device 0 with properties:
name: GeForce GT 650M
major: 3 minor: 0 memoryClockRate (GHz) 0.9
pciBusID 0000:01:00.0
Total memory: 1023.69MiB
Free memory: 129.24MiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:906] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:916] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:975] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GT 650M, pci bus id: 0000:01:00.0)
=> (setv init (tensorflow.global_variables_initializer))
=> (sess.run init)
=> (for [step (range 201)]
...   (do
...    (sess.run train)
...    (if (= (% step 20) 0)
...      (print step (sess.run Weights) (sess.run biases))
...      )
...    )
...   )
0 [ 0.55023992] [ 0.11040857]
20 [ 0.22272126] [ 0.24291013]
40 [ 0.13402393] [ 0.28417209]
60 [ 0.10943299] [ 0.2956118]
80 [ 0.10261524] [ 0.29878339]
100 [ 0.10072506] [ 0.29966271]
120 [ 0.10020103] [ 0.29990649]
140 [ 0.10005574] [ 0.29997408]
160 [ 0.10001545] [ 0.29999283]
180 [ 0.10000428] [ 0.29999802]
200 [ 0.10000119] [ 0.29999948]
=>
```
