# Learn TensorflowJS

Playground project to learn tensorflowJS.

Setting up tfjs-node-gpu can be a pain, so here are the resources I used to get that working. Since I used Windows 11, with an Nvidia RTX 4090 it made sense setting up gpu support that can accelerate training later on. Only bother with this if ur GPU has CUDA support.

https://blog.quantinsti.com/install-tensorflow-gpu/#:~:text=How%20To%20Install%20TensorFlow%20GPU%20%28With%20Detailed%20Steps%29,Tensorflow%20GPU%20...%207%20Step%207%3A%20Install%20Keras

With the above link you can stop after Step 5.

Always be sure to install Cuda Toolkit, CuDnn version that matches the version of TensorflowJS your installing as seen here
https://github.com/tensorflow/tfjs/tree/c4d11991f60c9232a89c781f17328653ae133169/tfjs-node

More on CuDnn setup here
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

Running a tensor flow example using @tensorflow/tfjs-node-gpu should show a header with the GPU name installed as seen below.
![image](https://github.com/nikilok/learnTensorFlowJS/assets/6220175/66247d9e-9839-4afb-b0f8-6984b23a5743)

