# Vehicle-License-Plate-System-by-FCN

## Intruction

It's a Vehicle License Plate System using fully connected neural network &amp; orthogonal projection method

_Trainning model has been uploaded to the project._

_这是采用了垂直投影法和全连接型神经网络训练出的车牌划分+识别的系统。训练模型已经上传至源码。_

给出一张原图：
Given a vehicle license plate picture:

 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/160.jpg?raw=true )
 
 it will be transfered as below:
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/216.jpg?raw=true )
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/217.jpg?raw=true )
 
 Then, devide the picture by orthogonal projection:
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/177.jpg?raw=true )
 
 To the picture like this:
 
 ![pic](https://github.com/AdamAlive/MarkdownRef/blob/master/159.jpg?raw=true )
 
 and will show you the result.
 
 ```
 这个车牌号为：
2017-10-25 17:44:20.013102: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
2017-10-25 17:44:25.774811: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
G
2017-10-25 17:44:30.278381: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
2017-10-25 17:44:30.345072: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
Q
2017-10-25 17:44:31.629228: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
2017-10-25 17:44:31.697049: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
K
2017-10-25 17:44:34.551740: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
2017-10-25 17:44:34.667551: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
8
2017-10-25 17:44:35.412358: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
2017-10-25 17:44:35.477273: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
8
2017-10-25 17:44:36.207343: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
2017-10-25 17:44:36.269908: I C:\tf_jenkins\home\workspace\rel-win\M\windows-gpu\PY\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0)
3
 ```
 
 *********************
 
 ## How to use
 
 Just use `` python devide_plate.py `` will be enough.
