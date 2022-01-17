## complex-valued DnCNN

Source code for [Deep Denoising Neural Network Assisted Compressive Channel Estimation for mmWave Intelligent Reflecting Surfaces](https://ieeexplore.ieee.org/document/9127834), modified from [DnCNN](https://github.com/cszn/DnCNN). 

```
S. Liu, Z. Gao, J. Zhang, M. D. Renzo and M. -S. Alouini, "Deep Denoising Neural Network Assisted Compressive Channel Estimation for mmWave Intelligent Reflecting Surfaces," in IEEE Transactions on Vehicular Technology, vol. 69, no. 8, pp. 9223-9228, Aug. 2020, doi: 10.1109/TVT.2020.3005402.
```


To successfully run the code, you should follow the steps below

- Make sure you have installed the required libs

- Generate channel dataset. We have used the geometric channel model in our paper with vary parameters. 

Note that the required shape is $[N,2,N_{IRS},N_{C}]$, where 2 represents the real part and imaginary part of the channel matrix.

- Set the parameters in `modelTrain.py` and create the corresponding folders before running the program.
- *If you want to evalutate the performance, please change the dataset configurations in `modelTrain.py`*

---

文章 [Deep Denoising Neural Network Assisted Compressive Channel Estimation for mmWave Intelligent Reflecting Surfaces](https://ieeexplore.ieee.org/document/9127834) 的源代码, 由 [DnCNN](https://github.com/cszn/DnCNN) 等修改。

为保证成功运行，您应按照以下步骤进行

- 确保您已安装所有需要的库
- 产生一组信道数据集。此处我们采用几何信道建模，参数在仿真中有变。

值得注意的是，信道数据集的尺寸要求为$[N,2,N_{IRS},N_{C}]$，其中2代表信道矩阵的实部和虚部。

- 运行`modelTrain.py`进行训练，在这之前要设置合适的参数（用于区别保存模型时的路径）。
- *若需要测试性能表现，请将训练文件中的数据集配置换成测试集*

---

> 原文采用OMP算法构造数据集，该部分代码可能在后续整理中发布。
