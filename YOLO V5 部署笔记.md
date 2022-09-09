[TOC]

# YOLO V5 部署笔记

## 安装环境

### 安装显卡驱动

命令行中输入命令`nvidia-smi`查看是否安装CUDA，

![](imgs/YOLO%20V5%20部署笔记.md/2022-08-14-15-02-59.png)

若未安装CUDA，在 [NVIDIA 官网](https://developer.nvidia.com/cuda-downloads)下载 CUDA ToolKit 并安装。

### 安装Anaconda

下载安装 [Anaconda](https://www.anaconda.com/products/distribution) 用来管理 Python 环境。

### 安装Pytorch

#### 创建虚拟环境

安装完成Anaconda后，运行`Anaconda Prompt`，执行如下命令以创建一个新环境：

`conda create -n myPytorch python=3.8`

上述命令创建了一个名为`myPytorch`的环境，Python版本指定为3.8。

接着执行`activate myTorch`激活该环境。

接下来需要安装Pytorch，在安装之前，最后切换源以提高下载速度。

参照 [清华大学开源软件镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) 切换第三方源。

#### 安装Pytorch

在[Pytorch官网](https://pytorch.org/get-started/locally/)选择Pytorch安装配置，得到一行安装命令。

![](imgs/YOLO%20V5%20部署笔记.md/2022-08-14-18-50-42.png)

`conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`

### 安装LabelImg

在myPytorch环境下，运行如下命令安装数据标注工具LabelImg

`pip install labelimg -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 数据标注

使用`labelimg`命令打开labelimg工具，进行数据标注。

点击`Open`选择图片所在的目录：

![](imgs/YOLO%20V5%20部署笔记.md/2022-08-14-19-07-41.png)

