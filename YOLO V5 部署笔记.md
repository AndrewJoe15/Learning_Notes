<H1>YOLO V5 部署笔记</H1>

[TOC]

# 1. 安装环境

在部署YOLO之前，需要安装如下环境：

- [Anaconda](#安装anaconda)
  - 管理Python包和环境
- [Pytorch](#安装pytorch)
  - 深度学习框架
- [CUDA](#安装显卡驱动和cuda)
  - 用于模型训练时显卡加速
- [PyCharm](https://www.jetbrains.com/pycharm/)
  - Python 开发 IDE
- LaelImg
  - 图片标注工具
- Git
  - 代码版本控制工具

## 1.1. 安装显卡驱动、CUDA 和 cuDNN

在Nvidia官网选择电脑配置的显卡型号，下载相应的[显卡驱动程序](https://www.nvidia.cn/geforce/drivers/)。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-09-40-45.png)

命令行中输入命令`nvidia-smi`查看是否安装CUDA，

![](imgs/YOLO%20V5%20部署笔记.md/2022-08-14-15-02-59.png)

若未安装CUDA，在 NVIDIA 官网下载 [CUDA ToolKit](https://developer.nvidia.com/cuda-downloads) 并安装。

安装好了驱动和CUDA，最后按照此说明[下载安装 cuDNN](https://blog.csdn.net/jhsignal/article/details/111401628)。

## 1.2. 安装 Anaconda

下载安装 [Anaconda](https://www.anaconda.com/products/distribution) 用来管理 Python 环境。

## 1.3. 安装 Pytorch

### 1.3.1. 创建虚拟环境

安装完成Anaconda后，运行`Anaconda Prompt`，执行如下命令以创建一个新环境：

`conda create -n myPytorch python=3.8`

上述命令创建了一个名为`myPytorch`的环境，Python版本指定为3.8。

接着执行`activate myPytorch`激活该环境。

接下来需要在该环境下安装Pytorch，在安装之前，切换源以提高下载速度。

参照 [清华大学开源软件镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) 切换第三方源的方法。

### 1.3.2. 安装Pytorch

在[Pytorch官网](https://pytorch.org/get-started/locally/)选择Pytorch安装配置，得到一行安装命令。

![](imgs/YOLO%20V5%20部署笔记.md/2022-08-14-18-50-42.png)

`conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`

安装需要相当长的一段时间。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-09-15-33-50.png)

## 1.4. 安装LabelImg

数据标注工具 LabelImg ([Github](https://github.com/heartexlabs/labelImg))

### 1.4.1. 方式1 pip3

在 Anaconda Prompt 中（base环境下）直接执行如下命令安装

`pip3 install labelImg`

执行`labelImg`打开软件

### 1.4.2. 方式2 从源码编译

打开 Git Bash 运行如下命令，下载 LabelImg 源码

`git clone https://github.com/heartexlabs/labelImg.git`

在 Anaconda Prompt 中（base环境下），**cd 到下载的源码路径下**，运行如下命令编译

```C
conda install pyqt=5
conda install -c anaconda lxml
pyrcc5 -o libs/resources.py resources.qrc
```

执行`python labelImg.py`打开软件

### 1.4.3. 方式3 exe

下载exe文件 [百度网盘 提取码: vj8f](https://pan.baidu.com/s/1yk8ff56Xu40-ZLBghEQ5nw)

## 1.5. 安装Git

选择Windows版本的[Git安装包](https://git-scm.com/download/win)下载安装，所有安装配置默认即可。

# 2. YOLO V5 模型训练

## 2.1. 下载 YOLO V5

在想要存放 YOLO V5 的文件夹中右键选择`Git Bash Here`，打开`Git Bash`

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-09-17-06-41.png)

执行如下命令将 YOLO V5 代码下载到本地。

`git clone https://github.com/ultralytics/yolov5`

## 2.2. 安装所需的第三方库

使用 Pycharm 打开 YOLO V5 项目

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-11-37-35.png)


在右下角点击添加 Python 解释器

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-11-38-20.png)

选择之前创建的 Anaconda myPytorch 环境

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-11-39-44.png)

加载环境后的 Python Console 窗口

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-11-45-03.png)


点击 Pycharm 下方的 Terminal

如果显示无法进入虚拟环境，则打开 `Anaconda Prompy` 执行如下命令初始化powershell

`conda init powershell`

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-15-45-57.png)

此时若打开命令行窗口时提示`无法加载文件……在此系统上禁止运行脚本……`，则以管理员身份运行 PowerShell

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-15-50-51.png)

执行 `set-ExecutionPolicy RemoteSigned` 然后输入 `Y` , 更改执行策略。

最后点击 Pycharm 下方的 Terminal 可以正常进入 myPytorch 环境

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-16-08-21.png)

执行如下命令安装YOLO所需的包

`pip install -r requirements.txt`

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-16-01-26.png)

## 2.3. 制作数据集

### 2.3.1. 数据标注

使用`labelimg`命令打开labelimg工具，进行数据标注。

点击`Open Dir`选择图片所在的目录（**注意路径中不能含有中文，否则会闪退**）：

![](imgs/YOLO%20V5%20部署笔记.md/2022-08-14-19-07-41.png)

点击`Change Save Dir`选择标注文件保存的目录，此后每次保存操作都会自动将当前图片对应的标注文件保存在该目录中。

数据格式选择默认的 PascalVOC 即可

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-09-20-34.png)

> 在 LabelImg 中也可以选择 YOLO 格式的输出文件，但我们这里选择 PascalVOC， 之后我们会将 PascalVOC 数据格式转换成 YOLO 的标注文件格式，这么做是因为 PascalVOC 格式的数据使用 XML 存储，较为直观，数据内容也更丰富。

按 `W` 键添加矩形框，并为之添加标签。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-09-13-48.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-13-17-18.png)

一张图片标注完成后，按 `Ctrl` + `C` 保存，然后按 `A`/`D` 切换上/下一张图片，依次完成所有图片的标注。 

### 2.3.2. 将数据集添加到YOLO项目中

#### 2.3.2.1. 拷贝数据集

用 Pycharm 打开 yolov5 项目，在根目录下添加如下几个文件夹

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-09-09-55.png)

将标注文件和图片分别拷贝到 `Annotations` 和 `Images` 文件夹下。
> 在用LabelImg进行数据标注的时候，也可以直接选择这两个文件夹作为标注文件输出目录和图片文件目录，从而省去这里的拷贝操作

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-09-14-03.png)

#### 2.3.2.2. 数据格式转换

在之前我们用 LabelImg 进行数据标注时，使用了 VOC 格式，要在 YOLO 中使用需要转成 YOLO 的数据格式。

在 `util` 文件夹下添加 `xml2yolo.py` 用来实现这一功能。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-13-09-28.png)

`util/xml2yolo.py`

```Python
import xml.etree.ElementTree as ET
import os
import random
from shutil import copyfile, rmtree
from utils.general import Path

# 分类
names = ["OK", "NG"]

# 训练集划分比例（百分数），剩下的为验证集
TRAIN_RATIO = 80


def convert(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def convert_annotation(xml_dir, txt_dir, image_id):
    if not os.path.isfile(xml_dir + '%s.xml' % image_id):
        return False
    in_file = open(xml_dir + '%s.xml' % image_id)
    out_file = open(txt_dir + '%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in names or int(difficult) == 1:
            continue
        cls_id = names.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()
    return True

# 保证文件夹存在且是空的
def check_dir(file_dir):
    if os.path.isdir(file_dir):
        rmtree(file_dir)
    os.mkdir(file_dir)

# 数据集路径
data_base_dir = "../datasets/"
# data_base_dir = Path(yaml['path'])
# 项目数据集路径
work_sapce_dir = os.path.join(data_base_dir, "Ampoule/")
annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
image_dir = os.path.join(work_sapce_dir, "Images/")
yolo_labels_dir = os.path.join(work_sapce_dir, "Labels/")
# 训练/验证集路径
# 图片
yolov5_images_dir = os.path.join(data_base_dir, "images/")
yolov5_images_train_dir = os.path.join(yolov5_images_dir, "train/")
yolov5_images_val_dir = os.path.join(yolov5_images_dir, "val/")
# 标签
yolov5_labels_dir = os.path.join(data_base_dir, "labels/")
yolov5_labels_train_dir = os.path.join(yolov5_labels_dir, "train/")
yolov5_labels_val_dir = os.path.join(yolov5_labels_dir, "val/")

# 检查训练数据文件目录，有则清空，则没有则创建
check_dir(yolov5_images_dir)
check_dir(yolov5_labels_dir)
check_dir(yolov5_images_train_dir)
check_dir(yolov5_images_val_dir)
check_dir(yolov5_labels_train_dir)
check_dir(yolov5_labels_val_dir)

list_imgs = os.listdir(image_dir)  # 图片文件列表

sum_train = sum_val = 0

for i in range(0, len(list_imgs)):
    path = os.path.join(image_dir, list_imgs[i])
    if os.path.isfile(path):
        image_path = image_dir + list_imgs[i]
        image_name = list_imgs[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
    # 转换xml格式为yolo格式标签
    if convert_annotation(annotation_dir, yolo_labels_dir, nameWithoutExtention):
        # 放到训练集或验证集
        prob = random.randint(1, 100)  # 用来划分数据集的随机数
        if prob < TRAIN_RATIO:  # train dataset
            if os.path.exists(annotation_path):
                sum_train += 1
                copyfile(image_path, yolov5_images_train_dir + image_name)
                copyfile(label_path, yolov5_labels_train_dir + label_name)
        else:  # val dataset
            if os.path.exists(annotation_path):
                sum_val += 1
                copyfile(image_path, yolov5_images_val_dir + image_name)
                copyfile(label_path, yolov5_labels_val_dir + label_name)
print('标注文件转换与数据集划分完成，训练集数量：' + sum_train.__str__() + ', 验证集数量：' + sum_val.__str__());
```

上述Python程序主要实现了

- 标注文件数据格式转换
  - 读取 `datasets/Ampoule/Annotations` 文件夹下的标注文件数据，写入到`datasets/Ampoule/Labels` 对应 `txt` 文件中。
- 训练集/验证集划分
  - 按照设定的划分比例，将数据集每一个实例随机分配，图片和标注文件分别拷贝到自动创建的`datasets/images`和`datasets/labels`文件夹中。

在 `xml2yolo.py` 中右键，点击 `Run` 执行程序

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-13-25-00.png)

可以看到数据集成功拷贝到`datasets/images`和`datasets/labels`的相应文件夹中。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-13-24-38.png)

### 2.3.3. 新建配置文件

我们已经将数据集放到 YOLO 项目中，接下来需要新建一个该数据集对应的配置文件，用来告诉 YOLO 此数据集的目录位置。

在 `data` 下新建 `Ampoule.yaml` 文件

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-13-46-23.png)

其内容如下，指定了数据集所在目录，在该目录下图片的训练集和验证集的目录，并列出了数据集的所有种类名称。

```XML
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: datasets
train: images/train
val: images/val
test: # images/test # test images (optional)

# Classes
names:
  0: OK
  1: NG
```

## 2.4. 训练

### 2.4.1. 下载预训练模型

在[此页面](https://github.com/ultralytics/yolov5/releases)下载预训练模型，即YOLO网络模型的的权重文件

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-17-35-40.png)

本文选择了`yolov5s.pt`，将其添加到YOLO根目录下

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-17-43-47.png)

`yolov5s.pt` 相当于我们训练的起点，使用我们自己的数据集进行训练后，会得到我们自己的权重文件。

### 2.4.2. 开始训练

做好了以上所有的准备工作，接下来就可以训练我们的YOLO了，首先打开根目录下的 `train.py` ，在编辑器中右键选择 `Moddify Run Configuration...`，配置训练参数。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-13-31-21.png)

在 `Parameters` 一栏填上

`--weights yolov5s.pt --data data/Ampoule.yaml`

即指定使用我们之前下载的预训练模型 `yolov5s.pt` 进行训练，数据集配置文件是我们的数据集对应的 `Ampoule.yaml`

训练参数保存后，点击 `Run` 开始训练。

> 如果遇到`[WinError 1455] 页面文件太小，无法完成操作`的错误，则需要更改电脑的虚拟内存，越大越好。具体参考[此文](https://blog.csdn.net/weixin_42067873/article/details/120887060)。

训练结束后，可以看到最终的结果和权重文件保存的位置

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-14-01-21.png)

打开 `runs/train/exp19` 文件夹，可以看到训练过程中的一些统计信息，`weights` 中存放的是最后一个周期的权重文件和整个训练期间最优的权重文件。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-14-04-38.png)

## 2.5. 测试

打开根目录下的 `detect.py`，该文件是真正用来进行目标检测的程序

在运行前先对程序进行配置

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-14-11-17.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-14-34-36.png)

在 `Parameters` 一栏填入参数

`--weights runs\train\exp19\weights\best.pt --source datasets/Ampoule/Images --data data/Ampoule.yaml --max-det 6 --device 0`

- weights 模型权重文件，训练得到的 `best.pt`
- source 即数据源，用来指定需要进行目标检测的图片、视频等 设置为0则为摄像头。
- max-det 最大检测数量，在本文中，每个图片都是6个物体，因此填6
- device 模型运算使用的设备，GPU或CPU，0表示默认可用的第一个GPU

点击运行，打开文件夹查看检测效果

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-14-44-46.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-14-45-17.png)

## 2.6. 输出

为了之后的部署，我们需要将训练好的模型输出为`onnx`文件。

打开 `export.py`，首先按照注释中的说明安装依赖包

```Py
$ pip install -r requirements.txt onnx onnx-simplifier onnxruntime  # CPU
$ pip install -r requirements.txt onnx onnx-simplifier onnxruntime-gpu  # GPU
```

然后配置运行参数如下

`--data data/Ampoule.yaml --weights runs/train/exp19/weights/best.pt --device 0 --include onnx`

点击运行，成功信息如下

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-14-54-27.png)

该`onnx`文件就是我们后期进行部署要用的模型文件

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-15-14-55-11.png)

使用我们输出的这个模型文件作为权重，将 `detect.py` 运行配置的 `weights` 改成 `best.onnx`的路径。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-13-19-16.png)

点击运行，在输出文件夹中查看效果。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-13-21-43.png)

# 3. 部署

接下来我们将使用输出的 `onnx` 文件将 YOLO 部署到我们自己的项目中。

我们需要一个推理框架可以使用 `onnx` 模型文件推理输出结果，对结果进行解析，并标出检测到的目标。

为了实现这一点，我们可以使用Github上的一个开源仓库[yolov5-net](https://github.com/mentalstack/yolov5-net)。

## 3.1. yolov5-net

本节的是对[yolov5-net](https://github.com/mentalstack/yolov5-net)项目的简单介绍，最终我们需要它生成`Yolov5Net.Scorer.dll`引用添加到我们最终的项目中，如果对此节内容不感兴趣，可以直接在我的仓库中下载[Yolov5Net.Scorer.dll](https://github.com/AndrewJoe15/RP_YOLO/blob/master/dll/)，并跳过这一节。

### 3.1.1. 下载

在 Git Bash 中执行如下命令来下载 `yolov5-net`

```Git
https://github.com/mentalstack/yolov5-net.git
```

打开VS工程文件

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-13-45-25.png)

打开NutGet包管理器，下载需要的包

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-13-48-11.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-14-41-53.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-14-42-32.png)

### 3.1.2. 使用 yolov5-net 进行目标检测

打开 `YoloV5Net-App` 项目下的 `Program.cs`，其 `Main` 函数如下

```CSharp
  static void Main(string[] args)
  {
      using var image = Image.FromFile("Assets/test.jpg");

      using var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5s.onnx");

      List<YoloPrediction> predictions = scorer.Predict(image);
          
      using var graphics = Graphics.FromImage(image);

      foreach (var prediction in predictions) // iterate predictions to draw results
      {
          double score = Math.Round(prediction.Score, 2);

          graphics.DrawRectangles(new Pen(prediction.Label.Color, 1), new[] { prediction.Rectangle });

          var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

          graphics.DrawString($"{prediction.Label.Name} ({score})",
              new Font("Consolas", 24, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
              new PointF(x, y));
      }
      image.Save("Assets/result.jpg");
  }
```

该段程序
- 首先加载一张图片进行预测
- 通过`new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5s.onnx")`加载 `onnx` 文件
- 然后`scorer.Predict(image)`得到预测结果
- 接着遍历每一个预测结果，分别标出它们的预测框和标签
- 最后将预测结果保存成图片。
  
其中 `YoloCocoP5Model` 相当于模型数据集的配置文件，如果要换成我们自己的数据集，只需要复制一份，修改一些属性（`Dimensions` 和 `Labels`）即可，这一步工作将在后面的章节进行。

```CSharp
public class YoloCocoP5Model : YoloModel
{
    public override int Width { get; set; } = 640;
    public override int Height { get; set; } = 640;
    public override int Depth { get; set; } = 3;

    public override int Dimensions { get; set; } = 85;

    public override int[] Strides { get; set; } = new int[] { 8, 16, 32 };

    public override int[][][] Anchors { get; set; } = new int[][][]
    {
        new int[][] { new int[] { 010, 13 }, new int[] { 016, 030 }, new int[] { 033, 023 } },
        new int[][] { new int[] { 030, 61 }, new int[] { 062, 045 }, new int[] { 059, 119 } },
        new int[][] { new int[] { 116, 90 }, new int[] { 156, 198 }, new int[] { 373, 326 } }
    };

    public override int[] Shapes { get; set; } = new int[] { 80, 40, 20 };

    public override float Confidence { get; set; } = 0.20f;
    public override float MulConfidence { get; set; } = 0.25f;
    public override float Overlap { get; set; } = 0.45f;

    public override string[] Outputs { get; set; } = new[] { "output" };

    public override List<YoloLabel> Labels { get; set; } = new List<YoloLabel>()
    {
        new YoloLabel { Id = 1, Name = "person" },
        new YoloLabel { Id = 2, Name = "bicycle" },
        new YoloLabel { Id = 3, Name = "car" },
        new YoloLabel { Id = 4, Name = "motorcycle" },
        new YoloLabel { Id = 5, Name = "airplane" },
        new YoloLabel { Id = 6, Name = "bus" },
        new YoloLabel { Id = 7, Name = "train" },
        new YoloLabel { Id = 8, Name = "truck" },
        new YoloLabel { Id = 9, Name = "boat" },
        new YoloLabel { Id = 10, Name = "traffic light" },
        new YoloLabel { Id = 11, Name = "fire hydrant" },
        new YoloLabel { Id = 12, Name = "stop sign" },
        new YoloLabel { Id = 13, Name = "parking meter" },
        new YoloLabel { Id = 14, Name = "bench" },
        new YoloLabel { Id = 15, Name = "bird" },
        new YoloLabel { Id = 16, Name = "cat" },
        new YoloLabel { Id = 17, Name = "dog" },
        new YoloLabel { Id = 18, Name = "horse" },
        new YoloLabel { Id = 19, Name = "sheep" },
        new YoloLabel { Id = 20, Name = "cow" },
        new YoloLabel { Id = 21, Name = "elephant" },
        new YoloLabel { Id = 22, Name = "bear" },
        new YoloLabel { Id = 23, Name = "zebra" },
        new YoloLabel { Id = 24, Name = "giraffe" },
        new YoloLabel { Id = 25, Name = "backpack" },
        new YoloLabel { Id = 26, Name = "umbrella" },
        new YoloLabel { Id = 27, Name = "handbag" },
        new YoloLabel { Id = 28, Name = "tie" },
        new YoloLabel { Id = 29, Name = "suitcase" },
        new YoloLabel { Id = 30, Name = "frisbee" },
        new YoloLabel { Id = 31, Name = "skis" },
        new YoloLabel { Id = 32, Name = "snowboard" },
        new YoloLabel { Id = 33, Name = "sports ball" },
        new YoloLabel { Id = 34, Name = "kite" },
        new YoloLabel { Id = 35, Name = "baseball bat" },
        new YoloLabel { Id = 36, Name = "baseball glove" },
        new YoloLabel { Id = 37, Name = "skateboard" },
        new YoloLabel { Id = 38, Name = "surfboard" },
        new YoloLabel { Id = 39, Name = "tennis racket" },
        new YoloLabel { Id = 40, Name = "bottle" },
        new YoloLabel { Id = 41, Name = "wine glass" },
        new YoloLabel { Id = 42, Name = "cup" },
        new YoloLabel { Id = 43, Name = "fork" },
        new YoloLabel { Id = 44, Name = "knife" },
        new YoloLabel { Id = 45, Name = "spoon" },
        new YoloLabel { Id = 46, Name = "bowl" },
        new YoloLabel { Id = 47, Name = "banana" },
        new YoloLabel { Id = 48, Name = "apple" },
        new YoloLabel { Id = 49, Name = "sandwich" },
        new YoloLabel { Id = 50, Name = "orange" },
        new YoloLabel { Id = 51, Name = "broccoli" },
        new YoloLabel { Id = 52, Name = "carrot" },
        new YoloLabel { Id = 53, Name = "hot dog" },
        new YoloLabel { Id = 54, Name = "pizza" },
        new YoloLabel { Id = 55, Name = "donut" },
        new YoloLabel { Id = 56, Name = "cake" },
        new YoloLabel { Id = 57, Name = "chair" },
        new YoloLabel { Id = 58, Name = "couch" },
        new YoloLabel { Id = 59, Name = "potted plant" },
        new YoloLabel { Id = 60, Name = "bed" },
        new YoloLabel { Id = 61, Name = "dining table" },
        new YoloLabel { Id = 62, Name = "toilet" },
        new YoloLabel { Id = 63, Name = "tv" },
        new YoloLabel { Id = 64, Name = "laptop" },
        new YoloLabel { Id = 65, Name = "mouse" },
        new YoloLabel { Id = 66, Name = "remote" },
        new YoloLabel { Id = 67, Name = "keyboard" },
        new YoloLabel { Id = 68, Name = "cell phone" },
        new YoloLabel { Id = 69, Name = "microwave" },
        new YoloLabel { Id = 70, Name = "oven" },
        new YoloLabel { Id = 71, Name = "toaster" },
        new YoloLabel { Id = 72, Name = "sink" },
        new YoloLabel { Id = 73, Name = "refrigerator" },
        new YoloLabel { Id = 74, Name = "book" },
        new YoloLabel { Id = 75, Name = "clock" },
        new YoloLabel { Id = 76, Name = "vase" },
        new YoloLabel { Id = 77, Name = "scissors" },
        new YoloLabel { Id = 78, Name = "teddy bear" },
        new YoloLabel { Id = 79, Name = "hair drier" },
        new YoloLabel { Id = 80, Name = "toothbrush" }
    };

    public override bool UseDetect { get; set; } = true;

    public YoloCocoP5Model()
    {

    }
}
```

`YoloV5Net-App` 将设为启动项目

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-01-03.png)

我们不做修改，点击启动，运行结束后，在文件夹中输出了结果

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-04-15.png)

### 3.1.3. 性能测试

在 `YoloV5Net-App` 项目下的 `Program.cs` 中添加一些代码，检测100次，看看性能怎么样。

```CSharp
class Program
{
    static void Main(string[] args)
    {
        ...
        ...

        BenchMark();
    }

    private static void BenchMark()
    {
        var sw = new Stopwatch();
        using (var image = Image.FromFile("Assets/test.jpg"))
        using (var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5s.onnx"))
        {
            List<long> stats = new List<long>();

            for (int i = 0; i < 100; i++)
            {
                sw.Restart();
                scorer.Predict(image);
                long fps = 1000 / sw.ElapsedMilliseconds;
                stats.Add(fps);
                sw.Stop();
            }

            stats.Sort();
            Console.WriteLine($@"
                Max FPS: {stats[stats.Count - 1]}
                Avg FPS: {Avg(stats)}
                Min FPS: {stats[0]}
            ");
        }
    }
    private static int Avg(List<long> stats)
    {
        long sum = 0;
        foreach (long i in stats)
        {
            sum += i;
        }
        return (int)(sum / stats.Count);
    }
}
```

点击启动，结果

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-10-56.png)

平均帧率只有5FPS，这是因为目前是使用CPU计算的，没有使用CUDA加速

对 `BenchMark()` 函数的代码稍作修改，以启用显卡设备参与计算

```CSharp
private static void BenchMark()
{
    ...

    SessionOptions sessionOptions = new SessionOptions();
    sessionOptions.AppendExecutionProvider_CUDA();

    using (var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5s.onnx", sessionOptions))
    {
        ...
    }
}
```

重新运行程序，结果如下

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-22-36.png)

帧率有所提升，但是还是不够快，这是由于在检测时大量的浮点运算耗费了时间，我们暂时先不做这部分的优化，继续部署我们的项目。

### 3.1.4. 导出 dll

选中 `Yolov5Net.Scorer` 右键重新生成

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-24-18.png)

```
已启动重新生成…
1>------ 已启动全部重新生成: 项目: Yolov5Net.Scorer, 配置: Debug Any CPU ------
已还原 C:\Users\user\Documents\MyProjects\DeepLearning\yolov5-net\src\Yolov5Net.App\Yolov5Net.App.csproj (用时 32 ms)。
已还原 C:\Users\user\Documents\MyProjects\DeepLearning\yolov5-net\src\Yolov5Net.Scorer\Yolov5Net.Scorer.csproj (用时 34 ms)。
1>Yolov5Net.Scorer -> C:\Users\user\Documents\MyProjects\DeepLearning\yolov5-net\src\Yolov5Net.Scorer\bin\Debug\netstandard2.0\Yolov5Net.Scorer.dll
========== 全部重新生成: 成功 1 个，失败 0 个，跳过 0 个 ==========
```

找到生成的 `Yolov5Net.Scorer.dll`，之后我们会将其导入到自己的目标检测项目。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-26-35.png)

### 3.1.5. 添加引用和模型文件

使用VS新建一个桌面程序项目作为我们的目标检测项目（Winform或WPF，本文使用WPF），添加引用，点击浏览，找到我们之前生成的 `Yolov5Net.Scorer.dll`

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-34-03.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-35-02.png)

添加文件夹如下

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-40-46.png)

其中
- `Models` 用来存放之前提到的模型数据集配置类（如`YoloCocoP5Model.cs`）
- `Weights` 用来存放模型权重文件（如 `.onnx` 文件）

将我们之前输出的`.onnx` 文件添加的文件夹`Weights`中，并重命名一个合适的名称。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-16-15-49-47.png)

在 `Models` 文件夹中新建 `YoloV5AmpouleModel.cs`，其内容如下

```CSharp
using System.Collections.Generic;
using System.Drawing;

using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models.Abstract;

namespace RP_YOLO.YOLO.Models
{
    internal class YoloV5AmpouleModel : YoloModel
    {
        public override int Width { get; set; } = 640;
        public override int Height { get; set; } = 640;
        public override int Depth { get; set; } = 3;

        public override int Dimensions { get; set; } = 7; // = 分类数 + 5

        public override int[] Strides { get; set; } = new int[] { 8, 16, 32 };

        public override int[][][] Anchors { get; set; } = new int[][][]
        {
            new int[][] { new int[] { 010, 13 }, new int[] { 016, 030 }, new int[] { 033, 023 } },
            new int[][] { new int[] { 030, 61 }, new int[] { 062, 045 }, new int[] { 059, 119 } },
            new int[][] { new int[] { 116, 90 }, new int[] { 156, 198 }, new int[] { 373, 326 } }
        };

        public override int[] Shapes { get; set; } = new int[] { 80, 40, 20 };

        public override float Confidence { get; set; } = 0.20f;
        public override float MulConfidence { get; set; } = 0.25f;
        public override float Overlap { get; set; } = 0.45f;

        public override string[] Outputs { get; set; } = new[] { "output" };

        public override List<YoloLabel> Labels { get; set; } = new List<YoloLabel>()
        {
            new YoloLabel { Id = 1, Name = "OK" , Color = Color.Green},
            new YoloLabel { Id = 2, Name = "NG" , Color = Color.Red}
        };

        public override bool UseDetect { get; set; } = true;

        public YoloV5AmpouleModel()
        {

        }
    }
}
```

其中
- `Dimensions` 的值 = 种类数 + 5，我们的数据集一共两个标签，因此 `Dimensions = 7`
- `Labels` 依次列出所有标签即可，这里我们将 OK 设置为绿色， NG 显示为红色。

**注意：**
如果是新版本 YOLO V5 的检测模型，最后的输出节点的名称为 `output0` 而不是 `output`，
上面代码的 `Outputs` 属性要做相应的更改：`public override string[] Outputs { get; set; } = new[] { "output0" };`。可以通过查看 YOLO V5 的源代码 `export.py`的此行代码确认这一点：

```Python
output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
```

也可以在一些能够查看`onnx`模型结构的网站（如[Netron](https://netron.app/)）查看导出的模型最后输出节点的名称。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-21-10-14-49.png)

## 3.2. 单张图片输入

我们现在需要一个界面，它具有如下功能

- 浏览选择`onnx`文件
- 加载并显示一张图片
- 运行目标检测并将结果显示到界面

UI界面原型设计如下

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-19-10-41-56.png)

用 WPF 实现界面如下：

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-21-17-05-33.png)

加载图片与onnx文件

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-21-17-08-18.png)

点击运行，执行目标检测算法

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-21-17-09-37.png)

点击运行时，执行的主要代码如下

```Csharp

private void btn_run_Click(object sender, RoutedEventArgs e)
{
    if (_onnxPath == null)
    {
        System.Windows.MessageBox.Show("请先选择onnx文件");
        return;
    }

    if (!_isRunning)
    {
        _isRunning = true;
        btn_run.Content = "停止";

        RunDetect();
    }
    else
    {
        _isRunning = false;
        btn_run.Content = "运行";
    }            
}

private void RunDetect()
{
    try
    {
        System.Drawing.Image image = System.Drawing.Image.FromFile(_originImagePath);

        ObjectDetect(image);

        uct_detectedImage.ShowImage(image);

    }
    catch (Exception e)
    {
        if (e.GetType() == typeof(System.IO.FileNotFoundException))
        {
            System.Windows.MessageBox.Show("请选择源文件");
        }
    }
}

private void ObjectDetect(System.Drawing.Image image)
{
    var scorer = new YoloScorer<YOLO.Models.YoloV5AmpouleModel>(_onnxPath);

    List<YoloPrediction> predictions = scorer.Predict(image);

    var graphics = Graphics.FromImage(image);

    foreach (var prediction in predictions) // iterate predictions to draw results
    {
        double score = Math.Round(prediction.Score, 2);

        graphics.DrawRectangles(new System.Drawing.Pen(prediction.Label.Color, 2), new[] { prediction.Rectangle });

        var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

        graphics.DrawString($"{prediction.Label.Name} ({score})",
            new Font("Consolas", 24, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
            new PointF(x, y));
    }
}
```

其中 `ObjectDetect()` 就是调用之前生成的 `Yolov5Net.Scorer.dll` 的相关方法进行目标检测和检测框的可视化，其代码和之前[demo](#使用-yolov5-net-进行目标检测)中的致相同。

左右两部分显示图片，显示区域有点小，也没必要，调整为一个显示区域显示图片，并增加检测识别时间的统计。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-22-16-49-56.png)

至此，YOLO V5 的部署工作已经基本完成。

但是对单张图片的检测并不能满足我们实际项目需求，接下来我们将探索如何使用相机作为输入进行目标检测。

## 3.3. 相机输入

### 3.3.1. MVS例程

安装海康威视的软件 MVS，打开安装目录中官方自带的样例工程。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-26-11-46-08.png)

其中 `BasicDemo` 项目包括了相机查找、连接、图片采集和参数设置等，基本满足我们的需求。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-26-11-55-51.png)

下面是几段关键代码。

点击查找设备按钮：
- 查找并列出相机设备

```CSharp
private void DeviceListAcq()
{
    // ch:创建设备列表 | en:Create Device List
    System.GC.Collect();
    cbDeviceList.Items.Clear();
    m_stDeviceList.nDeviceNum = 0;
    int nRet = MyCamera.MV_CC_EnumDevices_NET(MyCamera.MV_GIGE_DEVICE | MyCamera.MV_USB_DEVICE, ref m_stDeviceList);
    if (0 != nRet)
    {
        ShowErrorMsg("Enumerate devices fail!",0);
        return;
    }

    // ch:在窗体列表中显示设备名 | en:Display device name in the form list
    for (int i = 0; i < m_stDeviceList.nDeviceNum; i++)
    {
        MyCamera.MV_CC_DEVICE_INFO device = (MyCamera.MV_CC_DEVICE_INFO)Marshal.PtrToStructure(m_stDeviceList.pDeviceInfo[i], typeof(MyCamera.MV_CC_DEVICE_INFO));
        if (device.nTLayerType == MyCamera.MV_GIGE_DEVICE)
        {
            MyCamera.MV_GIGE_DEVICE_INFO gigeInfo = (MyCamera.MV_GIGE_DEVICE_INFO)MyCamera.ByteToStruct(device.SpecialInfo.stGigEInfo, typeof(MyCamera.MV_GIGE_DEVICE_INFO));
            
            if (gigeInfo.chUserDefinedName != "")
            {
                cbDeviceList.Items.Add("GEV: " + gigeInfo.chUserDefinedName + " (" + gigeInfo.chSerialNumber + ")");
            }
            else
            {
                cbDeviceList.Items.Add("GEV: " + gigeInfo.chManufacturerName + " " + gigeInfo.chModelName + " (" + gigeInfo.chSerialNumber + ")");
            }
        }
        else if (device.nTLayerType == MyCamera.MV_USB_DEVICE)
        {
            MyCamera.MV_USB3_DEVICE_INFO usbInfo = (MyCamera.MV_USB3_DEVICE_INFO)MyCamera.ByteToStruct(device.SpecialInfo.stUsb3VInfo, typeof(MyCamera.MV_USB3_DEVICE_INFO));
            if (usbInfo.chUserDefinedName != "")
            {
                cbDeviceList.Items.Add("U3V: " + usbInfo.chUserDefinedName + " (" + usbInfo.chSerialNumber + ")");
            }
            else
            {
                cbDeviceList.Items.Add("U3V: " + usbInfo.chManufacturerName + " " + usbInfo.chModelName + " (" + usbInfo.chSerialNumber + ")");
            }
        }
    }

    // ch:选择第一项 | en:Select the first item
    if (m_stDeviceList.nDeviceNum != 0)
    {
        cbDeviceList.SelectedIndex = 0;
    }
}
```

点击打开设备按钮：
- 打开相机
- 设置采集模式、参数

```CSharp
private void bnOpen_Click(object sender, EventArgs e)
{
    if (m_stDeviceList.nDeviceNum == 0 || cbDeviceList.SelectedIndex == -1)
    {
        ShowErrorMsg("No device, please select", 0);
        return;
    }

    // ch:获取选择的设备信息 | en:Get selected device information
    MyCamera.MV_CC_DEVICE_INFO device = 
        (MyCamera.MV_CC_DEVICE_INFO)Marshal.PtrToStructure(m_stDeviceList.pDeviceInfo[cbDeviceList.SelectedIndex],
                                                        typeof(MyCamera.MV_CC_DEVICE_INFO));

    // ch:打开设备 | en:Open device
    if (null == m_MyCamera)
    {
        m_MyCamera = new MyCamera();
        if (null == m_MyCamera)
        {
            return;
        }
    }

    int nRet = m_MyCamera.MV_CC_CreateDevice_NET(ref device);
    if (MyCamera.MV_OK != nRet)
    {
        return;
    }

    nRet = m_MyCamera.MV_CC_OpenDevice_NET();
    if (MyCamera.MV_OK != nRet)
    {
        m_MyCamera.MV_CC_DestroyDevice_NET();
        ShowErrorMsg("Device open fail!", nRet);
        return;
    }

    // ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if (device.nTLayerType == MyCamera.MV_GIGE_DEVICE)
    {
        int nPacketSize = m_MyCamera.MV_CC_GetOptimalPacketSize_NET();
        if (nPacketSize > 0)
        {
            nRet = m_MyCamera.MV_CC_SetIntValue_NET("GevSCPSPacketSize", (uint)nPacketSize);
            if (nRet != MyCamera.MV_OK)
            {
                ShowErrorMsg("Set Packet Size failed!", nRet);
            }
        }
        else
        {
            ShowErrorMsg("Get Packet Size failed!", nPacketSize);
        }
    }

    // ch:设置采集连续模式 | en:Set Continues Aquisition Mode
    m_MyCamera.MV_CC_SetEnumValue_NET("AcquisitionMode", (uint)MyCamera.MV_CAM_ACQUISITION_MODE.MV_ACQ_MODE_CONTINUOUS);
    m_MyCamera.MV_CC_SetEnumValue_NET("TriggerMode", (uint)MyCamera.MV_CAM_TRIGGER_MODE.MV_TRIGGER_MODE_OFF);

    bnGetParam_Click(null, null);// ch:获取参数 | en:Get parameters

    // ch:控件操作 | en:Control operation
    SetCtrlWhenOpen();
}

private void bnGetParam_Click(object sender, EventArgs e)
{
    MyCamera.MVCC_FLOATVALUE stParam = new MyCamera.MVCC_FLOATVALUE();
    int nRet = m_MyCamera.MV_CC_GetFloatValue_NET("ExposureTime", ref stParam);
    if (MyCamera.MV_OK == nRet)
    {
        tbExposure.Text = stParam.fCurValue.ToString("F1");
    }

    nRet = m_MyCamera.MV_CC_GetFloatValue_NET("Gain", ref stParam);
    if (MyCamera.MV_OK == nRet)
    {
        tbGain.Text = stParam.fCurValue.ToString("F1");
    }

    nRet = m_MyCamera.MV_CC_GetFloatValue_NET("ResultingFrameRate", ref stParam);
    if (MyCamera.MV_OK == nRet)
    {
        tbFrameRate.Text = stParam.fCurValue.ToString("F1");
    }
}

private void SetCtrlWhenOpen()
{
    bnOpen.Enabled = false;

    bnClose.Enabled = true;
    bnStartGrab.Enabled = true;
    bnStopGrab.Enabled = false;
    bnContinuesMode.Enabled = true;
    bnContinuesMode.Checked = true;
    bnTriggerMode.Enabled = true;
    cbSoftTrigger.Enabled = false;
    bnTriggerExec.Enabled = false;

    tbExposure.Enabled = true;
    tbGain.Enabled = true;
    tbFrameRate.Enabled = true;
    bnGetParam.Enabled = true;
    bnSetParam.Enabled = true;
}
```

点击开始采集按钮：
- 启动新线程接受流
  - 获取图像缓存
  - 显示图像
- 开始采集

```CSharp
private void bnStartGrab_Click(object sender, EventArgs e)
{
    // ch:标志位置位true | en:Set position bit true
    m_bGrabbing = true;

    m_hReceiveThread = new Thread(ReceiveThreadProcess);
    m_hReceiveThread.Start();

    m_stFrameInfo.nFrameLen = 0;//取流之前先清除帧长度
    m_stFrameInfo.enPixelType = MyCamera.MvGvspPixelType.PixelType_Gvsp_Undefined;
    // ch:开始采集 | en:Start Grabbing
    int nRet = m_MyCamera.MV_CC_StartGrabbing_NET();
    if (MyCamera.MV_OK != nRet)
    {
        m_bGrabbing = false;
        m_hReceiveThread.Join();
        ShowErrorMsg("Start Grabbing Fail!", nRet);
        return;
    }

    // ch:控件操作 | en:Control Operation
    SetCtrlWhenStartGrab();
}


public void ReceiveThreadProcess()
{
    MyCamera.MV_FRAME_OUT stFrameInfo = new MyCamera.MV_FRAME_OUT();
    MyCamera.MV_DISPLAY_FRAME_INFO stDisplayInfo = new MyCamera.MV_DISPLAY_FRAME_INFO();
    int nRet = MyCamera.MV_OK;

    while (m_bGrabbing)
    {
        nRet = m_MyCamera.MV_CC_GetImageBuffer_NET(ref stFrameInfo, 1000);
        if (nRet == MyCamera.MV_OK)
        {
            lock (BufForDriverLock)
            {
                if (m_BufForDriver == IntPtr.Zero || stFrameInfo.stFrameInfo.nFrameLen > m_nBufSizeForDriver)
                {
                    if (m_BufForDriver != IntPtr.Zero)
                    {
                        Marshal.Release(m_BufForDriver);
                        m_BufForDriver = IntPtr.Zero;
                    }

                    m_BufForDriver = Marshal.AllocHGlobal((Int32)stFrameInfo.stFrameInfo.nFrameLen);
                    if (m_BufForDriver == IntPtr.Zero)
                    {
                        return;
                    }
                    m_nBufSizeForDriver = stFrameInfo.stFrameInfo.nFrameLen;
                }

                m_stFrameInfo = stFrameInfo.stFrameInfo;
                CopyMemory(m_BufForDriver, stFrameInfo.pBufAddr, stFrameInfo.stFrameInfo.nFrameLen);
            }

            if (RemoveCustomPixelFormats(stFrameInfo.stFrameInfo.enPixelType))
            {
                m_MyCamera.MV_CC_FreeImageBuffer_NET(ref stFrameInfo);
                continue;
            }
            stDisplayInfo.hWnd = pictureBox1.Handle;
            stDisplayInfo.pData = stFrameInfo.pBufAddr;
            stDisplayInfo.nDataLen = stFrameInfo.stFrameInfo.nFrameLen;
            stDisplayInfo.nWidth = stFrameInfo.stFrameInfo.nWidth;
            stDisplayInfo.nHeight = stFrameInfo.stFrameInfo.nHeight;
            stDisplayInfo.enPixelType = stFrameInfo.stFrameInfo.enPixelType;
            m_MyCamera.MV_CC_DisplayOneFrame_NET(ref stDisplayInfo);

            m_MyCamera.MV_CC_FreeImageBuffer_NET(ref stFrameInfo);
        }
        else
        {
            if (bnTriggerMode.Checked)
            {
                Thread.Sleep(5);
            }
        }
    }
}

private void SetCtrlWhenStartGrab()
{
    bnStartGrab.Enabled = false;
    bnStopGrab.Enabled = true;

    if (bnTriggerMode.Checked && cbSoftTrigger.Checked)
    {
        bnTriggerExec.Enabled = true;
    }

    bnSaveBmp.Enabled = true;
    bnSaveJpg.Enabled = true;
    bnSaveTiff.Enabled = true;
    bnSavePng.Enabled = true;
}
```

海康相机SDK取图流程图：

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-29-13-15-15.png)

### 3.3.2. 实现

将`MvCameraControl.Net.dll`添加到我们的项目中

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-27-11-32-49.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-27-11-36-12.png)

新建一个页面

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-30-17-00-53.png)

它与之前的窗口类似，只不过左侧栏由原来的图片浏览列表变成了相机控制台。

`查找相机`、`连接`、`断开`、`开始采集`和`停止采集`按钮的点击事件直接复制海康 SDK Demo 中的代码即可。

在获取图像的线程处理函数中，加上我们的目标检测算法和图片显示的代码。

```CSharp
public void ReceiveThreadProcess()
{
    //获取 Payload Size
    MyCamera.MVCC_INTVALUE stParam = new MyCamera.MVCC_INTVALUE();
    int nRet = m_MyCamera.MV_CC_GetIntValue_NET("PayloadSize", ref stParam) ;

    if (nRet != MyCamera.MV_OK)
    {
        ShowErrorMsg("获取负载大小失败。", nRet);
        return;
    }
    uint nPayloadSize = stParam.nCurValue;
    //如果负载更大，重新分配缓存
    if (nPayloadSize > m_nBufSizeForDriver)
    {
        if (m_BufForFrame != IntPtr.Zero)
            Marshal.Release(m_BufForFrame);
        m_nBufSizeForDriver = nPayloadSize;
        m_BufForFrame = Marshal.AllocHGlobal((int)m_nBufSizeForDriver);
    }
    if (m_BufForFrame == IntPtr.Zero)
    {
        return;
    }

    MyCamera.MV_FRAME_OUT_INFO_EX stFrameInfo = new MyCamera.MV_FRAME_OUT_INFO_EX();

    while (m_bGrabbing)
    {
        lock (BufForDriverLock)
        {
            //获取单帧数据
            nRet = m_MyCamera.MV_CC_GetOneFrameTimeout_NET(m_BufForFrame, nPayloadSize, ref stFrameInfo, 1000);
            if (nRet == MyCamera.MV_OK)
            {
                m_stFrameInfo = stFrameInfo;
            }
        }

        if (nRet == MyCamera.MV_OK)
        {
            if (RemoveCustomPixelFormats(stFrameInfo.enPixelType))
            {
                continue;
            }
            //创建位图，用来存放帧数据
            Bitmap bitmap = new Bitmap(m_stFrameInfo.nWidth, m_stFrameInfo.nHeight, PixelFormat.Format24bppRgb);
            Rectangle rect = new Rectangle(0, 0, m_stFrameInfo.nWidth, m_stFrameInfo.nHeight);
            BitmapData bitmapData = bitmap.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
            unsafe
            {
                //拷贝帧数据
                Buffer.MemoryCopy(m_BufForFrame.ToPointer(), bitmapData.Scan0.ToPointer(), m_nBufSizeForDriver, m_nBufSizeForDriver);
            }
            bitmap.UnlockBits(bitmapData);

            if (m_isRunning)
            {
                //执行目标检测算法
                RunDetect(bitmap);
            }
            Dispatcher.Invoke(new Action(delegate
            {
                //显示图片
                uct_image.ShowImage(bitmap);
            }));
        }
    }
}

private void RunDetect(Bitmap bitmap)
{
    try
    {
        yolov5.ObjectDetect(bitmap, out DetectResult result);
        Dispatcher.Invoke(new Action(delegate {
            if (detectResults.Count == 0)
            {
                detectResults.Add(new DetectResult());
            }
            detectResults.RemoveAt(detectResults.Count - 1);
            detectResults.Add(result);
        }));
    }
    catch (Exception e)
    {
        if (e.GetType() == typeof(System.IO.FileNotFoundException))
        {
            System.Windows.MessageBox.Show("请选择源文件");
        }
    }
}
```

在上述代码中，主要是使用 `MV_CC_GetOneFrameTimeout_NET()` 获取一帧数据，存放在缓存 `m_BufForFrame` 中，然后把数据拷贝到位图的像素数据内存 `bitmapData.Scan0` 中，最后传入到我们的目标检测函数去处理图片。

> 要注意的是，位图的像素格式 `PixelFormat` 需要根据相机像素格式的设置而改变，才能正确地解析相机采集的图像。

运行查看结果

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-10-10-31-59.png)

发现图像的颜色不太对，红色安瓿瓶显示成了蓝色，这是像素色彩顺序不同导致的，打开 MVS 将相机的像素格式改为 `BGR 8`。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-10-10-34-54.png)

运行目标检测程序，一切正常

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-10-10-40-02.png)


为了方便后期的调用，我们把 YOLO V5 的相关代码封装成 `YOLOV5` 类：

```CSharp
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using RP_YOLO.Model;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models.Abstract;

namespace RP_YOLO.YOLO
{
    /// <summary>
    /// YOLOV5 封装类
    /// </summary>
    class YOLOV5<T> where T : YoloModel
    {
        private YoloScorer<T> m_scorer;

        public YOLOV5(string onnxPath)
        {
            //使用CUDA
            SessionOptions sessionOptions = new SessionOptions();
            sessionOptions.AppendExecutionProvider_CUDA();
            //加载模型文件
            m_scorer = new YoloScorer<T>(onnxPath, sessionOptions);
        }

        /// <summary>
        /// 目标检测
        /// </summary>
        /// <param name="image">输出图片</param>
        /// <param name="quantity">数组 依次存放各个种类的数量</param>
        /// <param name="during">检测所用时间</param>
        public void ObjectDetect(System.Drawing.Image image, out DetectResult result)
        {
            result = new DetectResult();

            Stopwatch stopwatch = new Stopwatch();//计时器用来计算目标检测算法执行时间
            stopwatch.Start();
            List<YoloPrediction> predictions = m_scorer.Predict(image);
            stopwatch.Stop();
            result.during = stopwatch.ElapsedMilliseconds;

            var graphics = Graphics.FromImage(image);

            // 遍历预测结果，画出预测框
            foreach (var prediction in predictions)
            {
                double score = Math.Round(prediction.Score, 2);

                graphics.DrawRectangles(new System.Drawing.Pen(prediction.Label.Color, 2), new[] { prediction.Rectangle });

                var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

                graphics.DrawString($"{prediction.Label.Name} ({score})",
                    new Font("Consolas", 24, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color), new PointF(x, y));

                switch (prediction.Label.Id)
                {
                    case 0:
                        result.OK++;
                        break;
                    case 1:
                        result.NG++;
                        break;
                }
            }
        }
    }
}
```

最后，我们部署项目的结构如下

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-10-10-44-35.png)

## 3.4. 示例

最后，我们再通过一个示例，完整地回顾一下部署的流程。由于我们之前已经做好了大量工作，所以将我们的工程应用于新项目时一般只需要稍作修改即可。

接下来我们将使用 YOLO V5 来检测螺柱。

### 3.4.1. 数据标注

我们在 YOLO V5 项目中新建一些文件夹，用来存放我们的数据集。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-10-00-33.png)

将拍摄好的螺栓图片存放到 `Image` 文件夹中

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-10-01-32.png)

使用 `LableImg` 软件进行数据标注，输出文件夹选择 `Annotations` 文件夹。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-09-58-43.png)

### 3.4.2. 训练

#### 3.4.2.1. 数据集预处理

打开 `xml2yolo.py`

更改分类列表，并将数据集的文件夹改为当前数据集的文件夹名。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-11-43-46.png)

运行 `xml2yolo.py` 将数据集标注文件格式转为 `YOLO` 的格式，并划分数据集。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-11-20-11.png)

在 `data` 目录下新建一个 `Solder.yaml` 文件

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-11-46-46.png)

`Solder.yaml` 与之前的 `Ampoule.yaml` 基本一致，只更改一下分类的名称列表即可：

```yaml
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: datasets
train: images/train
val: images/val

# Classes
names:
  0: Bolt
```

#### 3.4.2.2. 模型训练与输出

配置 `train.py` 的运行参数，指定权重文件和数据集的 `yaml` 文件

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-11-52-20.png)

运行 `train.py` 开始训练

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-11-56-59.png)

训练完成

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-13-43-20.png)

打开 `export.py` 配置参数并运行。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-14-56-46.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-14-57-16.png)

### 3.4.3. 部署

将上一步输出得到的 `best.onnx` 文件拷贝到部署项目的 YOLO 模型文件夹下，并重命名为 `yolov5_solder.onnx`

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-14-59-47.png)

添加到工程中

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-15-01-56.png)

在 `YOLO/Models` 文件夹下新建 `YoloV5SolderModel.cs`

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-15-05-26.png)

`YoloV5SolderModel.cs` 相比于之前的 `YoloV5AmpouleModel.cs` 只需要作两处修改：检测类别数量和类别标签列表

```CSharp
using System.Collections.Generic;
using System.Drawing;

using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models.Abstract;

namespace RP_YOLO.YOLO.Models
{
    internal class YoloV5SolderModel : YoloModel
    {
        public static int classCount = 1;

        ...

        public override List<YoloLabel> Labels { get; set; } = new List<YoloLabel>()
        {
            new YoloLabel { Id = 0, Name = "Bolt" , Color = Color.Green}
        };

        ...
    }
}
```

最后在 `Window_SingleImageDetect.xaml.cs` 中将 `yolov5` 对象的泛型类型改为 `YoloV5SolderModel` 即可：

```CSharp
namespace RP_YOLO.View
{
    /// <summary>
    /// Window_SingleImageDetect.xaml 的交互逻辑
    /// </summary>
    public partial class Window_CameraStreamDetect : Window
    {
        ...

        YOLOV5<YoloV5SolderModel> yolov5;

        ...

        private void btn_browse_modelFile_Click(object sender, RoutedEventArgs e)
        {
            ...
            if (result == System.Windows.Forms.DialogResult.OK)
            {
                string onnxPath = tbx_modelFile.Text = openFileDialog.FileName;
                yolov5 = new YOLOV5<YoloV5SolderModel>(onnxPath);
            }
        }
    }
}
```

### 3.4.4. 运行

运行程序查看目标检测效果：

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-12-16-47-59.png)


至此，YOLO V5 部署的主要工作就完成了，接下来是一些调优工作，这也是我们的程序能够应用于实际项目的关键。在对目标检测程序进行优化之前，我们先来了解 YOLO V5 的模型结构和预测过程。

# 4. YOLO V5 详解

## 4.1. 网络结构

![网络结构](imgs/YOLO%20V5%20部署笔记.md/yolov5s.onnx.svg)



![](imgs/YOLO%20V5%20部署笔记.md/2022-10-14-13-20-02.png)

## 4.2. 预测过程

## 4.3. 训练参数



# 5. 优化

我们的程序最主要的部分时采集图像和处理图像，也就是 `ReceiveThreadProcess()` 中 `while` 循环中所作的工作，我们看看是否可以优化其中的操纵来提高程序性能。

首先能想到的时提高目标检测算法运算速度。

计算每轮`while`循环的执行时间：

```CSharp
public void ReceiveThreadProcess()
{
    ...

    while (m_bGrabbing)
    {
        m_stopwatch.Restart();//计时

        ...
        ...
        ...

        m_stopwatch.Stop();
        Debug.WriteLine(m_stopwatch.ElapsedMilliseconds);
    }
}
```

结果

> ...
47
43
41
41
42
41
46
44
34
43
41
////// 开始运行目标检测
961
86
91
82
87
88
83
466
96
75
82
93
88
113
108
84
77
96
84
...

观察结果可以发现，未运行目标检测时，用时45ms左右，运行目标检测后，第一帧耗时较长，后面基本在80-100ms左右，在此前，因此还有很大的优化空间。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-14-16-28-27.png)


## 5.1. 浮点运算速度优化

OpenCL

## 5.2. 内存优化

# 6. 后期开发

## 6.1. 模型参数可调

如下图所示，在有些时候，一些目标并不是我们想检测的对象，但由于与目标相似度比较高或模型训练得不够好等原因，这些物体也会被检测出来，但置信度比较低。所以我们希望通过提高置信度的阈值参数将这些误识别的物体过滤掉。

![](imgs/YOLO%20V5%20部署笔记.md/2022-10-21-09-26-13.png)

模型的置信度等参数在 `YOLOV5***Model.cs` 文件中定义，在本例中为 `YOLOV5SolderModel.cs`。

```CSharp
using System.Collections.Generic;
using System.Drawing;

using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models.Abstract;

namespace RP_YOLO.YOLO.Models
{
    internal class YoloV5SolderModel : YoloModel
    {
        ...

        public override float Confidence { get; set; } = 0.20f;
        public override float MulConfidence { get; set; } = 0.25f;
        public override float Overlap { get; set; } = 0.45f;

        ...

    }
}
```

其中

- `Confidence` 表示“图像中包含物体的概率（称为 `Object Confidence`）”阈值，
- `MulConfidence` 表示“检测到的目标是某一类别的概率（叫做 `Multiple Confidence`）”阈值

我们在 `YoloScorer.cs` 中对检测结果进行解析的函数中可以清楚地看到这两个参数的作用

```CSharp
/// <summary>
/// Parses net output (detect) to predictions.
/// </summary>
private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
{
    ...

    Parallel.For(0, (int)output.Length / _model.Dimensions, (i) =>
    {
        if (output[0, i, 4] <= _model.Confidence) return; // skip low obj_conf results

        for (int j = 5; j < _model.Dimensions; j++)
        {
            output[0, i, j] = output[0, i, j] * output[0, i, 4]; // mul_conf = obj_conf * cls_conf

            if (output[0, i, j] <= _model.MulConfidence) continue; // skip low mul_conf results

            ...
        }
    });

    ...
}
```

当预测结果的 `Object Confidence` 或 `Multiple Confidence` 小于等于规定阈值时，会直接跳过该结果不做处理。 

### 6.1.1. 实现模型参数可编辑

要实现模型参数在用户界面可调，我们需要根据用户选择，显示当前模型的参数，并根据用户输入改变模型参数数值，也就是要实现数据绑定。
由于我们定义了多种模型，用更改代码的方式更改模型显然不可取，所以我们添加一个列表用来选择模型模板，并显示参数的数值。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-01-10-32-50.png)

当前我们使用泛型来初始化模型参数，这样很难根据用户选择动态绑定数据，所以我们将之前的代码进行改写。

不再使用泛型的方式传递模型参数，而是将模型作为对象，在构造函数中初始化。

```CSharp
/// <summary>
/// Creates new instance of YoloScorer with weights path and options.
/// </summary>
public YoloScorer(YoloModel yoloModel, string weights, SessionOptions opts = null)
{
    _model = yoloModel;
    oclCaller = new OclCaller();
    oclCaller.Init();
    
    _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
}
```

`YOLOV5`封装类中进行相应的改写。

```CSharp
public YOLOV5(YoloModel yoloModel, string onnxPath)
{
    //使用CUDA
    SessionOptions sessionOptions = new SessionOptions();
    sessionOptions.AppendExecutionProvider_CUDA();
    //加载模型文件
    scorer = new YoloScorer(yoloModel, onnxPath, sessionOptions);
}
```

将一些固定的参数，在模型模板的抽象类 `YoloModel` 中就赋好值。

```CSharp
using System.Collections.Generic;
using System.ComponentModel;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Model descriptor.
    /// </summary>
    public abstract class YoloModel : INotifyPropertyChanged
    {
        public int Width { get; set; } = 640;
        public int Height { get; set; } = 640;
        public int Depth { get; set; } = 3;

        public int Dimensions { get; set; }

        public int[] Strides { get; set; } = new int[] { 8, 16, 32 };
        public int[][][] Anchors { get; set; } = new int[][][]
            {
                new int[][] { new int[] { 010, 13 }, new int[] { 016, 030 }, new int[] { 033, 023 } },
                new int[][] { new int[] { 030, 61 }, new int[] { 062, 045 }, new int[] { 059, 119 } },
                new int[][] { new int[] { 116, 90 }, new int[] { 156, 198 }, new int[] { 373, 326 } }
            };
        public int[] Shapes { get; set; } = new int[] { 80, 40, 20 };

        ...

        public string[] Outputs { get; set; } = new[] { "output" };
        public List<YoloLabel> Labels { get; set; }
        public bool UseDetect { get; set; } = true;
    }
}
```

对于可调的参数，我们实现 `INotifyPropertyChanged` 接口为数据绑定做准备。

```CSharp
using System.Collections.Generic;
using System.ComponentModel;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Model descriptor.
    /// </summary>
    public abstract class YoloModel : INotifyPropertyChanged
    {
        ...

        public float Confidence
        {
            get => _Confidence;
            set
            {
                if (Confidence != value)
                {
                    _Confidence = value;
                    OnPropertyChanged(nameof(Confidence));
                }
            }
        }
        public float MulConfidence
        {
            get => _MulConfidence;
            set
            {
                if (MulConfidence != value)
                {
                    _MulConfidence = value;
                    OnPropertyChanged(nameof(MulConfidence));
                }
            }
        }
        public float Overlap
        {
            get => _Overlap;
            set
            {
                if (Overlap != value)
                {
                    _Overlap = value;
                    OnPropertyChanged(nameof(Overlap));
                }
            }
        }

        public int MaxDetections
        {
            get => _MaxDetections;
            set
            {
                if (MaxDetections != value)
                {
                    _MaxDetections = value;
                    OnPropertyChanged(nameof(MaxDetections));
                }
            }
        }
        ...

        public event PropertyChangedEventHandler PropertyChanged;
        private void OnPropertyChanged(string propertyName)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private float _Confidence;
        private float _MulConfidence;
        private float _Overlap;
        private int _MaxDetections;
    }
}
```

对于具体的模型模板，在构造函数中初始化参数。

```CSharp
namespace RP_YOLO.YOLO.Models
{
    internal class YoloV5OkNgModel : YoloModel
    {
        public static int classCount = 2;

        public YoloV5OkNgModel()
        {
            Dimensions = classCount + 5; // = 分类数 + 5

            Confidence = 0.20f;
            MulConfidence = 0.25f;
            Overlap = 0.45f;
            MaxDetections = 10;

            Labels = new List<YoloLabel>()
            {
                new YoloLabel { Id = 0, Name = "OK" , Color = Color.Green},
                new YoloLabel { Id = 1, Name = "NG" , Color = Color.Red}
            };
        }
    }
}
```

这样一来，我们只定义一个`YoloModel`对象即可

```CSharp
    private YoloModel m_yolov5Model;
    private void cbb_modelType_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        // OKNG 模型
        if (cbbi_modelType_OKNG.IsSelected)
        {
            m_yolov5Model = new YoloV5OkNgModel();
        }
        // 螺柱 模型
        else if (cbbi_modelType_bolt.IsSelected)
        {
            m_yolov5Model = new YoloV5SolderModel();
        }

        // 绑定上下文
        sp_modelParam.DataContext = m_yolov5Model;
    }
```

在`XAML`文件中指定绑定变量名、绑定模式和源更新的时机。

```XML
<StackPanel x:Name="sp_modelParam">
    <DockPanel>
        <Label Content="Confidence" Margin="5,1,10,1"/>
        <TextBox x:Name="txb_confidence" HorizontalAlignment="Stretch" HorizontalContentAlignment="Center"
                    Text="{Binding Path=Confidence, Mode=TwoWay, UpdateSourceTrigger=PropertyChanged}"/>
    </DockPanel>
    <DockPanel>
        <Label Content="MulConfidence" Margin="5,1,10,1"/>
        <TextBox x:Name="txb_mulConfidence" HorizontalAlignment="Stretch" HorizontalContentAlignment="Center"
                    Text="{Binding Path=MulConfidence, Mode=TwoWay, UpdateSourceTrigger=PropertyChanged}"/>
    </DockPanel>
    <DockPanel>
        <Label Content="Overlap" Margin="5,1,10,1"/>
        <TextBox x:Name="txb_overlap" HorizontalAlignment="Stretch" HorizontalContentAlignment="Center" 
                    Text="{Binding Path=Overlap, Mode=TwoWay, UpdateSourceTrigger=PropertyChanged}"/>
    </DockPanel>
    <DockPanel>
        <Label Content="最大检测数" Margin="5,1,10,1"/>
        <TextBox x:Name="txb_maxDetections" HorizontalAlignment="Stretch" HorizontalContentAlignment="Center" 
                    Text="{Binding Path=MaxDetections, Mode=TwoWay, UpdateSourceTrigger=PropertyChanged}"/>
    </DockPanel>
</StackPanel>
```

在选择模型模板时，模型参数会加载显示到界面。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-01-11-10-34.png)

但当我们运行目标检测算法时，最终需要调整的其实是`YoloScorer`对象的`_model`的属性，但它是`private`的，我们添加一个`public`属性用来访问和修改`_model`。

```CSharp
public class YoloScorer : IDisposable
{
    public YoloModel model
    {
        get => _model;
        set => _model = value;
    }

    private YoloModel _model;
    ...
}
```

同样的，`YOLOV5` 对象的 `scorer` 也要是 `public` 的。

```CSharp
internal class YOLOV5
{
    public YoloScorer scorer { get; set; }
    ...
}
```

我们在加载`onnx`文件后重新指定绑定的数据上下文。

```CSharp
private async void btn_browse_modelFile_Click_Async(object sender, RoutedEventArgs e)
{
    OpenFileDialog openFileDialog = new OpenFileDialog { Filter = "onnx files(*.onnx)|*.onnx" };
    openFileDialog.Title = "请选择模型onnx文件";

    DialogResult result = openFileDialog.ShowDialog();
    if (result == System.Windows.Forms.DialogResult.OK)
    {
        ...
        // 加载模型
        if (await Task.Run(() => LoadModel(onnxPath)))
        {
            ...

            // 绑定数据上下文
            sp_modelParam.DataContext = m_yolov5.scorer.model;
        }
    }
}
```

### 6.1.2. 标签列表可编辑

对于标签列表，我们也希望它是可编辑的，这样我们可以在用户界面修改标签名称和预测框的颜色。

首先，在 `XAML` 中添加绑定：

```XML
<StackPanel>
    <Label Content="标签"/>
    <DataGrid x:Name="dg_labels" ItemsSource="{Binding}" AutoGenerateColumns="False" CanUserAddRows="False">
        <DataGrid.Columns>
            <DataGridTextColumn Header="ID" Width="auto" Binding="{Binding Id}" IsReadOnly="True"/>
            <DataGridTextColumn Header="名称" Width="auto" Binding="{Binding Name}" IsReadOnly="False"/>
            <DataGridComboBoxColumn Header="颜色" Width="auto" SelectedItemBinding="{Binding Color}" ItemsSource="{Binding Source={x:Static yoloScorer:YoloLabel.Colors}}" IsReadOnly="False" />
        </DataGrid.Columns>
    </DataGrid>
</StackPanel>
```

其中颜色的元素变成了 `<DataGridComboBoxColumn>`，它的 `ItemsSource` 绑定的是在 `YoloLabel` 中定义的颜色数组：

```CSharp
...

namespace Yolov5Net.Scorer
{
    public class YoloLabel
    {
        ...

        public static Color[] Colors { get; } = new Color[]
        {
            Color.Green,
            Color.Red,
            Color.Blue,
            Color.Yellow,
            Color.Lime,
            Color.Cyan,
            Color.Magenta,
            Color.Brown,
            Color.DarkGreen,
            Color.DarkRed,
            Color.DarkGray,
            Color.LightCoral,
            Color.LightSalmon,
            Color.Maroon
        };
    }
}
```

在切换参数模板事件函数中添加绑定数据上下文的代码

```CSharp
...
private ObservableCollection<YoloLabel> m_yolov5ModelLabels;
...
private void cbb_modelType_SelectionChanged(object sender, SelectionChangedEventArgs e)
{
    ...

    // 绑定上下文
    ...
    m_yolov5ModelLabels = new ObservableCollection<YoloLabel>(m_yolov5?.scorer?.model.Labels);
    dg_labels.DataContext = m_yolov5ModelLabels;
}
```

最终效果：

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-04-11-12-41.png)

至此，我们实现了模型参数的可视可调，可以在运行目标检测时实时更改参数了。

## 6.2. 模板参数加载与保存

回顾我们的部署流程可以发现，当我们得到输出的 `onnx` 文件之后，都要新增一个数据集相关的 `YoloV5***Model.cs`，而在上一节中，我们已经实现了在用户界面修改其中的一些参数，还需要在代码中做的是修改类别数量和类别标签列表。

```CSharp
internal class YoloV5***Model : YoloModel
{
    public static int classCount = ...;

    public YoloV5***Model()
    {
        Dimensions = classCount + 5; // = 分类数 + 5

        ...

        Labels = new List<YoloLabel>()
        {
            new YoloLabel { Id = 0, Name = "Class1" , Color = Color.Green},
            new YoloLabel { Id = 1, Name = "Class2" , Color = Color.Red},
            ...
        };
    }
}
```

我们可以通过上面`YoloV5***Model`的方式定义一个参数模板，但是如果我们对模板做了修改，这些修改并不能被保存，下次运行程序时，还是加载的这些写死的参数。所以接下来我们用读取和保存`XML`的方式实现参数的加载与保存。

### 6.2.1. 用户操作流程

将界面稍作调整。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-02-11-21-50.png)

我们希望用户操作流程是

- 用户选择模型文件
- 模板默认为`Default`，程序自动加载`Default`的参数配置
  - 其中标签的数量根据`onnx`模型文件的输出来获取
  - 标签名默认为`Class1` `Class2`……
- 用户选择模板来切换参数配置
  - 当选择配置文件时，则打开文件浏览窗口选择`XML`文件，加载配置参数
- 用户点击保存按钮保存参数到`XML`文件

### 6.2.2. （反）序列化工具类

我们使用`.NET`的序列化和反序列化实现`XML`文件的加载（反序列化）与保存（序列化）。

定义一个工具类 `XmlUtil` 如下：

```CSharp
...

namespace RPSoft_Core.Utils
{
    public static class XmlUtil
    {
        /// <summary>
        /// 将对象obj序列化为xml文件
        /// </summary>
        /// <typeparam name="T">序列化对象类型</typeparam>
        /// <param name="obj">序列化对象</param>
        /// <param name="filePath">xml文件保存路径</param>
        public static void SerializeObject<T>(T obj, string filePath)
        {
            // 序列化设置
            XmlWriterSettings settings = new XmlWriterSettings
            {
                // 缩进
                Indent = true
            };

            // 文件流
            Stream fs = new FileStream(filePath, FileMode.Create);
            XmlWriter xmlWriter = XmlWriter.Create(fs, settings);

            try
            {
                // 序列化
                XmlSerializer xmlSerializer = new XmlSerializer(obj.GetType());
                xmlSerializer.Serialize(xmlWriter, obj);
            }
            catch (Exception)
            {
                MessageBoxUtil.ShowError("XML文件写入失败。");
            }
            finally
            {
                // 关闭文件流
                xmlWriter.Close();
                fs.Close();
            }

        }
        /// <summary>
        /// 读取xml文件，反序列化为对象
        /// </summary>
        /// <typeparam name="T">反序列化对象类型</typeparam>
        /// <param name="filePath">xml文件路径</param>
        /// <returns></returns>
        public static T DeserializeObject<T>(string filePath)
        {
            // 文件流
            Stream fs = new FileStream(filePath, FileMode.Open);
            XmlReader xmlReader = XmlReader.Create(fs);
            try
            {
                // 反序列化
                XmlSerializer xmlSerializer = new XmlSerializer(typeof(T));
                T obj = (T)xmlSerializer.Deserialize(xmlReader);
                return obj;
            }
            catch (Exception)
            {
                MessageBoxUtil.ShowError("XML文件读取失败。");
            }
            finally
            {
                // 关闭文件流
                xmlReader.Close();
                fs.Close();
            }
            return default;
        }
    }
}
```

序列化过程：

- 序列化设置
  - 使用 `XmlWriterSettings` 对象进行一些设置，这里设置缩进，否则输出的 `XML` 文件只有一行
- 打开文件流
  - 新建文件流，采用创建文件模式，创建一个 `XmlWriter` 准备写 `XML` 文件
- 序列化
  - 创建 `XmlSerializer` 对象，使用 `Serialize()` 将指定对象序列化为 `XML`

反序列化过程：

- 打开文件流
  - 新建文件流，采用读文件模式，创建一个 `XmlReader` 准备读 `XML` 文件
- 序列化
  - 创建 `XmlSerializer` 对象，使用 `Deserialize()` 将 `XML` 反序列化为指定对象

### 6.2.3. 参数保存与加载的实现

在模板参数区域，添加保存和加载按钮

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-04-10-15-20.png)

添加点击事件函数

```CSharp
private void btn_modelParam_load_Click(object sender, RoutedEventArgs e)
{
    if (m_yolov5 == null)
        return;

    OpenFileDialog openFileDialog = new OpenFileDialog
    {
        InitialDirectory = @"YOLO\",
        Filter = "XML文件|*.xml;*.XML"
    };
    DialogResult result = openFileDialog.ShowDialog();
    if (result == System.Windows.Forms.DialogResult.OK)
    {
        m_yolov5.scorer.model = XmlUtil.DeserializeObject<YoloModel>(openFileDialog.FileName);
        // 绑定上下文
        sp_modelParam.DataContext = m_yolov5?.scorer?.model;
        m_yolov5ModelLabels = new ObservableCollection<YoloLabel>(m_yolov5?.scorer?.model.Labels);
        dg_labels.DataContext = m_yolov5ModelLabels;
    }
}

private void btn_modelParam_save_Click(object sender, RoutedEventArgs e)
{
    YoloModel yoloModel = m_yolov5?.scorer?.model;

    if (yoloModel == null)
        return;

    // 保存文件对话框
    SaveFileDialog saveFileDialog = new SaveFileDialog
    {
        InitialDirectory = @"YOLO\",
        Filter = "XML|*.xml"
    };

    DialogResult result = saveFileDialog.ShowDialog();

    if (result == System.Windows.Forms.DialogResult.OK)
    {
        XmlUtil.SerializeObject(yoloModel, saveFileDialog.FileName);
    }
}
```

我们在界面中加载模型文件，并选择一项模板参数（如`OKNG`），点击保存按钮保存`OKNG.xml`文件。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-04-10-35-48.png)

查看这个 `XML` 文件

```XML
<?xml version="1.0" encoding="utf-8"?>
<YoloV5OKNGModel xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Width>640</Width>
  <Height>640</Height>
  <Depth>3</Depth>
  <Strides>
    <int>8</int>
    <int>16</int>
    <int>32</int>
  </Strides>
  <Anchors>
    <ArrayOfArrayOfInt>
      <ArrayOfInt>
        <int>10</int>
        <int>13</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>16</int>
        <int>30</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>33</int>
        <int>23</int>
      </ArrayOfInt>
    </ArrayOfArrayOfInt>
    <ArrayOfArrayOfInt>
      <ArrayOfInt>
        <int>30</int>
        <int>61</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>62</int>
        <int>45</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>59</int>
        <int>119</int>
      </ArrayOfInt>
    </ArrayOfArrayOfInt>
    <ArrayOfArrayOfInt>
      <ArrayOfInt>
        <int>116</int>
        <int>90</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>156</int>
        <int>198</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>373</int>
        <int>326</int>
      </ArrayOfInt>
    </ArrayOfArrayOfInt>
  </Anchors>
  <Shapes>
    <int>80</int>
    <int>40</int>
    <int>20</int>
  </Shapes>
  <Confidence>0.2</Confidence>
  <MulConfidence>0.25</MulConfidence>
  <Overlap>0.45</Overlap>
  <MaxDetections>10</MaxDetections>
  <Outputs>
    <string>output</string>
  </Outputs>
  <Labels>
    <YoloLabel>
      <Id>0</Id>
      <Name>OK</Name>
      <Kind>Generic</Kind>
      <Color />
    </YoloLabel>
    <YoloLabel>
      <Id>1</Id>
      <Name>NG</Name>
      <Kind>Generic</Kind>
      <Color />
    </YoloLabel>
  </Labels>
  <UseDetect>true</UseDetect>
</YoloV5OKNGModel>
```

这里有两个问题：

第一个问题， `YoloLabel` 的 `Color` 属性没有被正确地序列化 —— `<Color />` 节点没有数据。

要解决这个问题，我们只需要在 `YoloLabel` 类中添加一个 `string` 属性，用它来“代理” `Color` 属性的序列化和反序列化。

```CSharp
...
using System.Xml.Serialization;

namespace Yolov5Net.Scorer
{
    public class YoloLabel
    {
        ...

        [XmlIgnore()] //Color对象没法序列化
        public Color Color { get; set; }

        // 用于序列化Color
        [XmlElement(nameof(Color))]
        public string XmlColor
        {
            get => Color.Name;
            set => Color = Color.FromName(value);
        }
        ...
    }
}

```

第二个问题，输出的 `XML` 根节点为 `<YoloV5OKNGModel>`，而我们在反序列化时（`XmlUtil.DeserializeObject<YoloModel>(openFileDialog.FileName)`）泛型类型不是 `YoloV5OKNGModel` 而是 `YoloModel`，这就会导致在加载这个 `OKNG.xml` 文件时读取出错：

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-04-11-39-58.png)

实际上，我们已经把参数保存到 `XML` 文件中，所以从现在开始，我们不再需要使用 `YoloV5***Model.cs` 继承 `YoloModel` 的方式来配置每个参数模板。我们将几个参数模板都序列化为 `XML` 文件，并将根节点 `<YoloV5***Model>` 改为 `<YoloModel>`，删掉几个 `YoloV5***Model.cs`，用 `XML` 文件取代它们。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-04-11-49-49.png)

为了让 `YoloModel` 能够实例化对象，删掉 `YoloModel.cs` 中的 `abstract` 修饰符。

```CSharp
namespace Yolov5Net.Scorer
{
    public class YoloModel : INotifyPropertyChanged
    {
        ...
    }
}
```

至此，我们就通过反序列化的方式来创建模型对象了：

```CSharp
...
private readonly string m_yoloModelXml_OKNG = @"YOLO\Models\OKNG.xml";
private readonly string m_yoloModelXml_bolt = @"YOLO\Models\Bolt.xml";
...
private void cbb_modelType_SelectionChanged(object sender, SelectionChangedEventArgs e)
{
    if (m_yolov5 == null)
        return;
    
    if (cbbi_modelType_OKNG.IsSelected)
    {
        m_yolov5.scorer.model = XmlUtil.DeserializeObject<YoloModel>(m_yoloModelXml_OKNG);
    }
    // 螺柱 模型
    else if (cbbi_modelType_bolt.IsSelected)
    {
        m_yolov5.scorer.model = XmlUtil.DeserializeObject<YoloModel>(m_yoloModelXml_bolt);
    }

    ...
}
```

### 6.2.4. 默认模型参数

在之前我们提到希望在选择 `onnx` 文件后，程序能够自动加载Default的参数配置，其中标签的数量根据onnx模型文件的输出来获取，标签名默认为Class1 Class2……

首先我们有 `Default.xml`，它定义了所有的默认参数，但是标签列表为空。

```XML
<?xml version="1.0" encoding="utf-8"?>
<YoloModel xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <Width>640</Width>
  <Height>640</Height>
  <Depth>3</Depth>
  <Strides>
    <int>8</int>
    <int>16</int>
    <int>32</int>
  </Strides>
  <Anchors>
    <ArrayOfArrayOfInt>
      <ArrayOfInt>
        <int>10</int>
        <int>13</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>16</int>
        <int>30</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>33</int>
        <int>23</int>
      </ArrayOfInt>
    </ArrayOfArrayOfInt>
    <ArrayOfArrayOfInt>
      <ArrayOfInt>
        <int>30</int>
        <int>61</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>62</int>
        <int>45</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>59</int>
        <int>119</int>
      </ArrayOfInt>
    </ArrayOfArrayOfInt>
    <ArrayOfArrayOfInt>
      <ArrayOfInt>
        <int>116</int>
        <int>90</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>156</int>
        <int>198</int>
      </ArrayOfInt>
      <ArrayOfInt>
        <int>373</int>
        <int>326</int>
      </ArrayOfInt>
    </ArrayOfArrayOfInt>
  </Anchors>
  <Shapes>
    <int>80</int>
    <int>40</int>
    <int>20</int>
  </Shapes>
  <Confidence>0.2</Confidence>
  <MulConfidence>0.25</MulConfidence>
  <Overlap>0.45</Overlap>
  <MaxDetections>30</MaxDetections>
  <Outputs>
    <string>output</string>
    <string>output0</string>
  </Outputs>
  <Labels>
    <YoloLabel>
    </YoloLabel>
  </Labels>
  <UseDetect>true</UseDetect>
</YoloModel>
```

我们定义一个默认的模型参数对象，在加载 `onnx` 前，反序列化得到模型的默认参数：

```CSharp
...
private YoloModel m_yolov5Model_default;
private readonly string m_yoloModelXml_default = @"YOLO\Models\Default.xml";
...
private bool LoadModel(string onnxPath)
{
    // 默认的参数
    m_yolov5Model_default = XmlUtil.DeserializeObject<YoloModel>(m_yoloModelXml_default);
    m_yolov5 = new YOLOV5(m_yolov5Model_default, onnxPath);
    return m_yolov5 != null;
}
```

在模板参数列表选择 `Default` 时，将模型参数切换成默认参数：

```CSharp
private void cbb_modelType_SelectionChanged(object sender, SelectionChangedEventArgs e)
{
    ...
    // 默认模型参数
    if (cbbi_modelType_default.IsSelected)
    {
        m_yolov5.scorer.model = m_yolov5Model_default;
    }
}
```

由于我们没有给 `Default.xml` 添加标签，所以现在加载默认参数后，标签列表为空。

要想自动填充标签列表，首先我们希望能够根据 `onnx` 模型的输出结构来确定类别数，在读取 `onnx` 文件代码的后一行添加一行代码，打上断点，运行程序，加载一个模型后查看输出的数据。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-02-09-47-55.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-02-09-48-33.png)

我们知道 `Yolo V5` 最后输出的是 `1*25200*(类别数+2)`的张量，可以看到 `Dimensions` 数组正是记录了输出张量的尺寸，有了这个认识，我们就可以动态地为模型参数对象添加标签了：

```Csharp
public YoloScorer(YoloModel yoloModel, string weights, SessionOptions opts = null)
{
    oclCaller = new OclCaller();
    oclCaller.Init();
    
    _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
    
    AddLabels(yoloModel);
    _model = yoloModel;
}

private void AddLabels(YoloModel yoloModel)
{
    // Labels 列表没有标签
    if (yoloModel.Labels.Count == 0)
    {
        NodeMetadata data = _inferenceSession.OutputMetadata.Values.First();
        // 获取类别数
        if (data.Dimensions.Length == 3)
        {
            // 类别数      data.Dimension：1*25200*(classCount+5)
            int classCount = data.Dimensions[2] - 5;
            for (int i = 0; i < classCount; i++)
            {
                YoloLabel yoloLabel = new YoloLabel(i, "Class" + i, YoloLabel.Colors[i % YoloLabel.Colors.Length]);
                yoloModel.Labels.Add(yoloLabel);
            }
        }
        else
        {
            Debug.WriteLine("获取onnx输出元数据失败。");
        }
    }
}
```

## 6.3. 实现ROI模块功能

YOLO V5 的输入图像尺寸为 $[640, 640]$，当输入图像尺寸过大时，会首先执行`ResizeImage()`对图像进行压缩，将其缩放到输入尺寸：

`YoloScorer.cs`

```CSharp
/// <summary>
/// Resizes image keeping ratio to fit model input size.
/// </summary>
private Bitmap ResizeImage(Image image)
{
    PixelFormat format = image.PixelFormat;

    var output = new Bitmap(_model.Width, _model.Height, format);

    var (w, h) = (image.Width, image.Height); // image width and height
    var (xRatio, yRatio) = (_model.Width / (float)w, _model.Height / (float)h); // x, y ratios
    var ratio = Math.Min(xRatio, yRatio); // ratio = resized / original
    var (width, height) = ((int)(w * ratio), (int)(h * ratio)); // roi width and height
    var (x, y) = ((_model.Width / 2) - (width / 2), (_model.Height / 2) - (height / 2)); // roi x and y coordinates
    var roi = new Rectangle(x, y, width, height); // region of interest

    using (var graphics = Graphics.FromImage(output))
    {
        graphics.Clear(Color.FromArgb(0, 0, 0, 0)); // clear canvas

        graphics.SmoothingMode = SmoothingMode.None; // no smoothing
        graphics.InterpolationMode = InterpolationMode.Bilinear; // bilinear interpolation
        graphics.PixelOffsetMode = PixelOffsetMode.Half; // half pixel offset

        graphics.DrawImage(image, roi); // draw scaled
    }

    return output;
}
```

而工业相机分辨率一般都是百万或千万级像素，如果我们的目标在画面中占比不大，经过缩放后就会更小，这会导致训练和检测的效果变得很差。

对于训练，我们可以在制作数据集的之前，使用[图像工厂](http://www.pcfreetime.com/picosmos/index.php?language=zh)等工具将图像进行批量裁剪。

而对于检测，我们可以实现ROI框选的功能，指定一块图像区域作为输入。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-24-11-40-46.png)

要实现ROI功能，我们需要做到一下几点：

- 自定义控件
  - 可以显示锚点和边框
  - 可以拖动锚点改变ROI框的形状
- 绑定数据
  - 在主界面显示ROI框的位置和寸尺，数值能够随着拖动实时改变
- 裁剪图像
  - 将相机采集的图像按ROI框的区域进行裁剪

### 6.3.1. ROI控件

首先我们添加一个继承自 `DrawingVisual` 的类，它用来画出ROI的边框和锚点。

```CSharp
/// <summary>
/// ROI DrawingVisual，画出ROI框与锚点
/// </summary>
public class ROIDrawingVisual : DrawingVisual
{
    Pen pen = new Pen(Brushes.LightGreen, 1);
    public void Draw(Point anchor_tf, Point anchor_br)
    {
        using (DrawingContext dc = RenderOpen())
        {
            dc.DrawRectangle(Brushes.Transparent, pen, new Rect(anchor_tf, anchor_br));
            dc.DrawEllipse(Brushes.Green, pen, anchor_tf, 3, 3); // 左上锚点
            dc.DrawEllipse(Brushes.Green, pen, anchor_br, 3, 3); // 右下锚点
            dc.DrawEllipse(Brushes.Green, pen, Point.Add(anchor_tf, (anchor_br - anchor_tf) / 2.0), 3, 3); // 中心点
        }
    }
}
```

接着我们定义一个枚举类型，划定对ROI框的操作类型：

```CSharp
public enum OperationType
{
    Drag_TopLeft,
    Drag_TopRight,
    Drag_BottomLeft,
    Drag_BottomRight,
    Move,
    None
}
```

最后我们添加一个 `ROICanvas` 类，它继承自 `Canvas` 类，定义了 `ROI` 控件的一些属性和行为。

```CSharp
/// <summary>
/// ROI 画布
/// </summary>
public class ROICanvas : Canvas
{
    private OperationType m_operationType = OperationType.None;
    private readonly ROIDrawingVisual m_roi;
    private Point m_lastPoint; //最近一次记录的鼠标位置

    public ROICanvas()
    {
        m_roi = new ROIDrawingVisual();
        AddLogicalChild(m_roi);
        AddVisualChild(m_roi);
    }

    protected override int VisualChildrenCount => 1;
    protected override Visual GetVisualChild(int index)
    {
        return m_roi;
    }

    public Point TopLeftAnchor
    {
        get => (Point)GetValue(Property_TopLeftAnchor);
        set => SetValue(Property_TopLeftAnchor, value);
    }

    public static readonly DependencyProperty Property_TopLeftAnchor =
        DependencyProperty.Register
        (
            "TopLeftAnchor",
            typeof(Point),
            typeof(ROICanvas),
            new FrameworkPropertyMetadata(new Point(0, 0), new PropertyChangedCallback(OnTopLeftAnchorPropertyChanged))
        );
    private static void OnTopLeftAnchorPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        ROICanvas canvas = (ROICanvas)d;
        Point value = (Point)e.NewValue;

        Point coercedValue = CoerceTopLeftAnchor(value, canvas);
        if (coercedValue != value)
        {
            canvas.TopLeftAnchor = coercedValue;
        }
        canvas.m_roi.Draw(canvas.TopLeftAnchor, canvas.BottomRightAnchor);
    }

    public Point BottomRightAnchor
    {
        get => (Point)GetValue(Property_BottomRightAnchor);
        set => SetValue(Property_BottomRightAnchor, value);
    }

    public static readonly DependencyProperty Property_BottomRightAnchor =
        DependencyProperty.Register
        (
            "BottomRightAnchor",
            typeof(Point),
            typeof(ROICanvas),
            new FrameworkPropertyMetadata(new Point(0, 0), new PropertyChangedCallback(OnBottomRightAnchorPropertyChanged))
        );

    private static void OnBottomRightAnchorPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        ROICanvas canvas = (ROICanvas)d;
        Point value = (Point)e.NewValue;

        Point coercedValue = CoerceBottomRightAnchor(value, canvas);
        if (coercedValue != value)
        {
            canvas.BottomRightAnchor = coercedValue;
        }

        canvas.m_roi.Draw(canvas.TopLeftAnchor, canvas.BottomRightAnchor);
    }
    /// <summary>
    /// ROI中心
    /// </summary>
    public Point CenterAnchor
    {
        get => (Point)GetValue(Property_CenterAnchor);
        set => SetValue(Property_CenterAnchor, value);
    }
    public static readonly DependencyProperty Property_CenterAnchor =
        DependencyProperty.Register
        (
            "CenterAnchor",
            typeof(Point),
            typeof(ROICanvas),
            new FrameworkPropertyMetadata(new Point(0,0), new PropertyChangedCallback(OnCenterAnchorPropertyChanged))
        );

    private static void OnCenterAnchorPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        ROICanvas canvas = (ROICanvas)d;
        Point value = (Point)e.NewValue;
        canvas.CenterAnchor = value;
    }
    /// <summary>
    /// ROI框宽
    /// </summary>
    public double RoiWidth
    {
        get => (double)GetValue(Property_RoiWidth);
        set => SetValue(Property_RoiWidth, value);
    }

    public static readonly DependencyProperty Property_RoiWidth =
        DependencyProperty.Register
        (
            "RoiWidth",
            typeof(double),
            typeof(ROICanvas),
            new FrameworkPropertyMetadata(0.0, new PropertyChangedCallback(OnRoiWidthPropertyChanged))
        );

    private static void OnRoiWidthPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        ROICanvas canvas = (ROICanvas)d;
        double value = (double)e.NewValue;
        canvas.RoiWidth = value;
    }
    /// <summary>
    /// ROI框高
    /// </summary>
    public double RoiHeight
    {
        get => (double)GetValue(Property_RoiHeight);
        set => SetValue(Property_RoiHeight, value);
    }
    /// <summary>
    /// 注册属性到页面
    /// </summary>
    public static readonly DependencyProperty Property_RoiHeight =
        DependencyProperty.Register
        (
            "RoiHeight",
            typeof(double),
            typeof(ROICanvas),
            new FrameworkPropertyMetadata(0.0, new PropertyChangedCallback(OnRoiHeightPropertyChanged))
        );

    private static void OnRoiHeightPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
    {
        ROICanvas canvas = (ROICanvas)d;
        double value = (double)e.NewValue;
        canvas.RoiHeight = value;
    }
    protected override void OnMouseMove(MouseEventArgs e)
    {
        // 当前鼠标所在点
        Point point = e.GetPosition(this);

        // 判断操作类型
        if (m_operationType == OperationType.None)
        {
            // 鼠标接近左上锚点
            if (IsCloseTo(TopLeftAnchor, point))
            {
                // 改变鼠标指针
                Cursor = Cursors.SizeNWSE;
                // 鼠标左键按下
                if (e.LeftButton == MouseButtonState.Pressed)
                {
                    // 改变操作类型
                    m_operationType = OperationType.Drag_TopLeft;
                }
            }
            // 鼠标接近右下锚点
            else if (IsCloseTo(BottomRightAnchor, point))
            {
                // 改变鼠标指针
                Cursor = Cursors.SizeNWSE;
                // 鼠标左键按下
                if (e.LeftButton == MouseButtonState.Pressed)
                {
                    // 改变操作类型
                    m_operationType = OperationType.Drag_BottomRight;
                }
            }
            // 鼠标在ROI内
            else if (IsInterior(m_roi, point))
            {
                // 改变鼠标指针
                Cursor = Cursors.SizeAll;
                // 鼠标左键按下
                if (e.LeftButton == MouseButtonState.Pressed)
                {
                    // 改变操作类型 Move
                    m_operationType = OperationType.Move;
                }
            }
            else
            {
                // 恢复默认箭头
                Cursor = Cursors.Arrow;
            }
        }


        // 执行对应操作
        switch (m_operationType)
        {
            case OperationType.Drag_TopLeft:
                TopLeftAnchor = point;
                UpdateWidthHeight();
                UpdateCenterAnchor();
                break;
            case OperationType.Drag_TopRight:
                break;
            case OperationType.Drag_BottomLeft:
                break;
            case OperationType.Drag_BottomRight:
                BottomRightAnchor = point;
                UpdateWidthHeight();
                UpdateCenterAnchor();
                break;
            case OperationType.Move:
                MoveRoi(point, m_lastPoint);
                UpdateCenterAnchor();
                break;
            case OperationType.None:
                break;
            default:
                break;
        }
        // 更新最近鼠标位置
        m_lastPoint = point;
    }
    /// <summary>
    /// 更新宽高、中心点等属性的数据
    /// </summary>
    private void UpdateWidthHeight()
    {
        RoiWidth = (BottomRightAnchor - TopLeftAnchor).X;
        RoiHeight = (BottomRightAnchor - TopLeftAnchor).Y;
    }
    private void UpdateCenterAnchor()
    {
        CenterAnchor = Point.Add(TopLeftAnchor, (BottomRightAnchor - TopLeftAnchor) / 2.0);
    }

    protected override void OnMouseLeave(MouseEventArgs e)
    {
        base.OnMouseLeave(e);
        m_operationType = OperationType.None;
    }

    protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonDown(e);
        m_lastPoint = e.GetPosition(this);
    }
    protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
    {
        base.OnMouseLeftButtonUp(e);
        m_operationType = OperationType.None;
    }

    /// <summary>
    /// 移动ROI框
    /// </summary>
    /// <param name="point">当前位置</param>
    /// <param name="lastPoint">上一次记录的位置</param>
    private void MoveRoi(Point point, Point lastPoint)
    {
        double xOffset = point.X - lastPoint.X;//右方向为正
        double yOffset = point.Y - lastPoint.Y;//下方向为正

        // 由于菜单下拉状态时 Mouse Move 事件不触发，lastPosition不更新
        // 限制offset防止瞬时位移过大
        if (xOffset > 25 || yOffset > 25)
        {
            return;
        }

        if ((TopLeftAnchor.X == 0 && xOffset < 0) || //触及左边界，不能再往左 xOffset不能小于0
            (TopLeftAnchor.Y == 0 && yOffset < 0) || //触及上边界，不能再往上 yOffset不能小于0
            (BottomRightAnchor.X == ActualWidth && xOffset > 0) || //触及右边界，不能再往右  xOffset不能大于0
            (BottomRightAnchor.Y == ActualHeight && yOffset > 0)) //触及下边界，不能再往下 yOffset不能大于0
        {
            return;
        }

        var topLeft = CoerceTopLeftAnchor(new Point(TopLeftAnchor.X + xOffset, TopLeftAnchor.Y + yOffset), this);
        var bottomRight = CoerceBottomRightAnchor(new Point(BottomRightAnchor.X + xOffset, BottomRightAnchor.Y + yOffset), this);

        if (TopLeftAnchor != topLeft)
            TopLeftAnchor = topLeft;

        if (BottomRightAnchor != bottomRight)
            BottomRightAnchor = bottomRight;
    }


    /// <summary>
    /// 检测给定点是否在锚点附近
    /// </summary>
    /// <param name="anchor">锚点</param>
    /// <param name="point">给定点</param>
    /// <returns></returns>
    private bool IsCloseTo(Point anchor, Point point)
    {
        double r = 6;
        return Math.Abs((anchor - point).Length) <= r;
    }

    private bool IsInterior(DrawingVisual drawingVisual, Point point)
    {
        return drawingVisual.ContentBounds.Contains(point);
    }

    /// <summary>
    /// 左上锚点约束
    /// </summary>
    /// <param name="point">当前鼠标点</param>
    /// <param name="canvas">ROI画布</param>
    /// <returns>位置约束后的点</returns>
    private static Point CoerceTopLeftAnchor(Point point, ROICanvas canvas)
    {
        // 相对于右下锚点的约束
        if (point.X > canvas.BottomRightAnchor.X)
        {
            point.X = canvas.BottomRightAnchor.X;
        }
        if (point.Y > canvas.BottomRightAnchor.Y)
        {
            point.Y = canvas.BottomRightAnchor.Y;
        }

        // 相对于左、上边界的约束
        if (point.X < 0)
        {
            point.X = 0;
        }
        if (point.Y < 0)
        {
            point.Y = 0;
        }

        return point;
    }
    /// <summary>
    /// 右下锚点约束
    /// </summary>
    /// <param name="point">当前鼠标点</param>
    /// <param name="canvas">ROI画布</param>
    /// <returns>位置约束后的点</returns>
    private static Point CoerceBottomRightAnchor(Point point, ROICanvas canvas)
    {
        // 相对于左上锚点的约束
        if (point.X < canvas.TopLeftAnchor.X)
        {
            point.X = canvas.TopLeftAnchor.X;
        }
        if (point.Y < canvas.TopLeftAnchor.Y)
        {
            point.Y = canvas.TopLeftAnchor.Y;
        }

        // 相对于右、下边界的约束
        if (point.X > canvas.ActualWidth)
        {
            point.X = canvas.ActualWidth;
        }
        if (point.Y > canvas.ActualHeight)
        {
            point.Y = canvas.ActualHeight;
        }

        return point;
    }
    private static double CoerceROIWidth(double width, ROICanvas canvas)
    {
        if (width > canvas.Width)
        {
            width = canvas.Width;
        }

        return width;
    }
}
```

上述代码定义了ROI的一些属性：左上锚点、右下锚点、中心点、宽和高。鼠标靠近锚点，拖动锚点可以改变ROI的形状，鼠标在ROI框内可以拖动ROI框。

### 6.3.2. 添加用户控件

上一节我们添加了ROI控件类，现在我们来把控件放到页面中。首先需要添加命名控件指向 `ROI` 控件类所在的目录，然后添加元素，配置属性。

```XML
<UserControl ...
            xmlns:roi="clr-namespace:RPSoft_Core.GUI.ROI;assembly=RPSoft_Core"
            ...>
    <Grid>
        ...
        <Image Grid.Row="1" x:Name="img_image"/>
        <roi:ROICanvas x:Name="canvas_roi" Grid.Row="1"
                       Width="{Binding Path=ActualWidth, ElementName=img_image}"
                       Height="{Binding Path=ActualHeight, ElementName=img_image}"
                       HorizontalAlignment="Stretch" VerticalAlignment="Stretch" Background="Transparent"/>
        ...
    </Grid>
</UserControl>
```

我们想要ROI框在相机采集的图像上显示，所以我们将`<ROICanvas/>`放在与`<Image/>`相同的位置，控件的`Width`和`Height`属性（并不是ROI框的宽高）与`<Image/>`的宽高绑定，这样保证了ROI显示的区域和图像显示的区域是一致的。

接下来我们在主页面添加几个控件：

```XML
<StackPanel x:Name="sp_roi_param">
    <DockPanel Height="auto">
        <MenuItem Header="可见"/>
        <CheckBox x:Name="cb_roi_visibility"/>
    </DockPanel>
    <DockPanel Height="auto">
        <MenuItem Header="固定"/>
        <CheckBox x:Name="cb_roi_fixed"/>
    </DockPanel>
    <Grid>
        <Grid.Resources>
            <Style TargetType="TextBox">
                <Setter Property="Margin" Value="5,3,5,3"/>
                <Setter Property="HorizontalAlignment" Value="Stretch"/>
                <Setter Property="IsReadOnly" Value="True"/>
            </Style>
            <Style TargetType="Label">
                <Setter Property="FontSize" Value="12"/>
            </Style>
        </Grid.Resources>
        <Grid.ColumnDefinitions>
            <ColumnDefinition Width="auto"/>
            <ColumnDefinition Width="auto"/>
            <ColumnDefinition Width="*"/>
            <ColumnDefinition Width="auto"/>
            <ColumnDefinition Width="*"/>
        </Grid.ColumnDefinitions>
        <Grid.RowDefinitions>
            <RowDefinition/>
            <RowDefinition/>
        </Grid.RowDefinitions>
        <Label Content="中心" Grid.Row="0" Grid.Column="0"/>
        <Label Content="X" Grid.Row="0" Grid.Column="1"/>
        <Label Content="Y" Grid.Row="0" Grid.Column="3"/>
        <Label Content="尺寸" Grid.Row="1" Grid.Column="0"/>
        <Label Content="W" Grid.Row="1" Grid.Column="1"/>
        <Label Content="H" Grid.Row="1" Grid.Column="3"/>
        <TextBox x:Name="cbb_roi_x" Grid.Row="0" Grid.Column="2"/>
        <TextBox x:Name="cbb_roi_y" Grid.Row="0" Grid.Column="4"/>
        <TextBox x:Name="cbb_roi_w" Grid.Row="1" Grid.Column="2"/>
        <TextBox x:Name="cbb_roi_h" Grid.Row="1" Grid.Column="4"/>
    </Grid>
</StackPanel>
```

效果如图：

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-24-13-37-44.png)

### 6.3.3. 数据绑定

对于添加的几个控件，我们的需求如下：

- ROI选择框
  - 勾选，启用ROI功能，反之则取消ROI功能
- 可见选择框
  - 勾选，ROI框在图像区域显示，反之ROI框消失
- 固定选择框
  - 勾选，ROI固定不可拖动
- 中心XY
  - 显示中心点坐标
- 尺寸WH
  - 显示ROI框的宽高

要实现上述功能，我们需要将页面这些控件状态和ROI框属性绑定，这就需要一个“代理”来完成这一步，即需要一个`ViewModel`:

```CSharp
public class ROIViewModel : INotifyPropertyChanged
{
    public ROIViewModel(Point anchor_tl, Point anchor_br)
    {
        topLeftAnchor = anchor_tl;
        bottomRightAnchor = anchor_br;
        centerAnchor = Point.Add(anchor_tl, (anchor_br - anchor_tl) * 0.5);
        roiWidth = (anchor_br - anchor_tl).X;
        roiHeight = (anchor_br - anchor_tl).Y;
    }

    public event PropertyChangedEventHandler PropertyChanged;
    private void OnPropertyChanged([CallerMemberName] string name = "")
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }

    private bool isVisible = false;
    public bool IsVisible
    {
        get => isVisible;
        set
        {
            isVisible = value;
            OnPropertyChanged();
        }
    }

    private bool isFixed = false;
    public bool IsFixed
    {
        get => isFixed;
        set
        {
            isFixed = value;
            OnPropertyChanged();
        }
    }
    /// <summary>
    /// 左上锚点
    /// </summary>
    private Point topLeftAnchor = new Point(0, 0);
    public Point TopLeftAnchor
    {
        get { return topLeftAnchor; }
        set
        {
            topLeftAnchor = value;
            OnPropertyChanged();
        }
    }
    /// <summary>
    /// 右下锚点
    /// </summary>
    private Point bottomRightAnchor = new Point(640, 640);
    public Point BottomRightAnchor
    {
        get { return bottomRightAnchor; }
        set
        {
            bottomRightAnchor = value;
            OnPropertyChanged();
        }
    }
    /// <summary>
    /// 中心点
    /// </summary>
    private Point centerAnchor;
    public Point CenterAnchor
    {
        get => centerAnchor;
        set
        {
            centerAnchor = value;
            OnPropertyChanged();
        }
    }
    /// <summary>
    /// ROI框宽
    /// </summary>
    private double roiWidth;
    public double RoiWidth
    {
        get => roiWidth;
        set
        {
            roiWidth = value;
            OnPropertyChanged();
        }
    }
    /// <summary>
    /// ROI框高
    /// </summary>
    public double roiHeight;
    public double RoiHeight
    {
        get => roiHeight;
        set
        {
            roiHeight = value;
            OnPropertyChanged();
        }
    } 
}
```

上面的`ROIViewModel`类定义了一些ROI属性，如左上锚点、右下锚点、宽高等，与`<ROICanvas/>`控件的属性一致，还有`IsVisible`和`IsFixed`属性定义控制可见和固定状态。另外，`ROIViewModel`类实现了`INotifyPropertyChanged`接口，以实现双向数据绑定。

接下来我们就可以用`ROIViewModel`类来实现之前提到的需求。

添加ROI框勾选和取消勾选的事件：

```CSharp
private void cb_roi_Checked(object sender, RoutedEventArgs e)
{
    // 如果没有ROI，新建一个
    if (m_roi == null || m_roi.RoiWidth == 0 || m_roi.RoiHeight == 0)
    {
        // ROI初始化
        System.Windows.Point center = new System.Windows.Point(uct_image.img_image.ActualWidth * 0.5, uct_image.img_image.ActualHeight * 0.5);
        // roi初始在画面中心 长宽占图像长宽一半
        m_roi = new ROIViewModel(new System.Windows.Point(center.X * 0.5, center.Y * 0.5), new System.Windows.Point(center.X * 1.5, center.Y * 1.5))
        {
            IsVisible = true
        };

        uct_image.canvas_roi.DataContext = m_roi;
        sp_roi.DataContext = m_roi;
    }
    sp_roi_param.IsEnabled = true;
    cb_roi_visibility.IsChecked = true;
}

private void cb_roi_Unchecked(object sender, RoutedEventArgs e)
{
    if (m_roi != null)
    {
        cb_roi_visibility.IsChecked = false;
        sp_roi_param.IsEnabled = false;
    }
}
```

在`Checked`事件函数中，我们创建了ROI（`ROIViewModel`对象），并将其指定为ROI相关元素的`DataContext`，如此一来，我们在`XAML`中绑定对应的属性即可。

将`ROICanvas`的属性与`ROIViewModel`对象属性绑定:

```XML
<UserControl ... 
             xmlns:roi="clr-namespace:RPSoft_Core.GUI.ROI;assembly=RPSoft_Core"
             xmlns:cvt="clr-namespace:RPSoft_Core.GUI.Converter;assembly=RPSoft_Core"
             ...>
    <UserControl.Resources>
        <cvt:Bool2VisibilityConverter x:Key="bool2VisibilityConverter"/>
        <cvt:BoolInverseConverter x:Key="boolInversConverter"/>
    </UserControl.Resources>
    <Grid>
        <roi:ROICanvas x:Name="canvas_roi" Grid.Row="1"
                        Width="{Binding Path=ActualWidth, ElementName=img_image}"
                        Height="{Binding Path=ActualHeight, ElementName=img_image}"
                        HorizontalAlignment="Stretch" VerticalAlignment="Stretch" Background="Transparent"
                        Visibility="{Binding IsVisible, Converter={StaticResource bool2VisibilityConverter}, Mode=TwoWay}"
                        IsEnabled="{Binding IsFixed, Converter={StaticResource boolInversConverter}, Mode=TwoWay}"
                        BottomRightAnchor = "{Binding BottomRightAnchor, UpdateSourceTrigger=PropertyChanged, Mode=TwoWay}"
                        TopLeftAnchor = "{Binding TopLeftAnchor, UpdateSourceTrigger=PropertyChanged, Mode=TwoWay}"
                        CenterAnchor = "{Binding CenterAnchor, Mode=TwoWay}"
                        RoiWidth="{Binding RoiWidth, Mode=TwoWay}"
                        RoiHeight="{Binding RoiHeight, Mode=TwoWay}"/>
    </Grid>
</UserControl>
```

这里注意 `Visibility` 和 `IsEnabled` 的绑定还指定了 `Converter`，这是因为：

- `Visibility` 并不是布尔变量，需要一个转换器将 `IsVisible` 与 `Visibility` 的值对应
- `IsEnabled` 虽然是布尔变量，但它 与 `IsFixed` 的逻辑是反着的，需要一个转换器将它们的值对应起来

两个`Converter`的内容如下：

```CSharp
/// <summary>
/// Bool <-> Visibility
/// </summary>
public class Bool2VisibilityConverter : IValueConverter
{
    // 对象属性 -> 控件属性
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value == null)
        {
            return Visibility.Collapsed;
        }

        return (bool)value ? Visibility.Visible : Visibility.Collapsed;

    }

    // 控件属性 -> 对象属性
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return (Visibility)value == Visibility.Visible;
    }
}
```

```CSharp
/// <summary>
/// Bool 值反转
/// </summary>
public class BoolInverseConverter : IValueConverter
{
    // 对象属性 -> 控件属性
    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return !(bool)value;
    }

    // 控件属性 -> 对象属性
    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        return !(bool)value;
    }
}
```

最后我们将主界面的控件也与 `ROIViewModel` 的属性绑定：

```XML
<StackPanel Grid.Row="3" Margin="0,10,0,5" x:Name="sp_roi" DataContext="{Binding}">
    <DockPanel LastChildFill="False">
        <Label Grid.Column="0" Grid.Row="0" Content="ROI"/>
        <CheckBox DockPanel.Dock="Right" x:Name="cb_roi" Checked="cb_roi_Checked" Unchecked="cb_roi_Unchecked"/>
    </DockPanel>
    <StackPanel x:Name="sp_roi_param">
        <DockPanel Height="auto">
            <MenuItem Header="可见"/>
            <CheckBox x:Name="cb_roi_visibility" IsChecked="{Binding IsVisible}"/>
        </DockPanel>
        <DockPanel Height="auto">
            <MenuItem Header="固定"/>
            <CheckBox x:Name="cb_roi_fixed" IsChecked="{Binding IsFixed}"/>
        </DockPanel>
        <Grid>
            ...
            <TextBox x:Name="cbb_roi_x" Grid.Row="0" Grid.Column="2" Text="{Binding CenterAnchor.X, Mode=OneWay, StringFormat={}{0:F0}}"/>
            <TextBox x:Name="cbb_roi_y" Grid.Row="0" Grid.Column="4" Text="{Binding CenterAnchor.Y, Mode=OneWay, StringFormat={}{0:F0}}"/>
            <TextBox x:Name="cbb_roi_w" Grid.Row="1" Grid.Column="2" Text="{Binding RoiWidth, Mode=OneWay, StringFormat={}{0:F0}}"/>
            <TextBox x:Name="cbb_roi_h" Grid.Row="1" Grid.Column="4" Text="{Binding RoiHeight, Mode=OneWay, StringFormat={}{0:F0}}"/>
        </Grid>
    </StackPanel>
</StackPanel>
```

至此，ROI显示的功能就算基本实现了，拖动锚点查看中心和尺寸数值是否跟着变化。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-24-14-25-49.png)

### 6.3.4. 输入图像的裁剪

此前我们实现了ROI的基础功能，但有一个隐藏的问题：此时ROI框的宽高并不是实际所框选区域的原始像素尺寸，这是因为图像并不是按原始分辨率显示的，在图像显示区的画面是经过缩放的。以海康的一款200万像素相机（MV-CA020-10GM）为例，它的分辨率是 $1624*1240$，显然除非我们的屏幕分辨率足够大，否则不可能放得下那么大尺寸的图片。

虽然我们在上图中看到ROI框的宽高是 $133 * 290$，但实际框选区域的像素数要远远大于这个数值。因此在裁剪的时候，我们传入的尺寸数据需要等比例转换。

我们在运行检测的函数中添加这样的逻辑：

```CSharp
private void RunDetect(Bitmap bitmap)
{
    DetectResult result;
    // 如果启用了ROI
    if (m_roi != null && m_roi.RoiWidth != 0 && m_roi.IsUsing)
    {
        // 实际 / 显示 比例
        double ratio = bitmap.Width / uct_image.img_image.ActualWidth;
        Rectangle roiRect = new Rectangle((int)(m_roi.TopLeftAnchor.X * ratio), (int)(m_roi.TopLeftAnchor.Y * ratio), (int)(m_roi.RoiWidth * ratio), (int)(m_roi.RoiHeight * ratio));
        if (roiRect.Width * roiRect.Height > bitmap.Width * bitmap.Height)
        {
            roiRect = new Rectangle(0, 0, bitmap.Width , bitmap.Height);
        }
        m_yolov5.ObjectDetect(bitmap, roiRect, out result);
    }
    else
    {
        m_yolov5.ObjectDetect(bitmap, out result);
    }
    ...
}
```

在上面代码中，我们将ROI矩形框按图像实际与显示比例进行缩放，然后将ROI矩形框传入目标检测函数。

新的三参数目标检测函数对比原来略有修改，首先将传入的图像根据ROI框进行了裁剪，然后输入的目标检测模型后得到预测结果，最后将预测框偏移得到正确的预测结果。

```CSharp
public void ObjectDetect(Image image, Rectangle roi, out DetectResult result)
{
    ...
    List<YoloPrediction> predictions = scorer.Predict(CropImage(image, roi));
    stopwatch.Stop();
    ...

    var graphics = Graphics.FromImage(image);

    // ROI 左上角坐标偏移
    Point leftTop = new Point(roi.X, roi.Y);

    // 遍历预测结果，画出预测框
    foreach (var prediction in predictions)
    {
        ...

        // 将预测框偏移到正确位置
        RectangleF rect = prediction.Rectangle;
        rect.Offset(leftTop);
        prediction.Rectangle = rect;

        graphics.DrawRectangles(new Pen(prediction.Label.Color, 4), new[] { prediction.Rectangle }) ;

        ...
    }
}
/// <summary>
/// 裁剪图片
/// </summary>
/// <param name="image"></param>
/// <returns></returns>
public Bitmap CropImage(Image image, Rectangle roi)
{
    var output = new Bitmap(roi.Width, roi.Height, image.PixelFormat);

    using (var graphics = Graphics.FromImage(output))
    {
        graphics.Clear(Color.FromArgb(0, 0, 0, 0)); // clear canvas

        graphics.SmoothingMode = SmoothingMode.None; // no smoothing
        graphics.InterpolationMode = InterpolationMode.Bilinear; // bilinear interpolation
        graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

        graphics.DrawImage(image, 0, 0, roi, GraphicsUnit.Pixel); // draw scaled
    }

    return output;
}
```

至此，ROI模块的基本功能全部完成。

![](imgs/YOLO%20V5%20部署笔记.md/2022-11-24-21-14-11.png)

# 7. Debug

## 7.1. 点击停止采集按钮页面卡死

### 7.1.1. 现象

相机采集画面显示时，点击断开连接或停止采集按钮后，界面卡死。

### 7.1.2. 定位

点击停止采集按钮时执行的函数：

```CSharp
private void btn_stopGrabbing_Click(object sender, RoutedEventArgs e)
{
    // ch:标志位设为false | en:Set flag bit false
    m_bGrabbing = false;
    m_hReceiveThread.Join();

    // ch:停止采集 | en:Stop Grabbing
    int nRet = m_MyCamera.MV_CC_StopGrabbing_NET();
    if (nRet != MyCamera.MV_OK)
    {
        ShowErrorMsg("Stop Grabbing Fail!", nRet);
    }
}
```

调试发现，执行到 `m_hReceiveThread.Join();` 时界面就卡死了，该函数作用是阻止调用线程，直到线程终止。说明问题出在终止线程的过程中。

查看线程的处理函数

```CSharp
public void ReceiveThreadProcess()
{
    ...

    while (m_bGrabbing)
    {
        lock (BufForDriverLock)
        {
            nRet = m_MyCamera.MV_CC_GetOneFrameTimeout_NET(m_BufForFrame, nPayloadSize, ref stFrameInfo, 1000);
            if (nRet == MyCamera.MV_OK)
            {
                m_stFrameInfo = stFrameInfo;
            }
        }

        if (nRet == MyCamera.MV_OK)
        {
            if (RemoveCustomPixelFormats(stFrameInfo.enPixelType))
            {
                continue;
            }

            Bitmap bitmap = new Bitmap(m_stFrameInfo.nWidth, m_stFrameInfo.nHeight, PixelFormat.Format8bppIndexed);
            Rectangle rect = new Rectangle(0, 0, m_stFrameInfo.nWidth, m_stFrameInfo.nHeight);
            BitmapData bitmapData = bitmap.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            unsafe
            {
                Buffer.MemoryCopy(m_BufForFrame.ToPointer(), bitmapData.Scan0.ToPointer(), m_nBufSizeForDriver, m_nBufSizeForDriver);
            }
            bitmap.UnlockBits(bitmapData);

            if (m_isRunning)
            {
                RunDetect(bitmap);
            }
            Dispatcher.Invoke(new Action(delegate
            {
                uct_image.ShowImage(bitmap);
            }));
        }
    }
}
```

在函数的最后，我们使用 `Invoke()` 切到了主 UI 线程显示图片，也就是说，相机采集显示的线程与UI线程关联，界面一直显示图片，线程也就不会结束（猜测）。

将

```CSharp
Dispatcher.Invoke(new Action(delegate
{
    uct_image.ShowImage(bitmap);
}));
```
注释掉，界面卡死的现象消失，验证了我们的推断。

### 7.1.3. 解决

#### 7.1.3.1. 使用标志位

只有在采集图像时，才显示图像，利用图像采集标志位判断一下即可：

```CSharp
if (m_bGrabbing)
{
    Dispatcher.Invoke(new Action(delegate
    {
        uct_image.ShowImage(bitmap);
    }));
}
```

因为我们在执行 `m_hReceiveThread.Join();` 前，首先执行了 `m_bGrabbing = false;`，因此不会出错。

#### 7.1.3.2. 异步调用

当前使用的 `Invoke()` 是同步调用，线程会等待 `Action` 执行完毕后再继续执行，改为异步调用 `InvokeAsync()` 即可解决问题。

## 7.2. 

看来是电脑的问题