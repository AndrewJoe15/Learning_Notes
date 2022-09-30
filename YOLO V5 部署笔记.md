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
- LaelImg
  - 图片标注工具
- Git
  - 代码版本控制工具

## 1.1. 安装显卡驱动和CUDA

在Nvidia官网选择电脑配置的显卡型号，下载相应的[显卡驱动程序](https://www.nvidia.cn/geforce/drivers/)。

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-14-09-40-45.png)

命令行中输入命令`nvidia-smi`查看是否安装CUDA，

![](imgs/YOLO%20V5%20部署笔记.md/2022-08-14-15-02-59.png)

若未安装CUDA，在 NVIDIA 官网下载 [CUDA ToolKit](https://developer.nvidia.com/cuda-downloads) 并安装。

## 1.2. 安装Anaconda

下载安装 [Anaconda](https://www.anaconda.com/products/distribution) 用来管理 Python 环境。

## 1.3. 安装Pytorch

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

# 2. YOLO模型训练

## 2.1. 下载YOLO

在想要存放YOLO的文件夹中右键选择`Git Bash Here`，打开`Git Bash`

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-09-17-06-41.png)

执行如下命令将YOLO下载到本地。

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

使用我们输出的这个模型文件作为权重，将 `detect.py` 运行配置的 `weights` 改成 `best.onnx`。

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

但是对单张图片的检测并不能满足我们实际项目需求，接下来我们将探索如何使用视频或者相机作为输入。

## 3.3. 相机输入

### 3.3.1. MVS例程

安装MVS，打开官方自带的样例工程

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

### 实现

将`MvCameraControl.Net.dll`添加到我们的项目中

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-27-11-32-49.png)

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-27-11-36-12.png)

新建一个页面

![](imgs/YOLO%20V5%20部署笔记.md/2022-09-30-17-00-53.png)

它与之前的窗口类似，只不过左侧栏由原来的图片浏览列表变成了相机控制台。

`查找相机`、`连接`、`断开`、`开始采集`和`停止采集`按钮的点击事件直接复制海康SDK Demo 中的代码即可。

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
            Bitmap bitmap = new Bitmap(m_stFrameInfo.nWidth, m_stFrameInfo.nHeight, PixelFormat.Format8bppIndexed);
            Rectangle rect = new Rectangle(0, 0, m_stFrameInfo.nWidth, m_stFrameInfo.nHeight);
            BitmapData bitmapData = bitmap.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
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
```

在上述代码中，主要是使用 `MV_CC_GetOneFrameTimeout_NET()` 获取一帧数据，存放在缓存 `m_BufForFrame` 中，然后把数据拷贝到位图的像素数据内存 `bitmapData.Scan0` 中，最后传入到我们的目标检测函数去处理图片。

> 要注意的是，位图的像素格式 `PixelFormat` 需要根据相机的设置改变，才能正确地解析相机采集的图像。

# 4. 优化

## 4.1. 浮点运算速度优化
