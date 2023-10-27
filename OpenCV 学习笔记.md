<H1>OpenCV 学习笔记</H1>

[TOC]

# 入门

## 安装

### apt install 方式

Linux 系统中，C++环境下 OpenCV 的安装，可以使用如下的命令：

```Shell
sudo apt update
sudo apt install libopencv-dev
```
在 CMakeLists.txt 中添加如下语句以实现调用：

```CMake
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES})
# other settings
target_link_libraries(${TARGET} PRIVATE ${OpenCV_LIBRARIES})
```

### 源码安装方式

除了上面的安装方式，还可以通过源码编译的方式安装OpenCV。

#### 构建

```Shell
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
# Build
cmake --build .
```

按顺序执行命令即可。其中下载`opencv.zip`和`cmake ../opencv-4.x`这两步需要下载文件，可能需要使用科学上网的方式。

前几步时间都很快，几秒到几十秒，最后`cmake --build .`需要的时间比较长，我用了大概40分钟。

构建完成后，执行安装命令：

```Shell
sudo make install
```

#### 验证和配置安装

检查OpenCV是否安装成功：

```Shell
ldconfig -v | grep opencv
```

检查OpenCV的路径是否正确设置：

```Shell
cat /etc/ld.so.conf.d/opencv.conf
```

如果不存在opencv.conf，需要配置路径，新建配置文件：

```Shell
sudo nano /etc/ld.so.conf.d/opencv.conf
```

添加OpenCV的lib文件夹路径，保存并退出:

```Shell
/usr/local/lib
```

运行如下命令，以使更改生效：

```Shell
sudo ldconfig
```

#### 在QT中配置

我们已经安装了OpenCV，要在 QT Creator 中使用，在`.pro`文件中引入即可:

```MakeFile
INCLUDEPATH += /usr/local/include/opencv4/opencv2 \
LIBS += /usr/local/lib/libopencv*
```

## 读写图片

下面的类演示了图片读取、显示和存储功能。

```C++
#include "ImageReadWrite.h"
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

ImageReadWrite::ImageReadWrite()
{

}

void ImageReadWrite::Run()
{
    std::string image_path = "/home/lyx/WorkingSpace/img/image.bmp";
    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }

}
```

## 核心功能

### Mat

#### 简单介绍

OpenCV 中 `Mat` 是一个类，`Mat`类的对象用来表示和存储图像对象，它由两部分构成：矩阵头部和数据指针。

- 矩阵头部
  - 包含信息：矩阵尺寸，存储方法，矩阵存储地址等
- 数据指针
  - 指向矩阵数据本身，即矩阵含有的像素值
  - 维度取决于存储方法

矩阵头部的尺寸是个定值，不过整个矩阵本身的尺寸是由数据量级决定的。

OpenCV 是一个图像处理库，它包含了大量图像处理函数。我们使用这些函数时，经常会在函数之间传递图像参数，为了减少由于图像拷贝带来的不必要的性能损失，OpenCV 使用了 `Mat`。

每个 `Mat` 对象都有自己的头部，但两个`Mat`对象会共享矩阵（他们的数据指针指向同一个地址）。`Mat` 对象的复制操作符只会拷贝矩阵头部和数据指针，不会拷贝数据本身。

```C++
Mat A, C; // 创建头部部分
A = imread(argv[1], IMREAD_COLOR); // here we'll know the method used (allocate matrix)
Mat B(A); // 使用复制构造函数
C = A; // 赋值运算
```

在上面的示例中，三个 `Mat` 对象的数据指针指向同一个矩阵。三个不同的对象仅仅只是头部不同，它们提供了对同一个底层数据的不同的获取途径。

真正有趣的是，你可以创建一个头部，指向整个数据的一部分。例如，要创建一个 ROI，只需创建一个新的头部。

```C++
Mat D (A, Rect(10, 10, 100, 100) ); // 使用矩形
Mat E = A(Range::all(), Range(1,3)); // 使用行列范围
```

#### 引用计数

现在你可能会问，如果矩阵本身属于多个 `Mat` 对象，那么当它不再需要使用的时候，谁负责清理数据呢？

简短的回答是：最后一个使用矩阵的对象。这是由引用计数机制控制的。一旦 `Mat` 对象发生了拷贝，矩阵的计数器计数就会增加。同理，头部被清除，计数器计数也会减少。当计数器为零时，矩阵就会被释放。

有时我们也需要拷贝矩阵本身，这时就需要使用`cv::Mat::clone()`或`cv::Mat::copyTo()`函数。

```C++
Mat F = A.clone();
Mat G;
A.copyTo(G);
```

需要记住的几点：

- OpenCV函数的输出图像的内存分配是自动的
- 使用OpenCV的C++接口无需进行内存管理
- 赋值操作符和复制构造器只会拷贝头部
- 图像底层的矩阵数据可以使用`cv::Mat::clone()`或`cv::Mat::copyTo()`函数进行拷贝

#### 存储方法

即如何存储像素值。我们可以选择使用的颜色空间和数据类型。

- RGB，与人的视觉系统近似。不过注意，OpenCV中使用的标准显示系统是BGR色域。
- HSV/HLS，将颜色分解成色相（Hue）、饱和度（Saturation）和值/明度（Value/Luminance），这是我们描述颜色最自然的方式。比如，你可以消除最后一个成分，使得算法对输入图像的照明条件不那么敏感。
- YCrCb，JPEG图像格式使用。
- CIE-L*a*b，感官上均匀的颜色空间，使用它你可以很方便地衡量出给定的一种颜色与另一种颜色的色差。

#### 显示创建`Mat`对象

在前面[读写图片](#读写图片)的教程中，我们了解了使用`cv::imwrite()`函数来写入图像的矩阵。在调试的时候，我们最好能够看到矩阵里的值。要想查看这些数值，可以使用`<<`操作符来实现。（注意只适用于二维矩阵）

尽管`Mat`是图像的容器，同时它也是一般的矩阵类。因此，它也能创建和操作多维矩阵。

创建`Mat`对象的方式有多种。

- `cv::Mat::Mat`构造器
  ```C++
  Mat M(2,2, CV_8UC3, Scalar(0,0,255));
  cout << "M = " << endl << " " << M << endl << endl;
  ```
  对于二维图像，我们先定义了它的尺寸：行数和列数。
  
  然后，我们需要指定存储元素的数据类型和通道数，即宏变量`CV_8UC3`的作用，它遵从以下约定。

  `CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]`

  `CV_8UC3`表示数值类型使用8bit无符号`char`类型，每个像素有3个数值对应3个通道。

- 使用数组构造
  ```C++
  int sz[3] = {2,2,2};
  Mat L(3,sz, CV_8UC(1), Scalar::all(0));
  ```

  对于高于二维的矩阵，可以使用上述的方法，使用数组指定每个维度的尺寸。

- `cv::Mat::create`函数
  ```C++
  M.create(4,4, CV_8UC(2));
  cout << "M = "<< endl << " " << M << endl << endl;
  ```
  此种方式不能初始化矩阵值。

- MATLAB 风格的初始化器：`cv::Mat::zeros``cv::Mat::ones``cv::Mat::eye`。
  ```C++
  Mat E = Mat::eye(4, 4, CV_64F);
  cout << "E = " << endl << " " << E << endl << endl;
  Mat O = Mat::ones(2, 2, CV_32F);
  cout << "O = " << endl << " " << O << endl << endl;
  Mat Z = Mat::zeros(3,3, CV_8UC1);
  cout << "Z = " << endl << " " << Z << endl << endl;
  ```

- 对于一些小的矩阵，可以使用逗号分隔数值进行初始化。
  ```C++
  Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
  cout << "C = " << endl << " " << C << endl << endl;
  C = (Mat_<double>({0, -1, 0, -1, 5, -1, 0, -1, 0})).reshape(3);
  cout << "C = " << endl << " " << C << endl << endl;
  ```

- 使用`cv::Mat::clone`和`cv::Mat::copyTo`。
  ```C++
  Mat RowClone = C.row(1).clone();
  cout << "RowClone = " << endl << " " << RowClone << endl << endl;
  ```

### 扫描图像、查找表、时间测量

#### 测试案例

使用`unsigned char`类型存储矩阵元素的话，每像素的一个通道可以有256个不同的值，三通道图像就可以有1600万中不同的颜色。如此多的颜色数据无疑会严重影响图像算法性能，有些时候我们只需要少得多的颜色数据就已经足够了。

所以我们需要压缩颜色空间。这意味着我们要分割当前颜色空间的值，得到更少的颜色。例如0-9的值我们取0，10-19的值取10，以此类推。

在C/C++中我们用`int`变量对`uchar`变量做整除运算，转换为字符类型的变量，即整除的小数部分的余数将被抹掉。利用这一点，我们可以很容易实现上述的压缩。

$$
I_{new} = (I_{old} / 10) * 10
$$

带入图像矩阵中的每一个值到上述的公式，我们就实现了一个简单的色彩空间压缩的算法。值得注意的是，其中有一个乘法和除法操作，而这些操作是非常耗时的。如果有可能，应该换成更经济的操作，如减法、加法或者最好只是简单的赋值。

此外，注意我们只有有限的输入值，上述情况中256个可能的输入值。因此，对于更大图像，明智的做法是：预先计算所有可能的值，运行压缩算法时只做赋值操作。实现这一目标我们用到了查找表。

查找表是一维或多维的数组，一个输入变量对应一个输出变量。利用它我们可以不用计算，直接读取结果。

接下来，我们读取一幅图像并应用我们的压缩算法。

在OpenCV中，遍历一副图像的所有像素主要有三种方法。我们使用三种方法，统计每一种方法的耗时。

首先是计算查找表：

```C++
// input: int divideWith
uchar table[256];
for (int i = 0; i < 256; ++i)
  table[i] = (uchar)(divideWith * (i/divideWith));
```

然后是对耗时的计算，用到了`cv::getTickCount()`和`cv::getTickFrequency()`：

```C++
double t = (double)getTickCount();
// do something ...
t = ((double)getTickCount() - t)/getTickFrequency();
cout << "Times passed in seconds: " << t << endl;
```

#### 图像矩阵的存储

灰度图：

![](imgs/OpenCV%20学习笔记.md/2023-10-27-16-35-50.png)

BGR色彩系统：

![](imgs/OpenCV%20学习笔记.md/2023-10-27-16-36-56.png)

在大多数情况下，内存空间是足够的，可以一行接一行地连续存储，形成一行长长的数据，这种情况可以加快我们的遍历速度。使用`cv::Mat::isContinuous()`可以判断矩阵是否处于这种情况。

# 参考资料
- [OpenCV官方教程](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)