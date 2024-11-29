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

  `CV_[The number of bits per item][Signed or Unsigned][Type Prefix][The channel number]`

  `CV_8UC3`表示数值类型使用8bit无符号`char`类型，每个像素有3个数值对应3个通道。

- 使用数组构造
  ```C++
  int sz[3] = {2,2,2};
  Mat L(3,sz, CV_8UC(1), Scalar::all(0));
  ```

  对于高于二维的矩阵，可以使用上述的方法，先指定维度，再使用数组指定每个维度的尺寸。

- `cv::Mat::create`函数
  ```C++
  M.create(4,4, CV_8UC(2));
  cout << "M = "<< endl << " " << M << endl << endl;
  ```
  此种方式不能初始化矩阵值。

- MATLAB 风格的初始化器：`cv::Mat::eye``cv::Mat::zeros``cv::Mat::ones`，分别可创建单位（Identity, I -> Eye）矩阵、全1矩阵和全0矩阵。
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

在OpenCV中，遍历一副图像的所有像素主要有三种方法。我们会使用三种方法，统计每一种方法的耗时。

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

#### 更有效率的方式

我们已经了解了查找表和图像矩阵的连续存储，利用这两点，我们可以用更有效率的方式应用我们的压缩算法。

```C++
Mat& ScanImageAndReduceC(Mat& I, const uchar* const table)
{
  // accept only char type matrices
  CV_Assert(I.depth() == CV_8U);
  int channels = I.channels();
  int nRows = I.rows;
  int nCols = I.cols * channels;
  if (I.isContinuous())
  {
    nCols *= nRows;
    nRows = 1;
  }
  int i,j;
  uchar* p;
  for( i = 0; i < nRows; ++i)
  {
      p = I.ptr<uchar>(i);
    for ( j = 0; j < nCols; ++j)
    {
      p[j] = table[p[j]];
    }
  }
  return I;
}
```

这里我们基本上只获取了每一行开始的指针，并遍历每一行。在特殊的情况下，矩阵以连续方式存储，这时我们只需要获取第一行开始的指针进行遍历即可。

我们需要查找彩色图像，所以我们有三个通道，每行需要遍历三倍的元素。

有另一种方式，`Mat`对象的`data`数据成员返回第一行第一列的指针。如果图像是连续存储的，我们可以以此遍历整个数据的指针。

对于灰度图像：

```C++
uchar* p = I.data;
for( unsigned int i = 0; i < ncol*nrows; ++i)
  *p++ = table[*p];
```

这种方式得到的结果相同，但代码可读性略差。

#### 迭代器（安全）方式

在上一节中，我们负责保证遍历`uchar`字段的数目正确并跳过行间可能有的地址间隔。

迭代器的方式更安全，因为它取代了我们完成这些任务。我们需要做的只是访问图像矩阵的开头和结尾，增加开始的迭代器直到结尾。

获取迭代器指向的值需要使用`*`操作符。

```C++
Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table)
{
  // accept only char type matrices
  CV_Assert(I.depth() == CV_8U);
  const int channels = I.channels();
  switch(channels)
  {
    case 1:
    {
      MatIterator_<uchar> it, end;
      for( it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it)
        *it = table[*it];
      break;
    }
    case 3:
    {
      MatIterator_<Vec3b> it, end;
      for( it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it)
      {
        (*it)[0] = table[(*it)[0]];
        (*it)[1] = table[(*it)[1]];
        (*it)[2] = table[(*it)[2]];
      }
    }
  }
  return I;
}
```

如果是彩色图像，每一列有三个`uchar`元素。这可以看作一个短向量，在OpenCV中命名为`Vec3b`。要获取第n个子列，我们可以使用`[]`操作符。

OpenCV的迭代器遍历每一列并自动跳到下一行。因此，对于彩色图像，如果只是简单使用一个`uchar`迭代器，只能获取到蓝色通道的值。

#### 带引用返回的动态地址计算

最后一种方法，`cv::Mat::at()`，不建议用于扫描图像，它用来获取或修改图像中随机的元素。

```C++
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
{
  // accept only char type matrices
  CV_Assert(I.depth() == CV_8U);
  const int channels = I.channels();
  switch(channels)
  {
    case 1:
    {
      for( int i = 0; i < I.rows; ++i)
        for( int j = 0; j < I.cols; ++j )
          I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];
      break;
    }
    case 3:
    {
      Mat_<Vec3b> _I = I;
      for( int i = 0; i < I.rows; ++i)
        for( int j = 0; j < I.cols; ++j )
        {
          _I(i,j)[0] = table[_I(i,j)[0]];
          _I(i,j)[1] = table[_I(i,j)[1]];
          _I(i,j)[2] = table[_I(i,j)[2]];
        }
      I = _I;
      break;
    }
  }
  return I;
}
```

`cv::Mat::at()`依据输入的函数类型和坐标参数计算查询项的地址，然后返回指向它的一个引用。

出于安全，在Debug模式下会检查输入坐标是否有效且存在。

如果你需要多次查找，每次都需要输入类型和`at`关键字会很麻烦，OpenCV中有`cv::Mat_`数据类型可以解决这个问题。和`Mat`一样，需要在定义的时候指定数据类型。不过，不同的是，你可以使用`()`操作符进行快速访问。这实现的效果（运行时间也一样）和`at`一样，只是写法上更简便一些。

如上面例子中演示的，我们很容易就可以实现`cv::Mat`和`cv::Mat_`数据类型的相互转化。

#### 核心模块函数

在图像处理中，我们经常会修改给定图像的值。OpenCV核心模块提供的`cv::LUT()`函数函数来修改图像的值，不需要编写扫描图像的逻辑。

首先我们构建一个`Mat`类型的查找表：

```C++
Mat lookUpTable(1, 256, CV_8U);
uchar* p = lookUpTable.ptr();
for( int i = 0; i < 256; ++i)
  p[i] = table[i];
```

然后调用函数：

```C++
LUT(I, lookUpTable, J);
```

#### 性能差异

|方法|用时(ms)|
|:-|:-|
|有效率的方法|79.4717|
|迭代器|83.7201|
|动态地址|93.7878|
|LUT函数|32.5759|

最快的方法是LUT函数。这是因为OpenCV库是通过英特尔线程构建块启用多线程的。但是，如果需要编写一个简单的图像扫描，首选指针方法。迭代器是一种更安全的选择，但是相当慢。在调试模式下，使用动态引用访问方法进行全像扫描是成本最高的。

### 矩阵掩膜操作

掩膜操作就是：根据掩膜矩阵（或者叫核）重新计算图像每个像素的值。掩膜存储的值决定了对当前及其相邻像素影响的大小。从数学角度来看，我们的计算即使用一些特定的数值来求加权平均。

#### 测试用例

接下来我们来考虑一种图像对比度增强算法的问题。对图像的每个像素应用如下公式：
$$
I(i,j) = 5 * I(i,j) - [I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1)] \\
\leftrightarrows I(i,j) * M,  \\
M = \begin{matrix}
  i/j & -1 & 0 & +1 \\
  -1 & 0 & -1 & 0 \\
  0 & \Bigg[-1 & 5 & -1\Bigg] \\
  +1 & 0 & -1 & 0
\end{matrix}
$$

##### 代码

此代码在OpenCV源代码库的示例文件目录中： `samples/cpp/tutorial_code/core/mat_mask_operations/mat_mask_operations.cpp`

```C++
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;
static void help(char* progName)
{
    cout << endl
        <<  "This program shows how to filter images with mask: the write it yourself and the"
        << "filter2d way. " << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_path -- default lena.jpg] [G -- grayscale] "        << endl << endl;
}
void Sharpen(const Mat& myImage,Mat& Result);
int main( int argc, char* argv[])
{
    help(argv[0]);
    const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    Mat src, dst0, dst1;
    if (argc >= 3 && !strcmp("G", argv[2]))
        src = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
    else
        src = imread( samples::findFile( filename ), IMREAD_COLOR);
    if (src.empty())
    {
        cerr << "Can't open image ["  << filename << "]" << endl;
        return EXIT_FAILURE;
    }
    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow( "Input", src );
    double t = (double)getTickCount();
    Sharpen( src, dst0 );
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Hand written function time passed in seconds: " << t << endl;
    imshow( "Output", dst0 );
    waitKey();
    Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                   -1,  5, -1,
                                    0, -1,  0);
    t = (double)getTickCount();
    filter2D( src, dst1, src.depth(), kernel );
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Built-in filter2D time passed in seconds:     " << t << endl;
    imshow( "Output", dst1 );
    waitKey();
    return EXIT_SUCCESS;
}
void Sharpen(const Mat& myImage,Mat& Result)
{
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images
    const int nChannels = myImage.channels();
    Result.create(myImage.size(),myImage.type());
    for(int j = 1 ; j < myImage.rows-1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current  = myImage.ptr<uchar>(j    );
        const uchar* next     = myImage.ptr<uchar>(j + 1);
        uchar* output = Result.ptr<uchar>(j);
        for(int i= nChannels;i < nChannels*(myImage.cols-1); ++i)
        {
            output[i] = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);
        }
    }
    Result.row(0).setTo(Scalar(0));
    Result.row(Result.rows-1).setTo(Scalar(0));
    Result.col(0).setTo(Scalar(0));
    Result.col(Result.cols-1).setTo(Scalar(0));
}
```

现在我们来看看代码是如何实现的。

#### 基本的方法

```C++
void Sharpen(const Mat& myImage,Mat& Result)
{
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images
    const int nChannels = myImage.channels();
    Result.create(myImage.size(),myImage.type());
    for(int j = 1 ; j < myImage.rows-1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current  = myImage.ptr<uchar>(j    );
        const uchar* next     = myImage.ptr<uchar>(j + 1);
        uchar* output = Result.ptr<uchar>(j);
        for(int i= nChannels;i < nChannels*(myImage.cols-1); ++i)
        {
            output[i] = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);
        }
    }
    Result.row(0).setTo(Scalar(0));
    Result.row(Result.rows-1).setTo(Scalar(0));
    Result.col(0).setTo(Scalar(0));
    Result.col(Result.cols-1).setTo(Scalar(0));
}
```

首先我们确保输入图像数据是无符号字符格式的。我们使用`cv::CV_Assert`函数，如果判断为`false`将会抛出一个错误。

```C++
    CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images
```

获取通道数，我们可以知道有多少个子列需要遍历。

```C++
    const int nChannels = myImage.channels();
```

我们创建一个输出图像，它的尺寸和类型与输入图像相同。

```C++
    Result.create(myImage.size(),myImage.type());
```

我们使用纯C语言的`[]`操作符来获取像素。因为我们需要同时获取多行，所以我们每行分别取一个指针（前一行、当前行和下一行）。

```C++
    for(int j = 1 ; j < myImage.rows-1; ++j)
    {
        const uchar* previous = myImage.ptr<uchar>(j - 1);
        const uchar* current  = myImage.ptr<uchar>(j    );
        const uchar* next     = myImage.ptr<uchar>(j + 1);
        uchar* output = Result.ptr<uchar>(j);
        for(int i= nChannels;i < nChannels*(myImage.cols-1); ++i)
        {
            output[i] = saturate_cast<uchar>(5*current[i]
                         -current[i-nChannels] - current[i+nChannels] - previous[i] - next[i]);
        }
    }
```

在上面提到的公式中，图像边缘的一些像素位置是不存在的（例如（-1，-1））。比较简单的解决办法是对于边缘像素不应用公式，而是把像素值都置为0。

#### `filter2D` 函数

上面的操作在图像处理中是很常见的，OpenCV有一个函数负责掩膜操作——`filter2D()`。

首先我们需要定义一个`Mat`对象作为掩膜。

```C++
    filter2D( src, dst1, src.depth(), kernel );
```

该函数还有几个额外的参数：

- 参数5，指定掩膜的中心
- 参数6，将计算后的像素值加上一个值
- 参数7，决定对边缘像素的操作

OpenCV对`filter2D()`有所优化，因此比我们自己写的算法运算速度要快。

### 图像操作

#### 输入输出

加载图像

```C++
        Mat img = imread(filename);
        Mat img_gray = imread(filename, IMREAD_GRAYSCALE);
```

> 格式取决于开头几个字节的内容

保存图像

```C++
        imwrite(filename, img);
```

> 格式取决于扩展名

使用`cv::imdecode`和`cv::imencode`可以读取内存中而不是文件中的图像。

#### 图像基本操作

##### 获取像素值

为了获取像素强度值，我们需要知道图像类型和通道数。

对于单通道灰度图像（类型`8UC1`）：

```C++
          Scalar intensity = img.at<uchar>(y, x);
```

> 注意 `x` `y` 的顺序。

或者，我们也可以使用如下的写法：

```C++
          Scalar intensity = img.at<uchar>(Point(x, y));
```

对于三通道（BGR）图像:

```C++
          Vec3b intensity = img.at<Vec3b>(y, x);
          uchar blue = intensity.val[0];
          uchar green = intensity.val[1];
          uchar red = intensity.val[2];
```

同样地，我们也可以更改像素值：

```C++
          img.at<uchar>(y, x) = 128;
```

对于一个`Mat`对象，我们也可以用同样的方法获取一个点：

```C++
          vector<Point2f> points;
          //... fill the array
          Mat pointsMat = Mat(points);
          Point2f point = pointsMat.at<Point2f>(i, 0);
```

##### 内存管理和引用计数

在实际应用中，对于同一个数据，我们可能会有多个`Mat`实例。`Mat`使用引用计数来判断是否可以清理数据。在之前[Mat/引用计数](#引用计数)中我们已经讲到过。

##### 原始操作

矩阵中有一些用起来很方便的操作。

比如，对一个灰度图像`img`，可以用如下方法将其变为一个纯黑图像：
```C++
img = Scalar(0);
```
选择一个感兴趣区域：
```C++
Rect r(10, 10, 100, 100);
Mat smallImg = img(r);
```
彩色图像转灰度图像：
```C++
Mat img = imread("image.jpg");
Mat grey;
cvtColor(img, grey, COLOR_BGR2GRAY);
```
更改图像类型：
```C++
src.convertTo(dst, CV_32F);
```

##### 图像可视化

OpenCV提供了方便的图像可视化方法：

```C++
Mat img = imread("image.jpg");
namedWindow("image", WINDOW_AUTOSIZE);
imshow("image", img);
waitKey();
```

`waitKey()`将调用一个消息传递循环，图像窗口将等待一个任意按键被按下。
32位浮点类型的图像需要转换成8位无符号类型：
```C++
Mat img = imread("image.jpg");
Mat grey;
cvtColor(img, grey, COLOR_BGR2GRAY);
Mat sobelx;
Sobel(grey, sobelx, CV_32F, 1, 0);
double minVal, maxVal;
minMaxLoc(sobelx, &minVal, &maxVal); //find minimum and maximum intensities
Mat draw;
sobelx.convertTo(draw, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
namedWindow("image", WINDOW_AUTOSIZE);
imshow("image", draw);
waitKey();
```

### 使用OpenCV叠加图像

#### 目标

本节将要了解：

- 什么是线性叠加，为什么要用它？
- 如何使用`addWeighted()`叠加两个图像

#### 理论

> 注意
>   下面的解释来自Richard Szeliski的书《计算机视觉：算法和应用》(Computer Vision: Algorithm and Applications)。

在之前的章节，我们已经知道了一些像素算子。接下来看一个有趣的二元算子——线性叠加算子：

$$
g(x) = (1-\alpha)f_0(x) + \alpha f_1(x)
$$

使用这个算子，通过令$\alpha$从$0\rightarrow1$变化，就可以在两个图像或视频间产生交叉溶解（cross-dissolve）的效果。

#### 源码

```C++
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
 
using namespace cv;
 
// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cin;
using std::cout;
using std::endl;
 
int main( void )
{
   double alpha = 0.5; double beta; double input;
 
   Mat src1, src2, dst;
 
   cout << " Simple Linear Blender " << endl;
   cout << "-----------------------" << endl;
   cout << "* Enter alpha [0.0-1.0]: ";
   cin >> input;
 
   // We use the alpha provided by the user if it is between 0 and 1
   if( input >= 0 && input <= 1 )
     { alpha = input; }
 
   src1 = imread( samples::findFile("LinuxLogo.jpg") );
   src2 = imread( samples::findFile("WindowsLogo.jpg") );
 
   if( src1.empty() ) { cout << "Error loading src1" << endl; return EXIT_FAILURE; }
   if( src2.empty() ) { cout << "Error loading src2" << endl; return EXIT_FAILURE; }
 
   beta = ( 1.0 - alpha );
   addWeighted( src1, alpha, src2, beta, 0.0, dst);
 
   imshow( "Linear Blend", dst );
   waitKey(0);
 
   return 0;
}
```

### 调整图像对比度和亮度

#### 目标

在本节中，你将了解到：
- 怎样获取像素值
- 怎样初始化矩阵
- `cv:saturate_cast`是什么，有什么用
- 像素变换
- 调整图像亮度

#### 理论

##### 图像处理

- 图像处理算子即函数，它将一个或多个输入图像进行处理，得到一个输出图像
- 图像变换可以分为两种算子
  - 点算子（像素变换）
  - 邻域（区域）算子

##### 像素变换

- 对于这种图像处理变换，每一个输出像素值只取决于对应输入像素值（另外，也可能有一些全局参数）。
- 这种算子有亮度/对比度调整、色彩校正和变换。

#### 亮度和对比度调整

- 对于一个像素点，两个基本的操作就是乘法和加法：
$$
g(x)=\alpha f(x) + \beta
$$
- 参数$\alpha > 0$和$\beta$通常被叫做增益和偏置，有时也会说这两个参数分别控制对比度和亮度。
- 可以把$f(x)$看作原图像，$g(x)$看作输出图像，那么就有一个更直观的表达方式：
$$
g(i,j) = \alpha \cdot f(i,j) + \beta
$$
其中$i$和$j$分别代表像素的行和列索引。

#### 代码

```C++
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
 
// we're NOT "using namespace std;" here, to avoid collisions between the beta variable and std::beta in c++17
using std::cin;
using std::cout;
using std::endl;
using namespace cv;
 
int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv, "{@input | lena.jpg | input image}" );
    Mat image = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    if( image.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }
 
    Mat new_image = Mat::zeros( image.size(), image.type() );
 
    double alpha = 1.0; /*< Simple contrast control */
    int beta = 0;       /*< Simple brightness control */
 
    cout << " Basic Linear Transforms " << endl;
    cout << "-------------------------" << endl;
    cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
    cout << "* Enter the beta value [0-100]: ";    cin >> beta;
 
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                new_image.at<Vec3b>(y,x)[c] =
                  saturate_cast<uchar>( alpha*image.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }
 
    imshow("Original Image", image);
    imshow("New Image", new_image);
 
    waitKey();
    return 0;
}
```

#### 解释

- 因为$\alpha \cdot f(i,j) + \beta$的值有可能超出范围或者非整数，所以我们用`cv::saturate_cast`来确保值有效。

除了使用`for`循环来获取每个像素，我们还可以使用更简单快速的命令：

```C++
image.convertTo(new_image, -1, alpha, beta);
```

#### 实际例子

在本节，我们将把学到的应用到实践中，通过调整图像亮度和对比度来校正曝光不足的图像。我们也将了解另一种校正图像亮度的技术——伽马校正。

##### 亮度和对比度调整

增加/减少$\beta$值会提高/降低每个像素的值。像素值超过[0:255]范围的将会被赋值为边界值（小于0，赋值0；大于255，赋值255）。

![直方图](imgs/OpenCV%20学习笔记.md/2024-09-05-10-48-35.png)

直方图表示每个色阶的像素数量。较暗的图像会有更多的像素位于低色值，因此直方图在这个区域会显现出高峰。偏置值增大，直方图会向右移动，因为我们是对所有像素都加上偏置值。

参数$\alpha$调整色阶的扩展程度。如果$\alpha<1$，色阶会被压缩，图像对比度会降低。

![直方图](imgs/OpenCV%20学习笔记.md/2024-09-05-10-58-48.png)

调整$\beta$偏置值会提高亮度，但是同时图像会出现一层薄雾，因为对比度降低了。$\alpha$增益可以用来减弱这种影响，但是由于值溢出截取的原因，在原本明亮区域会失去一些细节。

##### 伽马校正

伽马校正可以用来校正亮度，这是通过在输入和映射输出之间的非线性变换实现的：

$$
O = \left( \frac{I}{255} \right)^{\gamma} \times 255
$$

因为这个关系不是非线性的，所以效果在不同像素上会不一样，取决于像素的原始值。

![](imgs/OpenCV%20学习笔记.md/2024-09-06-10-56-23.png)

当$\gamma < 1$时，原本较暗区域会变亮，直方图右移。$\gamma > 1$则相反。

##### 校正曝光不足的图像

下图经过曝光和对比度调整$\alpha=1.3$$\beta=40$。

![](imgs/OpenCV%20学习笔记.md/2024-09-06-15-07-37.png)

整体的亮度都被提高了，但是云彩过于饱和，大部分细节都丢失了。

下图使用伽马校正，$\gamma=0.4$。

![](imgs/OpenCV%20学习笔记.md/2024-09-06-15-10-08.png)

伽马校正引起的饱和问题要少很多，因为映射是非线性的，不存在前面方法的问题。

![](imgs/OpenCV%20学习笔记.md/2024-09-06-15-14-10.png)

上图比较了三幅图像的直方图。我们注意到，在原图中，大部分像素值集中在较低的部分。经过$\alpha$$\beta$调整，可以看到，在255处有一个明显的峰值，这是灰度值饱和还有向右的偏移造成的。伽马校正之后，直方图右移了，但是阴影区域的像素移动得要比明亮区域移动得多。

##### 代码

```C++
Mat lookUpTable(1, 256, CV_8U);
uchar* p = lookUpTable.ptr();
for( int i = 0; i < 256; ++i)
    p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_) * 255.0);

Mat res = img.clone();
LUT(img, lookUpTable, res);
```

### 离散傅里叶变换

通过本节，我们将回答一下几个问题：

- 什么是傅里叶变换？如何使用它？
- 在OpenCV中如何操作？
- 一些函数的使用，如：`copyMakeBorder()` , `merge()` , `dft()` , `getOptimalDFTSize()` , `log()` 和 `normalize()` 。

#### 源代码

可以在[这里](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp)下载，或在OpenCV源代码库中找到：`samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp`。

一个`dft()`的应用示例：

```C++
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
 
#include <iostream>
 
using namespace cv;
using namespace std;
 
static void help(char ** argv)
{
    cout << endl
        <<  "This program demonstrated the use of the discrete Fourier transform (DFT). " << endl
        <<  "The dft of an image is taken and it's power spectrum is displayed."  << endl << endl
        <<  "Usage:"                                                                      << endl
        << argv[0] << " [image_name -- default lena.jpg]" << endl << endl;
}
 
int main(int argc, char ** argv)
{
    help(argv);
 
    const char* filename = argc >=2 ? argv[1] : "lena.jpg";
 
    Mat I = imread( samples::findFile( filename ), IMREAD_GRAYSCALE);
    if( I.empty()){
        cout << "Error opening image" << endl;
        return EXIT_FAILURE;
    }
 
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
 
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
 
    dft(complexI, complexI);            // this way the result may fit in the source matrix
 
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
 
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
 
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
 
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
 
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
 
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
 
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
 
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
 
    imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);
    waitKey();
 
    return EXIT_SUCCESS;
}
```

#### 解释

傅里叶变换会将图像分解成正弦和余弦分量。换句话说，它会将图像从空间域转换到频率域。这个理念是，任何函数都可以用无穷多个正弦和余弦函数的和来精确近似。傅里叶变换就是实现这个过程的一种方法。二维图像的数学表示：

$$
F(k,l) = \displaystyle\sum\limits_{i=0}^{N-1}\sum\limits_{j=0}^{N-1} f(i,j)e^{-i2\pi(\frac{ki}{N}+\frac{lj}{N})} \\ 
e^{ix} = \cos{x} + i\sin {x}
$$

$f$是图像在空间域的值，$F$是图像在频率域的值。变换的结果是复数，我们可以通过实像和复像或者通过幅度图和相位图来显示这一结果。然而，在整个图像处理算法中，我们只对幅度图感兴趣，因为它包含了我们所需的关于图像几何结构的所有信息。然而，如果打算在这些形式下对图像进行一些修改，然后再重新转换，就需要同时保留这两者。

示例展示了如何计算并显示傅里叶变换的幅度图。数字图像是离散的，这意味着它们的取值来自某个特定的数值范围。例如，基本的灰度图像中的值通常在0到255之间。因此，傅里叶变换也需要是离散的，也就是离散傅里叶变换（DFT）。每当你需要从几何的角度来确定图像的结构时，就可以使用这个方法。

以下是要遵循的步骤：

##### 展开图像至最佳尺寸

DFT的性能依赖于图像尺寸。对于尺寸是2、3和5的倍数的图像，DFT通常是最快的。因此，为了达到最佳性能，通常可以通过在图像周围填充边界值来获得这样的尺寸。`getOptimalDFTSize()`函数可以返回这个最佳大小，而我们可以使用`copyMakeBorder()`函数来扩展图像的边界（附加的像素会初始化为零）。

##### 开辟内存控件

傅里叶变换的结果是复数。这意味着对于图像的每个值，结果都有两个分量值对应。此外，频域的范围远大于其空间域。因此，我们通常至少以浮点格式存储这些值。因此，我们将把输入图像转换为这种类型，并扩展一个通道来保存复数值。

##### 进行离散傅里叶变换

##### 实数值和虚数值转换为幅度

复数由实部（Re）和虚部（Im）组成。离散傅里叶变换（DFT）的结果是复数。DFT的幅度是：

$$
M = \sqrt[2]{ {Re(DFT(I))}^2 + {Im(DFT(I))}^2}
$$

##### 切换至对数尺度

结果是傅里叶系数的动态范围太大，无法在屏幕上显示。我们有一些小值和一些变化大的高值，但这样我们无法观察到。因此，高值都会显示为白点，而小值则显示为黑点。为了使用灰度值进行可视化，我们可以将线性刻度转换为对数刻度：

$$
M_1 = \log{(1 + M)}
$$

##### 裁剪和重排列

我们在第一步时扩展了图像，现在需要把这部分裁剪掉。为了方便查看，我们还需要把图像四个角重新排列，这样原点正好对应图像中心。

##### 标准化

同样为了方便显示，我们使用`cv::normalize()`把幅度值标准化到[0,1]范围内。

#### 结论

一个应用思路是确定图像中呈现的几何方向。例如，看看图像中文本是否水平？观察一些文本可以发现，对于一行文字，它的形式是水平的，而字母的形式是垂直的。在傅里叶变换的情况下，也可以看到文本片段的这两个主要组成部分。

水平文本的情况：

![](imgs/OpenCV%20学习笔记.md/2024-09-06-17-45-40.png)

旋转文本的情况：

![](imgs/OpenCV%20学习笔记.md/2024-09-06-17-46-39.png)

可以看到，频域最具影响力的分量(幅度图像上最亮的点)遵循图像上物体的几何旋转。由此，我们可以计算偏移量并执行图像旋转以纠正最终未对准的偏转量。

### 使用 XML 和 YAML 作为输入和输出文件

#### 目标

通过本节，我们将了解：

- 如何使用`YAML`或`XML`文件打印\读取文本条目到文件\OpenCV？
- 如何对OpenCV的数据结构做同样的事？
- 如何对自己的数据结构做同样的事？
- OpenCV数据结构如 `cv::FileStorage` `cv::FileNode` `cv::FileNodeIterator` 的使用。

#### 源代码

可以在[这里](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/core/file_input_output/file_input_output.cpp)下载，或在OpenCV源代码库中找到：`samples/cpp/tutorial_code/core/file_input_output/file_input_output.cpp`。

下面是一段示例代码，说明如何实现目标列表中列出的所有内容。

```C++
#include <opencv2/core.hpp>
#include <iostream>
#include <string>
 
using namespace cv;
using namespace std;
 
static void help(char** av)
{
    cout << endl
        << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl
        << "usage: "                                                                      << endl
        <<  av[0] << " outputfile.yml.gz"                                                 << endl
        << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
        << "specifying this in its extension like xml.gz yaml.gz etc... "                  << endl
        << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
        << "For example: - create a class and have it serialized"                         << endl
        << "             - use it to read and write matrices."                            << endl;
}
 
class MyData
{
public:
    MyData() : A(0), X(0), id()
    {}
    explicit MyData(int) : A(97), X(CV_PI), id("mydata1234") // explicit to avoid implicit conversion
    {}
    void write(FileStorage& fs) const                        //Write serialization for this class
    {
        fs << "{" << "A" << A << "X" << X << "id" << id << "}";
    }
    void read(const FileNode& node)                          //Read serialization for this class
    {
        A = (int)node["A"];
        X = (double)node["X"];
        id = (string)node["id"];
    }
public:   // Data Members
    int A;
    double X;
    string id;
};
 
//These write and read functions must be defined for the serialization in FileStorage to work
static void write(FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}
 
// This function will print our custom class to the console
static ostream& operator<<(ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}
 
int main(int ac, char** av)
{
    if (ac != 2)
    {
        help(av);
        return 1;
    }
 
    string filename = av[1];
    { //write
        Mat R = Mat_<uchar>::eye(3, 3),
            T = Mat_<double>::zeros(3, 1);
        MyData m(1);
 
        FileStorage fs(filename, FileStorage::WRITE);
        // or:
        // FileStorage fs;
        // fs.open(filename, FileStorage::WRITE);
 
        fs << "iterationNr" << 100;
        fs << "strings" << "[";                              // text - string sequence
        fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
        fs << "]";                                           // close sequence
 
        fs << "Mapping";                              // text - mapping
        fs << "{" << "One" << 1;
        fs <<        "Two" << 2 << "}";
 
        fs << "R" << R;                                      // cv::Mat
        fs << "T" << T;
 
        fs << "MyData" << m;                                // your own data structures
 
        fs.release();                                       // explicit close
        cout << "Write Done." << endl;
    }
 
    {//read
        cout << endl << "Reading: " << endl;
        FileStorage fs;
        fs.open(filename, FileStorage::READ);
 
        int itNr;
        //fs["iterationNr"] >> itNr;
        itNr = (int) fs["iterationNr"];
        cout << itNr;
        if (!fs.isOpened())
        {
            cerr << "Failed to open " << filename << endl;
            help(av);
            return 1;
        }
 
        FileNode n = fs["strings"];                         // Read string sequence - Get node
        if (n.type() != FileNode::SEQ)
        {
            cerr << "strings is not a sequence! FAIL" << endl;
            return 1;
        }
 
        FileNodeIterator it = n.begin(), it_end = n.end(); // Go through the node
        for (; it != it_end; ++it)
            cout << (string)*it << endl;
 
 
        n = fs["Mapping"];                                // Read mappings from a sequence
        cout << "Two  " << (int)(n["Two"]) << "; ";
        cout << "One  " << (int)(n["One"]) << endl << endl;
 
 
        MyData m;
        Mat R, T;
 
        fs["R"] >> R;                                      // Read cv::Mat
        fs["T"] >> T;
        fs["MyData"] >> m;                                 // Read your own structure_
 
        cout << endl
            << "R = " << R << endl;
        cout << "T = " << T << endl << endl;
        cout << "MyData = " << endl << m << endl << endl;
 
        //Show default behavior for non existing nodes
        cout << "Attempt to read NonExisting (should initialize the data structure with its default).";
        fs["NonExisting"] >> m;
        cout << endl << "NonExisting = " << endl << m << endl;
    }
 
    cout << endl
        << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;
 
    return 0;
}
```

#### 结果

XML:

```XML
<?xml version="1.0"?>
<opencv_storage>
<iterationNr>100</iterationNr>
<strings>
  image1.jpg Awesomeness baboon.jpg</strings>
<Mapping>
  <One>1</One>
  <Two>2</Two></Mapping>
<R type_id="opencv-matrix">
  <rows>3</rows>
  <cols>3</cols>
  <dt>u</dt>
  <data>
    1 0 0 0 1 0 0 0 1</data></R>
<T type_id="opencv-matrix">
  <rows>3</rows>
  <cols>1</cols>
  <dt>d</dt>
  <data>
    0. 0. 0.</data></T>
<MyData>
  <A>97</A>
  <X>3.1415926535897931e+000</X>
  <id>mydata1234</id></MyData>
</opencv_storage>
```

YAML:

```YAML
%YAML:1.0
iterationNr: 100
strings:
   - "image1.jpg"
   - Awesomeness
   - "baboon.jpg"
Mapping:
   One: 1
   Two: 2
R: !!opencv-matrix
   rows: 3
   cols: 3
   dt: u
   data: [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ]
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 0., 0., 0. ]
MyData:
   A: 97
   X: 3.1415926535897931e+000
   id: mydata1234
```

### 使用 `parallel_for_` 并行化代码

#### 目标

本节将介绍OpenCV`parallel_for_`框架的用法，轻松实现代码的并行执行。为了解释这一概念，我们将写一段代码来执行图像的卷积操作。

#### 预备内容

##### 并行执行框架

首先要有支持并行框架的OpenCV。在OpenCV 4.5中，有下面这些框架：

- Intel 线程构建块（第三方库，需显式激活）
- OpenMP（编译器继承，需显式激活）
- APPLE GCD（系统范围内，自动使用（仅限苹果））
- Windows 并发（运行时的一部分，自动使用（仅限 Windows - MSVC++ >= 10））
- Pthreads

第三方库需要在编译前在CMake中显式激活。

## 图像处理

### 基本绘制

#### 目标

本节我们将学习如何：

- 使用`line()`函数画线
- 使用`ellipse()`函数画椭圆
- 使用`rectangle()`函数画矩形
- 使用`circle()`函数画圆
- 使用`fillPoly()`函数填充多边形

#### OpenCV 相关理论

在本节教程中，我们会多次使用两种结构体：`cv:Point`和`cv:Scalar`。

##### Point

`Point`代表二维平面上的点，由图像的$x$和$y$坐标指定。

```C++
Point pt;
pt.x = 10;
pt.y = 8;
```

或

```C++
Point pt = Point(10, 8);
```

##### Scalar

- 代表4元素向量。在OpenCV中，Scalar类型通常用于传递像素值。
- 这本节中，我们用该类型的变量表示BGR颜色的灰度值（三个参数）。第四个参数我们不用所以不需要指定它。
- 例如：
  ```C++
  Scalar( a, b, c )
  ```

#### 代码

```C++
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
 
#define w 400
 
using namespace cv;
 
void MyEllipse( Mat img, double angle );
void MyFilledCircle( Mat img, Point center );
void MyPolygon( Mat img );
void MyLine( Mat img, Point start, Point end );
 
int main( void ){
 
 char atom_window[] = "Drawing 1: Atom";
 char rook_window[] = "Drawing 2: Rook";
 
 Mat atom_image = Mat::zeros( w, w, CV_8UC3 );
 Mat rook_image = Mat::zeros( w, w, CV_8UC3 );
 
 
 MyEllipse( atom_image, 90 );
 MyEllipse( atom_image, 0 );
 MyEllipse( atom_image, 45 );
 MyEllipse( atom_image, -45 );
 
 MyFilledCircle( atom_image, Point( w/2, w/2) );
 
 
 MyPolygon( rook_image );
 
 rectangle( rook_image,
 Point( 0, 7*w/8 ),
 Point( w, w),
 Scalar( 0, 255, 255 ),
 FILLED,
 LINE_8 );
 
 MyLine( rook_image, Point( 0, 15*w/16 ), Point( w, 15*w/16 ) );
 MyLine( rook_image, Point( w/4, 7*w/8 ), Point( w/4, w ) );
 MyLine( rook_image, Point( w/2, 7*w/8 ), Point( w/2, w ) );
 MyLine( rook_image, Point( 3*w/4, 7*w/8 ), Point( 3*w/4, w ) );
 
 imshow( atom_window, atom_image );
 moveWindow( atom_window, 0, 200 );
 imshow( rook_window, rook_image );
 moveWindow( rook_window, w, 200 );
 
 waitKey( 0 );
 return(0);
}
 
 
void MyEllipse( Mat img, double angle )
{
 int thickness = 2;
 int lineType = 8;
 
 ellipse( img,
 Point( w/2, w/2 ),
 Size( w/4, w/16 ),
 angle,
 0,
 360,
 Scalar( 255, 0, 0 ),
 thickness,
 lineType );
}
 
void MyFilledCircle( Mat img, Point center )
{
 circle( img,
 center,
 w/32,
 Scalar( 0, 0, 255 ),
 FILLED,
 LINE_8 );
}
 
void MyPolygon( Mat img )
{
 int lineType = LINE_8;
 
 Point rook_points[1][20];
 rook_points[0][0] = Point( w/4, 7*w/8 );
 rook_points[0][1] = Point( 3*w/4, 7*w/8 );
 rook_points[0][2] = Point( 3*w/4, 13*w/16 );
 rook_points[0][3] = Point( 11*w/16, 13*w/16 );
 rook_points[0][4] = Point( 19*w/32, 3*w/8 );
 rook_points[0][5] = Point( 3*w/4, 3*w/8 );
 rook_points[0][6] = Point( 3*w/4, w/8 );
 rook_points[0][7] = Point( 26*w/40, w/8 );
 rook_points[0][8] = Point( 26*w/40, w/4 );
 rook_points[0][9] = Point( 22*w/40, w/4 );
 rook_points[0][10] = Point( 22*w/40, w/8 );
 rook_points[0][11] = Point( 18*w/40, w/8 );
 rook_points[0][12] = Point( 18*w/40, w/4 );
 rook_points[0][13] = Point( 14*w/40, w/4 );
 rook_points[0][14] = Point( 14*w/40, w/8 );
 rook_points[0][15] = Point( w/4, w/8 );
 rook_points[0][16] = Point( w/4, 3*w/8 );
 rook_points[0][17] = Point( 13*w/32, 3*w/8 );
 rook_points[0][18] = Point( 5*w/16, 13*w/16 );
 rook_points[0][19] = Point( w/4, 13*w/16 );
 
 const Point* ppt[1] = { rook_points[0] };
 int npt[] = { 20 };
 
 fillPoly( img,
 ppt,
 npt,
 1,
 Scalar( 255, 255, 255 ),
 lineType );
}
 
void MyLine( Mat img, Point start, Point end )
{
 int thickness = 2;
 int lineType = LINE_8;
 
 line( img,
 start,
 end,
 Scalar( 0, 0, 0 ),
 thickness,
 lineType );
}
```

#### 解释

我们计划绘制两个例子，一个是原子示意图，一个是国际象棋的车，所以我们需要创建两幅图像和两个显示图像的窗口。

```C++
  char atom_window[] = "Drawing 1: Atom";
  char rook_window[] = "Drawing 2: Rook";
 
  Mat atom_image = Mat::zeros( w, w, CV_8UC3 );
  Mat rook_image = Mat::zeros( w, w, CV_8UC3 );
```

……

#### 结果

![](imgs/OpenCV%20学习笔记.md/2024-10-30-21-42-35.png)

### 随机生成器和OpenCV文本

#### 目标

在本节中，我们将学习如何：

- 使用*Random Number generator*（`cv:RNG`）类，以及如何从均匀分布中获取一个随机数。
- 使用`cv:putText`函数在OpenCV窗口中显示文本。

#### 代码

- 在前一节中，我们通过给定一些输入参数（如坐标、颜色、粗度等）绘制了各种各样的几何图形。
- 在本节中，我们尝试使用随机数值作为绘制参数。

### Smoothing Images

#### 目标

在本节教程中，我们将学习如何应用不同的线性滤波器来平滑图像，用到的OpenCV函数有下面几个：
- `blur()`
- `GaussionBlur()`
- `medianBlur()`
- `bilateralFilter()`

#### 理论

> 注意
> > 下面的解释来自Richard Szeliski和LearningOpenCV的《计算机视觉：算法和应用》一书。

- 平滑，也称为模糊，是一种简单而常用的图像处理操作。
- 平滑的原因有许多。在本教程中，我们将重点关注为了减少噪声而进行的平滑（其他用途将在后续教程中看到）。
- 线性滤波：
  $$
g(i,j) = \sum_{k,l} f(i+k, j+l) h(k,l)
  $$
  $h(k,l)$称为卷积核，它是滤波器系数。
- 有许多中滤波器，最常用的有：

##### 归一化块滤波器

- 最简单的滤波器。每个输出像素是其内核领域像素的平均值（所有邻域像素贡献的权重相等）
- 其核函数如下：
  $$
K = \dfrac{1}{K_{width} \cdot K_{height}} \begin{bmatrix}
        1 & 1 & 1 & ... & 1 \\
        1 & 1 & 1 & ... & 1 \\
        . & . & . & ... & 1 \\
        . & . & . & ... & 1 \\
        1 & 1 & 1 & ... & 1
       \end{bmatrix}
  $$
- 代码
  ```CPP
  void cv::blur	(
    InputArray src, // 输入数据；可以是任意通道数，独立处理；像素类型必须是CV_8U, CV_16U, CV_16S, CV_32F 或 CV_64F
    OutputArray dst,// 输出数据
    Size ksize,     // 核尺寸
    Point anchor = Point(-1,-1), // 锚点 默认核中心（-1，-1）
    int borderType = BORDER_DEFAULT // 图像外边缘填充模式，默认镜像  gfedcb|abcdefgh|gfedcba
    )	
  ```

##### 高斯滤波器

- 或许是最有用的滤波器（尽管不是最快的）。高斯滤波是通过将输入数组中的每个点与高斯核进行卷积，然后将它们全部求和以产生输出数组来完成的。
- 2D 高速函数可以表示为
  $$
G_{0}(x, y) = A  e^{ \dfrac{ -(x - \mu_{x})^{2} }{ 2\sigma^{2}_{x} } +  \dfrac{ -(y - \mu_{y})^{2} }{ 2\sigma^{2}_{y} } }
  $$
  其中$u$为均值，$\sigma^{2}$为方差。
- 代码
  ```C++
  void cv::GaussianBlur (
    InputArray src,   // 输入数据
    OutputArray dst,  // 输出数据
    Size ksize,       // 核尺寸
    double sigmaX,    // X方向标准差
    double sigmaY = 0,// Y方向标准差，XY标准差均为0则自动计算
    int borderType = BORDER_DEFAULT,  // 边缘填充类型
    AlgorithmHint hint = cv::ALGO_HINT_DEFAULT  // 算法提示，定义一些行为？？？
    )	
  ```


##### 中值滤波器

中值滤波器遍历信号的每个元素（在本例中是图像），并用相邻像素的中值替换每个像素（位于求值像素周围的正方形邻域中）。


- 代码
  ```C++
  void cv::medianBlur (
    InputArray src, // 	input 1-, 3-, or 4-channel image
    OutputArray dst,
    int ksize 
    )	
  ```

##### 双边滤波器

- 到目前为止，我们已经解释了一些滤波器，其主要目标是平滑输入图像。然而，有时滤波器不仅可以消除噪声，还可以平滑边缘。为了避免这种情况（至少在一定程度上），我们可以使用双边过滤器。
- 与高斯滤波器类似，双边滤波器也考虑相邻像素，并为每个像素分配权重。这些权重有两个组成部分，其中第一个是高斯滤波器使用的相同权重。第二个分量考虑相邻像素与求值像素之间的强度差异。
- 代码
  ```C++
  void cv::bilateralFilter (
    InputArray src, // 	Source 8-bit or floating-point, 1-channel or 3-channel image.
    OutputArray dst,
    int d,          // 像素领域直径
    double sigmaColor,  // 颜色空间标准差
    double sigmaSpace,  // 坐标空间标准差
    int borderType = BORDER_DEFAULT 
    )	
  ```

### 膨胀和腐蚀

#### 目标

- 学会使用两个十分常用的形态学算子：膨胀和腐蚀。
在OpenCV中的函数为：
  - `cv::erode`
  - `cv::dilate`

#### 形态学处理

- 总的来说，就是一系列基于形状的图像处理操作。形态学处理将一个结构元素应用于输入图像并生成输出图像。
- 最基本的形态学处理：膨胀和腐蚀。他们有很多用处，例如：
  - 去除噪点
  - 分离出单个元素；连接分开的元素
  - 寻找亮度凸块或孔洞
- 我们使用下面的图片，简单解释一下腐蚀和膨胀
![](imgs/OpenCV%20学习笔记.md/2024-11-26-15-32-27.png)

##### 膨胀

- 卷积核扫描图像，每个像素取邻域中的最大值，图像中较亮的区域将会“生长”（所以叫膨胀）。
- 用公式表示：$\texttt{dst} (x,y) =  \max _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')$
- 上图应用膨胀操作后：
  ![](imgs/OpenCV%20学习笔记.md/2024-11-26-15-42-00.png)
- 代码
  ```C++
  void Dilation(int, void*)
  {
      int dilation_type = 0;
      if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
      else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
      else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

      Mat element = getStructuringElement(dilation_type,
          Size(2 * dilation_size + 1, 2 * dilation_size + 1),
          Point(dilation_size, dilation_size));

      dilate(src, dilation_dst, element);
      imshow("Dilation Demo", dilation_dst);
  }
  ```

##### 腐蚀

- 与膨胀相反，腐蚀计算给定核区域的最小值。
- 用公式表示：$\texttt{dst} (x,y) =  \min _{(x',y'):  \, \texttt{element} (x',y') \ne0 } \texttt{src} (x+x',y+y')$
- 同样地，我们将原图进行腐蚀操作，可以发现字母部分变细了。
  ![](imgs/OpenCV%20学习笔记.md/2024-11-26-15-55-56.png)
- 代码
  ```C++
  void Erosion(int, void*)
  {
      int erosion_type = 0;
      if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
      else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
      else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

      Mat element = getStructuringElement(erosion_type,
          Size(2 * erosion_size + 1, 2 * erosion_size + 1),
          Point(erosion_size, erosion_size));

      erode(src, erosion_dst, element);
      imshow("Erosion Demo", erosion_dst);
  }
  ```

### 其他形态学变换

#### 目标

在本章节，你将学习如何：
- 使用OpenCV函数`cv:morphologyEx`来实现形态学变换，如：
  - 开运算
  - 闭运算
  - 形态学梯度
  - 顶帽
  - 黑帽

#### 理论

在前一节，我们介绍了两种基本的形态学操作：膨胀和腐蚀。
基于这两个操作，我们可以实现更复杂的变换。这里我们简短地讨论OpenCV中提供的五个操作。

##### 开运算

- 先腐蚀后膨胀
  $$
  dst = open( src, element) = dilate( erode( src, element ) )
  $$
- 可用于去除小元素（假定这些小元素亮于背景）
- 如下图，在进行开运算后，左边的小颗粒被去除了
  ![](imgs/OpenCV%20学习笔记.md/2024-11-26-16-45-09.png)

##### 闭运算

- 先膨胀后腐蚀
  $$
  dst = close( src, element ) = erode( dilate( src, element ) )
  $$
- 用于去除小孔洞
  ![](imgs/OpenCV%20学习笔记.md/2024-11-26-16-46-43.png)

##### 形态学梯度

- 膨胀与腐蚀的差
  $$
  dst = morph_{grad}( src, element ) = dilate( src, element ) - erode( src, element )
  $$
- 可用于寻找一个目标的外轮廓
  ![](imgs/OpenCV%20学习笔记.md/2024-11-26-16-49-11.png)

##### 顶帽

- 图像与其开运算的差
  $$
  dst = tophat( src, element ) = src - open( src, element )
  $$
  ![](imgs/OpenCV%20学习笔记.md/2024-11-26-16-51-33.png)

##### 黑帽

- 图像闭运算与原图像的差
  $$
  dst = blackhat( src, element ) = close( src, element ) - src
  $$
  ![](imgs/OpenCV%20学习笔记.md/2024-11-26-17-23-26.png)

#### 代码

```C++
void Morphology_Operations( int, void* )
{
  // Since MORPH_X : 2,3,4,5 and 6
  int operation = morph_operator + 2;
 
  Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
 
  morphologyEx( src, dst, operation, element );
  imshow( window_name, dst );
}
```

其中关键函数是`cv::morphologyEX`：

```C++
void cv::morphologyEx ( 
  InputArray src,
  OutputArray dst,
  int op,
  InputArray kernel,
  Point anchor = Point(-1,-1),
  int iterations = 1,
  int borderType = BORDER_CONSTANT,
  const Scalar & borderValue = morphologyDefaultBorderValue() 
  )	
```

参数：
- `src`         输入
- `dst`         输出
- `op`          操作类型，0-腐蚀 1-膨胀 2-开 3-闭 4-梯度 5-顶帽 6-黑帽 7-Hit-or-Miss
- `kernel`      结构元素
- `anchor`      核锚点
- `iterations`  腐蚀或膨胀的迭代次数
- `borderType`  边界类型
- `borderValue` 常量边界时的边界值

> **注意**
> 迭代次数指的是腐蚀或膨胀执行的次数，例如，迭代次数为1时的开运算：腐蚀->膨胀，迭代次数为2时的开运算：腐蚀->腐蚀->膨胀->膨胀，而不是腐蚀->膨胀->腐蚀->膨胀。

当参数`op`为7时，函数执行的是Hit-or-Miss变换，将在下一节进行介绍。

### Hit-or-Miss

#### 目标

在本教程中，您将学习如何通过使用“命中-未命中变换”（也称为“命中与未命中变换”）在二值图像中找到给定的配置或模式。该变换也是更高级形态学操作（如细化或修剪）的基础。

#### 理论

“命中或未命中变换（Hit-or-Miss transformation）用于在二进制图像中寻找图案。特别是，它找到那些其邻域与第一个结构元素 \( B_1 \) 形状匹配，但同时又不与第二个结构元素 \( B_2 \) 形状匹配的像素。数学上，作用在图像 \( A \) 上的操作可以表达如下：
$$
A\circledast B = (A\ominus B_1) \cap (A^c\ominus B_2)
$$

所以，Hit-or-Miss 操作可以分为三步：

1. 用结构元素$B_1$腐蚀图像$A$
2. 用结构元素$B_2$腐蚀图像$A$的补集$A^c$
3. 第一步和第二步的结果取交集

结构元素$B_1$和$B_2$可以合并成一个元素$B$。例如：
![](imgs/OpenCV%20学习笔记.md/2024-11-27-11-50-28.png)

在这种情况下，我们要查找的图案是：中心像素为背景色，东西南北方向像素为前景色，其他像素任意。

把这个结构元素应用到下面的图像：
![](imgs/OpenCV%20学习笔记.md/2024-11-27-11-54-34.png)
得到结果：
![](imgs/OpenCV%20学习笔记.md/2024-11-27-11-55-04.png)

### 提取水平和垂直线

#### 目标

在本教程中，您将学习如何：

- 应用两个非常常见的形态学操作（即膨胀和腐蚀），通过创建自定义内核，以提取水平轴和垂直轴上的直线。

#### 理论

##### 形态学处理

形态学是一组图像处理操作，这些操作基于预定义的结构元素（也称为内核）对图像进行处理。输出图像中每个像素的值是通过将输入图像中相应像素与其邻域进行比较得出的。通过选择内核的大小和形状，可以构建一种形态学操作，使其对输入图像中特定的形状更为敏感。

最基本的两种形态学操作是膨胀和腐蚀。膨胀操作向图像中物体的边界添加像素，而腐蚀则恰好相反，它去除边界上的像素。添加或去除的像素数量，分别取决于用于处理图像的结构元素的大小和形状。一般来说，这两种操作遵循如下规则：

- **膨胀**：输出像素的值是所有落在结构元素大小和形状范围内的像素值的最大值。例如，在二值图像中，如果输入图像中有任何像素位于内核范围内且其值为1，则输出图像中对应的像素也会被设置为1。这一规则适用于任何类型的图像（如灰度图像、BGR图像等）。
  - 二值图像膨胀
    ![](imgs/OpenCV%20学习笔记.md/2024-11-27-16-39-56.png)
  - 灰度图像膨胀
    ![](imgs/OpenCV%20学习笔记.md/2024-11-27-16-40-30.png)
- **腐蚀**：对于腐蚀操作，情况正好相反。输出像素的值是所有落在结构元素大小和形状范围内的像素值中的最小值。请参阅下面的示例图：
  - 二值图像腐蚀
    ![](imgs/OpenCV%20学习笔记.md/2024-11-27-16-42-07.png)
  - 灰度图像腐蚀
    ![](imgs/OpenCV%20学习笔记.md/2024-11-27-16-42-10.png)

##### 结构元素  
如上所述，通常在任何形态学操作中，用于探测输入图像的结构元素是最重要的部分。  

结构元素是一个仅由0和1组成的矩阵，可以具有任意的形状和大小。通常，结构元素比待处理的图像小得多，其中值为1的像素定义了邻域范围。结构元素的中心像素被称为“原点”，用于标识感兴趣的像素，即当前正在处理的像素。  

例如，下图展示了一个大小为 7x7 的菱形结构元素：
![](imgs/OpenCV%20学习笔记.md/2024-11-27-16-45-07.png)

结构元素可以具有多种常见形状，例如直线、菱形、圆盘、周期性直线和圆形，并且可以有不同的大小。通常，您会选择与输入图像中想要处理或提取的目标对象具有相同大小和形状的结构元素。例如，要在图像中找到直线，可以创建一个线性结构元素，稍后您将会看到具体示例。

#### 关键代码

本节示例中，我们将从下图曲谱中提取音符和五线格。
![](imgs/OpenCV%20学习笔记.md/2024-11-28-13-45-36.png)

代码全文在[这里](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/cpp/tutorial_code/ImgProc/morph_lines_detection/Morphology_3.cpp)

##### 灰度图

```C++
    // Transform source image to gray if it is not already
    Mat gray;
 
    if (src.channels() == 3)
    {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = src;
    }
```

##### 灰度图转二值图

```C++
    // Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    Mat bw;
    adaptiveThreshold(~gray, bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
```

这里用到了`cv::adaptiveThreshold`函数：

```C++

void cv::adaptiveThreshold (
  InputArray src,
  OutputArray dst,
  double maxValue,
  int adaptiveMethod,
  int thresholdType,
  int blockSize,
  double C 
  )	
```

这个函数将灰度图像转为二值图像，其原理依照下面的公式：
- **THRESH_BINARY**
  $$
  dst(x, y)=\left\{\begin{array}{ll}
  \operatorname{maxValue} & \text { if } \operatorname{src}(x, y)>T(x, y) \\
  0 & \text { otherwise }
  \end{array}\right.
  $$
- **THRESH_BINARY_INV**
  $$
  dst(x, y)=\left\{\begin{array}{ll}
  0 & \text { if } \operatorname{src}(x, y)>T(x, y) \\
  \operatorname{maxValue} & \text { otherwise }
  \end{array}\right.
  $$

其中$T(x,y)$是针对每个像素单独计算出来的阈值（见参数`adaptiveMethod`）。

**参数**
- `src` 输入图像，8-bit单通道
- `dst` 输出图像
- `maxValue`  最大值，即赋给非零像素的值
- `adaptiveMethod`  自适应阈值算法，包括
  - 均值法`ADAPTIVE_THRESH_MEAN_C`
  - 高斯法（邻域加权求平均）`ADAPTIVE_THRESH_GAUSSIAN_C `
- `thresholdType` 阈值类型，通常选择 `THRESH_BINARY` 或 `THRESH_BINARY_INV`
- `blockSize` 邻域区域的大小，必须为奇数
- `C` 常数，用来调节阈值计算的严格程度。即`adaptiveMethod`中需要用到的常数

##### 结构元素

为了提取我们想要的目标对象，我们需要创建一个相应的结构元素。由于我们想要提取水平线，因此用于此目的的结构元素应具有以下形状：
![](imgs/OpenCV%20学习笔记.md/2024-11-28-13-41-06.png)

在代码中我们是这样实现的：

```C++
    // Specify size on horizontal axis
    int horizontal_size = horizontal.cols / 30;
 
    // Create structure element for extracting horizontal lines through morphology operations
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
 
    // Apply morphology operations
    erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
 
    // Show extracted horizontal lines
    show_wait_destroy("horizontal", horizontal);
```

得到效果：
![](imgs/OpenCV%20学习笔记.md/2024-11-28-13-47-27.png)

同样地，针对竖直的线条，我们可以创建相应的结构元素：
![](imgs/OpenCV%20学习笔记.md/2024-11-28-13-46-49.png)

对应代码：
```C++
    // Specify size on vertical axis
    int vertical_size = vertical.rows / 30;
 
    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
 
    // Apply morphology operations
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));
 
    // Show extracted vertical lines
    show_wait_destroy("vertical", vertical);
```

得到效果：
![](imgs/OpenCV%20学习笔记.md/2024-11-28-13-48-15.png)

##### 优化边缘

正如您所见，我们已经接近目标。然而，此时您会注意到音符的边缘有些粗糙。出于这个原因，我们需要对边缘进行优化，以获得更平滑的结果：

```C++
    // Inverse vertical image
    bitwise_not(vertical, vertical);
    show_wait_destroy("vertical_bit", vertical);
 
    // Extract edges and smooth image according to the logic
    // 1. extract edges
    // 2. dilate(edges)
    // 3. src.copyTo(smooth)
    // 4. blur smooth img
    // 5. smooth.copyTo(src, edges)
 
    // Step 1
    Mat edges;
    adaptiveThreshold(vertical, edges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
    show_wait_destroy("edges", edges);
 
    // Step 2
    Mat kernel = Mat::ones(2, 2, CV_8UC1);
    dilate(edges, edges, kernel);
    show_wait_destroy("dilate", edges);
 
    // Step 3
    Mat smooth;
    vertical.copyTo(smooth);
 
    // Step 4
    blur(smooth, smooth, Size(2, 2));
 
    // Step 5
    smooth.copyTo(vertical, edges);
 
    // Show final result
    show_wait_destroy("smooth - final", vertical);
```

其中步骤1和2创建膨胀蒙版，步骤3和4平滑图像，步骤5将膨胀蒙版下的平滑图像拷贝到目标图像。

### 图像金字塔

#### 目标

本节我们将学习如何使用OpenCV的函数`pyrUp()`和`pyrDown()`来对给定的图像进行下采样和上采样。

#### 理论

- 有时我们需要改变图像的尺寸，放大（Zoom in）或者缩小（Zoom out）。
- 尽管 OpenCV 提供了一个几何变换函数 `resize` 来调整图像大小（我们将在后续教程中展示），在本节中，我们先分析**图像金字塔（Image Pyramids）**的使用方法。这种技术广泛应用于各种计算机视觉任务中。

#### 图像金字塔

图像金字塔有两种常见类型：  
- **高斯金字塔（Gaussian Pyramid）**：用于对图像进行下采样（降低分辨率）。  
- **拉普拉斯金字塔（Laplacian Pyramid）**：用于从金字塔中较低层级（分辨率较低）的图像重建上采样图像。

在本教程中，我们将使用高斯金字塔。

#### 高斯金字塔

- 想象金字塔有很多层，越往上层尺寸越小。
  ![](imgs/OpenCV%20学习笔记.md/2024-11-28-14-50-08.png)
- 从下往上数层数，所以第$(i+1)$层（记为$G_{i+1}$）比第$i$层($G_i$)更小。
- 要生成高斯金字塔的第$(i+1)$层，我们进行如下操作：
  - 用高斯卷积核对$G_i$进行卷积，高斯核如下
    $$
    \frac{1}{256} \begin{bmatrix} 1 & 4 & 6 & 4 & 1  \\ 4 & 16 & 24 & 16 & 4  \\ 6 & 24 & 36 & 24 & 6  \\ 4 & 16 & 24 & 16 & 4  \\ 1 & 4 & 6 & 4 & 1 \end{bmatrix}
    $$
  - 移除所有奇数行和奇数列
- 您可以很容易地注意到，生成的图像面积将恰好是其前一层图像的四分之一。对输入图像（原始图像）重复这一过程即可生成整个金字塔结构。
- 上述过程对图像进行下采样非常有用。如果我们想将图像放大，该如何操作呢？以下是步骤：
  - 将图像尺寸放大：将图像在每个维度上放大到原来的两倍。放大后，新增的偶数行和偶数列会用零填充，形成一个稀疏的网格。
  - 对放大的图像进行卷积：使用之前提到的核，但乘以一个因子 4，用于近似计算“缺失像素”的值。
- 上述两种操作（下采样和上采样）已在 OpenCV 中通过函数 `pyrDown()` 和 `pyrUp()` 实现。

#### 代码

- 上采样
  ```C++
  pyrUp( src, src, Size( src.cols*2, src.rows*2 ) );
  ```
- 下采样
  ```C++
  pyrDown( src, src, Size( src.cols/2, src.rows/2 ) );
  ```

参数：
- 输入
- 输出
- 目标尺寸

### 基本阈值处理

#### 目标

- 使用`cv::threshold`执行基本阈值处理操作

#### 理论

##### 阈值处理

- 最简单的分割方法  
- 应用示例：将图像中与我们需要分析的对象对应的区域分离出来。这种分离基于对象像素与背景像素之间的强度差异。  
- 为了将我们感兴趣的像素与其他像素区分开（其他像素最终会被排除），我们会将每个像素的强度值与一个阈值进行比较（阈值根据要解决的问题确定）。  
- 一旦成功分离出重要像素，我们可以为这些像素设置一个确定的值来标识它们（例如，可以将它们设置为 \(0\)（黑色）、\(255\)（白色）或任何适合需求的值）。

###### 阈值类型

- OpenCV 提供了函数 `cv::threshold` 来执行阈值操作。
- 我们可以使用此函数实现 5 种类型的阈值处理。接下来的小节将对此进行详细解释。
- 为了说明这些阈值处理的工作原理，考虑我们有一张源图像，其像素强度值为 \( src(x, y) \)。下面的图示中，水平蓝线表示固定的阈值 \( thresh \)。
  ![](imgs/OpenCV%20学习笔记.md/2024-11-28-15-57-24.png)

**Threshold Binary**

- 这种阈值处理操作可以表示为：
  $$
  dst(x, y)=\left\{\begin{array}{ll}
  0 & \text { if } \operatorname{src}(x, y)>\text{threshold} \\
  \operatorname{maxValue} & \text { otherwise }
  \end{array}\right.
  $$
  ![](imgs/OpenCV%20学习笔记.md/2024-11-28-16-00-38.png)

**Threshold Binary, Inverted**

- 这种阈值处理操作可以表示为：
  $$
  dst(x, y)=\left\{\begin{array}{ll}
  0 & \text { if } \operatorname{src}(x, y)>\text{threshold} \\
  \operatorname{maxValue} & \text { otherwise }
  \end{array}\right.
  $$
  ![](imgs/OpenCV%20学习笔记.md/2024-11-28-16-02-43.png)

**Truncate**

- 这种阈值处理操作可以表示为：
  $$
  dst(x, y)=\left\{\begin{array}{ll}
  \text{threshold} & \text { if } \operatorname{src}(x, y)>\text{threshold} \\
  \operatorname{src}(x, y) & \text { otherwise }
  \end{array}\right.
  $$
  ![](imgs/OpenCV%20学习笔记.md/2024-11-28-16-05-12.png)

**Threshold to Zero**

- 这种阈值处理操作可以表示为：
  $$
  dst(x, y)=\left\{\begin{array}{ll}
  \operatorname{src}(x, y) & \text { if } \operatorname{src}(x, y)>\text{threshold} \\
  0 & \text { otherwise }
  \end{array}\right.
  $$
  ![](imgs/OpenCV%20学习笔记.md/2024-11-28-16-07-06.png)

**Threshold to Zero, Inverted**

- 这种阈值处理操作可以表示为：
  $$
  dst(x, y)=\left\{\begin{array}{ll}
  0 & \text { if } \operatorname{src}(x, y)>\text{threshold} \\
  \operatorname{src}(x, y) & \text { otherwise }
  \end{array}\right.
  $$
  ![](imgs/OpenCV%20学习笔记.md/2024-11-28-16-10-11.png)

#### 代码

```C++
double cv::threshold(
  InputArray src,
  OutputArray dst,
  double thresh,
  double maxVal,
  int type 
  )	
```

参数

- `src` 输入，8-bit 或 32-bit 多通道图像
- `dst` 输出
- `thresh` 阈值
- `maxVal` 非零像素值，`THRESH_BINARY`和`THRESH_BINARY_INV`时使用
- `type`  阈值类型

### 使用`inRange`进行阈值处理

#### 目标

在本节中你将学习如何：

- 使用 OpenCV 的`cv::inRange`函数执行基本的阈值处理
- 在 HSV 色彩空间中基于像素值范围检测对象

#### 理论

- 在之前的教程中，我们学习了如何使用 `cv::threshold` 函数执行阈值处理。  
- 在本教程中，我们将学习如何使用 `cv::inRange` 函数来实现这一操作。  
- 概念仍然相同，但现在我们需要指定一个像素值范围。

#### HSV 色彩空间

HSV（色相、饱和度、明度）色彩空间是一种类似于 RGB 色彩模型的颜色表示模型。

- **色相（Hue）**：表示颜色的类型，因此在需要根据颜色分割对象的图像处理任务中非常有用。    
- **饱和度（Saturation）**：表示颜色的纯度，从不饱和（表示灰色阴影）到完全饱和（无白色成分）。  
- **明度（Value）**：描述颜色的亮度或强度。  

下图展示了 HSV 色彩空间的圆柱体模型。
![](imgs/OpenCV%20学习笔记.md/2024-11-29-09-33-00.png)

由于 RGB 色彩空间中的颜色是通过三个通道编码的，因此根据颜色对图像中的对象进行分割会更困难一些。
![](imgs/OpenCV%20学习笔记.md/2024-11-29-09-34-19.png)

#### 代码

完整代码可以在[这里](https://github.com/opencv/opencv/tree/4.x/samples/cpp/tutorial_code/ImgProc/Threshold_inRange.cpp)获取。

- 通过默认或提供的设备捕获视频流
  ```C++
   VideoCapture cap(argc > 1 ? atoi(argv[1]) : 0);

   ...
   while(true){
    cap >> frame;
    if(frame.empty())
    {
        break;
    }
    ...
   }
  ```
- 转换色彩空间
  ```C++
  // Convert from BGR to HSV colorspace
  cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
  ```
- 使用`inRange`函数进行阈值操作
  ```C++
  // Detect the object based on HSV Range Values
  inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);
  ```



# 参考资料
- [OpenCV官方教程](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
