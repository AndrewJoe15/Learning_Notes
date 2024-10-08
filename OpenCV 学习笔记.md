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



# 参考资料
- [OpenCV官方教程](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
