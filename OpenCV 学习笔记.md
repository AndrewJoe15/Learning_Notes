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

为了获取像素强度值，我们可以需要知道图像类型和通道数。

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



# 参考资料
- [OpenCV官方教程](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
