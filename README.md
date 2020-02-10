# Neural Network
## Deep Neural Network for C model

## Environment
- Ubuntu 18.04.1 LTS (GNU/Linux 4.15.0-34-generic x86\_64)

## Prerequisite
- g++ 7.4.0

## Compile
```shell
$ make
```

## Usage
### Train
```shell
$ ./network 0
```

### Inference
```shell
$ ./network 1
```

## Files Structure
```
./
+-- lib/
|   +-- libdqn.so
+-- mnist/
|   +-- mnist_x_test.dat 
|   +-- mnist_x_train.dat
|   +-- mnist_y_test.dat 
|   +-- mnist_y_train.dat
+-- libdqn.h
+-- libarr_2.cpp
+-- libarr.cpp
+-- libdqn.cpp
+-- libfunct.cpp
+-- libnet.cpp
+-- libpair.cpp
+-- libstruct.cpp
+-- libtuple.cpp
+-- main.cpp
+-- mnist_model_fc_001.pmt
+-- network
+-- Makefile
+-- README.md

```
- `lib/libdqn.so`: 編譯完成的 library 檔。
- `libdqn.h`: Prototype 宣告。
- `libarr_2.cpp`: Array\<T\> 物件實作。
- `libdqn.cpp`: DataReader 物件實作。
- `libnet.cpp`: Layer 、 Optimizer 、 Model 物件實作。
- `libpair.cpp`: Pair 物件實作。
- `libstruct.cpp`: FixPoint 物件實作。
- `libtuple.cpp`: Tuple 物件實作。
- `main.cpp`: 範例 main program。
- `mnist_model_fc_001.pmt`: 已訓練模型參數(Array\<T\> 檔案型式)。
- `Makefile`: 自動化腳本。

## Analysis main.cpp
- 首先 include 函式庫的 Prototype
```cpp=
#include <iostream>
#include "libdqn.h"
```

- 先宣告函式 showImg(const Array\<FixPoint\>& img, const FixPoint& threshold) 用來打印 MNIST 圖形
```cpp=
void showImg(const Array<FixPoint>& img, const FixPoint& threshold) {
  int i, j;
  cout << "   ------------------------------------------------------------" << endl;
  cout << "   |                        Show Image                        |" << endl;
  cout << "   ";
  for (i = 0; i < 30; i++) {
    for (j = 0; j < 30; j++) {
      if (i == 0 || i == 29) {
        cout << "--";
      }
      else if (j == 0) {
        cout << "| ";
      }
      else if (j == 29) {
        cout << " |";
      }
      else {
        if (img.at(0, 0, i - 1, j - 1) > threshold)
          cout << "＠";
        else
          cout << "  ";
      }
    }
    cout << endl;
    if (i != 29) cout << "   ";
  }
}
```

:::info
Array<T> 資料行別應使用 at 函式取值，如img.at(0, 0, i - 1, j - 1)
:::

- main 一開始先初始化隨機種子，並將 train 、 test 資料載入 DataReader
```cpp=
int main(int argc, char **argv) {
  rnd_init();
  try {
    int i, j;
    DataReader dr("mnist/mnist_x_train.dat",
                  "mnist/mnist_y_train.dat",
                  "mnist/mnist_x_test.dat",
                  "mnist/mnist_y_test.dat");
```

- 建構 Model
```cpp=
    // ---------- Build Model ----------
    Model model;
    model.add(new Inputs(1, 28, 28, -1));
    // model.add(new Conv2D(16, 3, 1, 1, 1)); // Conv2D(filters, kernel_size, strides, padding, bias_used);
    // model.add(new MaxPooling2D(2));        // MaxPooling2D(pooling_size);
    // model.add(new Conv2D(36, 3, 1, 1, 1)); // Conv2D(filters, kernel_size, strides, padding, bias_used);
    // model.add(new MaxPooling2D(2));        // MaxPooling2D(pooling_size);
    model.add(new Flatten());
    model.add(new Dense(128));
    model.add(new ReLU());
    model.add(new Dense(10));
    model.compile(new SoftmaxCrossEntropy(), new RMSprop(0.001));
    model.summary();
    cout << endl;
    // ---------------------------------
```

- 當程式為訓練模式，隨機抽樣0~59999等256個數字放進 batch ，當作這批次訓練資料的 index ，利用 dr.read_x_train(batch) 及 dr.read\_y\_train(batch) 來索引訓練資料及標籤，接著使用 model.fit 來更新模型，最後打印出 loss 及 accuracy
```cpp=
    // ----------- Training ------------
    if (argc == 1 || argv[1][0] == '0') {
      for (i = 0; i < 100; ++i) {
          batch = Tuple<int>::random_int(256, 60000); // random_int(length, max);
          score = model.fit(dr.read_x_train(batch), dr.read_y_train(batch));
          cout << "Epoch: " << i << ", loss: " << score[0] << ", accuracy: " << score[1] << endl;
      }
      model.save_paras("mnist_model_fc_001.pmt");
    }
    // ---------------------------------
```

- 當程式為預測模式，使用者可以輸入0~9999數字挑選任一測試圖片，利用 model.predict 函式進行預測，最後打印出預測結果及圖形
```cpp=
    // ---------- Inference ------------
    else {
      model.load_paras("mnist_model_fc_001.pmt");
      bool loop(true);
      int n;
      FixPoint threshold(0.5);
      int* acc_in, *acc_out;
      while (loop) {
        cout << "Please input a number (0 ~ 9999): ";
        cin >> n;
        cout << endl;
        if (n < 0 || n > 9999 || cin.fail()) {
          loop = false;
          cout << "Good Bye\n";
          continue;
        }
        batch      = n;
        input_data = dr.read_x_test(batch);
        pred     = model.predict(input_data).argmax(1);
        target   = dr.read_y_test(batch).argmax(1);

        // --------- Show Result ---------
        cout << "  ============== Software (Full-Connected) Finish ==============" << endl;
        cout << endl;
        cout << "  ====================== Pred: " << pred.at(0)
                                     << " True: " << target.at(0)
                                     << " =======================" << endl;
        cout << endl;
        showImg(input_data, threshold);
        cout << endl;
        // -------------------------------
        delete [] acc_in;
        //delete [] acc_out;
      }
    }
  }
  catch (const char* msg) {
    cerr << msg << endl;
    return 1;
  }
  return 0;
}
```
## Authors
[Yu-Tong Shen](https://github.com/yutongshen/)

###### tags: `Neural Network`
