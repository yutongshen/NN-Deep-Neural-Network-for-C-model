#include <iostream>
#include "libdqn.h"

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
        if (img.at(0, 0, i - 1, j - 1, -1) > threshold) 
          cout << "＠";
        else
          cout << "  ";
      }
    }
    cout << endl;
    if (i != 29) cout << "   ";
  }
}

int main(int argc, char **argv) {
  rnd_init();
  try {
    int i, j;
    DataReader dr("mnist/mnist_x_train.dat",
                  "mnist/mnist_y_train.dat",
                  "mnist/mnist_x_test.dat",
                  "mnist/mnist_y_test.dat");
    Tuple<int> batch;
    Tuple<float> score;
    Array<int> pred, target;
    Array<FixPoint> input_data, conv_out;
    
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
    // ---------- Inference ------------
    else {
      model.load_paras("mnist_model_fc_001.pmt");
      bool loop(true);
      int n;
      FixPoint threshold(0.5);
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
      }
    }
    // ---------------------------------
  }
  catch (const char* msg) {
    cerr << msg << endl;
    return 1;
  }
  return 0;
}
