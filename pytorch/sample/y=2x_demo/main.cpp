#include <iostream>

// dataset
int total = 3;
float dataset_x[] = {1, 2, 3, 4, 5};
float dataset_y[] = {2, 4, 6, 8, 10};

// this is the model, designed by yourself
float predict(float w, float b, float x) { return w * x + b; }

float partial_w(float w, float b) {
    float sum = 0.0;
    for (int i = 0; i < total; i++) {
        float y_hat = predict(w, b, dataset_x[i]);
        sum += (y_hat - dataset_y[i]) * dataset_x[i];
    }
    return sum / total * 2;
}

float partial_b(float w, float b) {
    float sum = 0.0;
    for (int i = 0; i < total; i++) {
        float y_hat = predict(w, b, dataset_x[i]);
        sum += (y_hat - dataset_y[i]);
    }
    return sum / total * 2;
}

float loss(float w, float b) {
    float sum = 0.0;
    for (int i = 0; i < total; i++) {
        float y_hat = predict(w, b, dataset_x[i]);
        sum += (y_hat - dataset_y[i]) * (y_hat - dataset_y[i]);
    }
    return sum / total;
}

void update(float learn_rate, float &w, float &b) {
    w -= learn_rate * partial_w(w, b);
    b -= learn_rate * partial_b(w, b);
}

int main(int argc, const char **argv) {
    // init params
    float w = 0, b = 0;
    for (int i = 0; i < 30000; i++) {
        std::cout << ">>>>> iter " << i << std::endl;
        std::cout << "w: " << w << ", b: " << b << ", loss: " << loss(w, b)
                  << std::endl;
        update(0.01, w, b);
    }

    return 0;
}
