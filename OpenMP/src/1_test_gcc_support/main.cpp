#include <iostream>
#include <omp.h>

int main(int argc, char *argv[]) {

    auto thread_count = std::stoi(argv[1]);
    int j = 0;

#pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < 10000; ++i) {
        auto tid = omp_get_thread_num();
        auto nthreads = omp_get_num_threads();

        // 同一时刻只有一个线程在执行 j++ 操作
#pragma omp atomic
        j++;
    }
    std::cout << j << std::endl;

    return 0;
}
