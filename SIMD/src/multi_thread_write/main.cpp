#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <x86intrin.h>

void part_1() {
    std::vector<uint64_t> arr(1000000, 0);
    for (int i = 0; i < 1000000; i++) {
        arr[i] = i;
    }
}

void part_1_simd() {
    std::vector<uint64_t> arr(1000000, 0);

    for (int i = 0; i < 1000000; i += 8) {
        __m512i v = _mm512_set_epi64(i, i + 1, i + 2, i + 3, i + 4, i + 5,
                                     i + 6, i + 7);
        _mm512_storeu_si512((__m512i *)&arr[i], v);
    }
}

void part_2() {
    std::vector<uint64_t> arr(1000000, 0);
    std::vector<std::thread> threads;
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([i, &arr]() {
            for (int j = i; j < 1000000; j += 2) {
                arr[j] = j;
            }
        });
    }
    for (auto &t : threads) {
        t.join();
    }
}

int main() {
    auto now = std::chrono::high_resolution_clock::now();
    part_1();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - now);
    std::cout << "part_1: " << duration.count() << " ns" << std::endl;

    now = std::chrono::high_resolution_clock::now();
    part_1_simd();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - now);
    std::cout << "part_1_simd: " << duration.count() << " ns" << std::endl;

    now = std::chrono::high_resolution_clock::now();
    part_2();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - now);
    std::cout << "part_2: " << duration.count() << " ns" << std::endl;

    return 0;
}
