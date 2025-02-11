#include <chrono>
#include <cstdlib>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <malloc.h>
#include <pmmintrin.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <xmmintrin.h>

void normal(float *a, float *b, float *r) {
    for (int i = 0; i < 15; ++i) {
        r[i] = a[i] * b[i];
    }
}

void xmm(float *a, float *b, float *r) {
    __m512 va = _mm512_load_ps(a);
    __m512 vb = _mm512_load_ps(b);
    __m512 vr = _mm512_mul_ps(va, vb);
    std::cout << "vr is: " << vr[0] << " " << vr[1] << " " << vr[2] << " "
              << vr[3] << " " << vr[4] << " " << vr[5] << " " << vr[6] << " "
              << vr[7] << " " << vr[8] << " " << vr[9] << " " << vr[10] << " "
              << vr[11] << " " << vr[12] << " " << vr[13] << " " << vr[14]
              << " " << vr[15] << std::endl;
    _mm512_store_ps(r, vr);
}

int main() {
    alignas(64) float a[15] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                               1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0};
    alignas(64) float b[15] = {5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0,
                               5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0};
    alignas(64) float c[15];
    // auto a = (float *)std::aligned_alloc(64, 4 * sizeof(float));
    // auto b = (float *)std::aligned_alloc(64, 4 * sizeof(float));
    // auto c = (float *)std::aligned_alloc(64, 4 * sizeof(float));

    // auto now = std::chrono::high_resolution_clock::now();
    // normal(a, b, c);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration =
    //     std::chrono::duration_cast<std::chrono::nanoseconds>(end - now);
    // std::cout << "normal: " << duration.count() << " ns" << std::endl;

    // auto now = std::chrono::high_resolution_clock::now();
    xmm(a, b, c);
    for (int i = 0; i < 15; ++i) {
        std::cout << c[i] << " ";
    }
    // end = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end -
    // now); std::cout << "xmm: " << duration.count() << " ns" << std::endl;
    return 0;
}
