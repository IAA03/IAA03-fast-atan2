
/**
 * IAA03_fast_math_tests.h
 * Test methods for IAA03_fast_math
 * * Author: [Abdulrahman Alharbi / IAA03_Dev]
 * University: King Saud University (KSU)
 * Year: 2026
 * * Description:
 * A header-only C++ made to test and Verify accuracy ,
 * * License: MIT (or Apache 2.0)
 */

/* * USAGE:
 *  #define IAA03_IMPLEMENT_MATH
 *  #include "IAA03_fast_math_tests.h"
 *
 *
 * * NOTE : use "generate_atan2_test" to generate the data to do the accuracy test on, or use "generatePairs" to generate random values to speed test
 */

#pragma once
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>
#include <iostream>
#include "IAA03_fast_math.h"

volatile float global_volatile_sum = 0.0f;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define IAA03_SIMD_LEVEL 3
#include <immintrin.h>
#endif
#if defined(__AVX2__)
using simd8f = IAA03::simd8f;
using simd8func2f = simd8f (*)(simd8f, simd8f);

void print_Limits_simd8(simd8func2f atan2_target)
{
    float pz = 0.0f;
    float nz = -0.0f;
    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();
    auto test_atan_simd8 = [&](float y, float x, const char *label)
    {
        simd8f vy(y), vx(x);
        float res = atan2_target(vy, vx).extract_lane(0);
        std::cout << std::left << std::setw(20) << label
                  << " atan2(" << std::setw(4) << y << "," << std::setw(4) << x << ") = "
                  << std::setw(10) << res
                  << " [Signbit: " << (((*(uint *)(&res)) & 0x80'00'00'00) > 0) << "]" << std::endl;
    };
    std::cout << "--- ORIGIN EDGES (The 4 Zeros) ---" << std::endl;
    test_atan_simd8(pz, pz, "Top-Right (Q1)");    // Expected: 0
    test_atan_simd8(pz, nz, "Top-Left (Q2)");     // Expected: PI (3.14159)
    test_atan_simd8(nz, nz, "Bottom-Left (Q3)");  // Expected: -PI (-3.14159)
    test_atan_simd8(nz, pz, "Bottom-Right (Q4)"); // Expected: -0

    std::cout << "\n--- AXIS EDGES ---" << std::endl;
    test_atan_simd8(1.0f, pz, "North Pole");  // Expected: PI/2 (1.5708)
    test_atan_simd8(-1.0f, pz, "South Pole"); // Expected: -PI/2 (-1.5708)
    test_atan_simd8(pz, -1.0f, "West Pole");  // Expected: PI (3.14159)

    std::cout << "\n--- INFINITY OCTANTS ---" << std::endl;
    test_atan_simd8(inf, inf, "45 deg");     // Expected: 0.7853
    test_atan_simd8(inf, -inf, "135 deg");   // Expected: 2.3561
    test_atan_simd8(-inf, -inf, "-135 deg"); // Expected: -2.3561
    test_atan_simd8(-inf, inf, "-45 deg");   // Expected: -0.7853

    std::cout << "\n--- EXTREME RATIOS ---" << std::endl;
    test_atan_simd8(1e-38f, -1.0f, "Near West (Top)");     // Extremely flat Q2
    test_atan_simd8(-1e-38f, -1.0f, "Near West (Bottom)"); // Extremely flat Q3
    test_atan_simd8(inf, 1.0f, "Extreme North");           // Parallel to Y
    // Testing the 45-degree transition (y almost equals x)
    test_atan_simd8(0.999999f, 1.000000f, "Near 45 deg");
    test_atan_simd8(1.000000f, 0.999999f, "Just past 45 deg");

    // Testing the very steep/very flat edges
    test_atan_simd8(1.0f, 1e-7f, "Near Vertical");
    test_atan_simd8(1e-7f, 1.0f, "Near Horizontal");
    // 1. The "Black Hole" (0/0)
    test_atan_simd8(0.0f, 0.0f, "The Origin");

    // 2. The "Wall" (1/0)
    test_atan_simd8(1.0f, 0.0f, "North");
    // 2.2. The "Wall" (1/0)
    test_atan_simd8(-1.0f, -1e-20f, "South");
    // 3. The "Void" (inf/inf)
    test_atan_simd8(inf, inf, "Infinite Diagonal");

    // 4. The "Ghost" (-0.0)
    test_atan_simd8(-0.0f, 1.0f, "Negative Zero Y");
}

// Benchmarking function
double speed_test_simd8(const std::string &name, simd8func2f func, size_t number_of_tests,
                        const std::vector<float> &y_vals, const std::vector<float> &x_vals)
{

    auto start = std::chrono::high_resolution_clock::now();
    float sum_angles = 0.0f;
    float angles[32];
    for (size_t i = 0; i < y_vals.size(); i += 32)
    {
        func(&y_vals[i], &x_vals[i]).extract_to(angles);
        func(&y_vals[i + 8], &x_vals[i + 8]).extract_to(&angles[8]);
        func(&y_vals[i + 16], &x_vals[i + 16]).extract_to(&angles[16]);
        func(&y_vals[i + 24], &x_vals[i + 24]).extract_to(&angles[24]);
        for (int o = 0; o < 32; o++)
            sum_angles += angles[o];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    global_volatile_sum = sum_angles;
    double avrg = (duration_ms.count() * number_of_tests / y_vals.size());
    std::cout << "--- Benchmarking " << name << " ---\n";
    std::cout << "Time taken for " << y_vals.size() << " calls: "
              << duration_ms.count() << " ms\n";
    std::cout << "Average time per call: "
              << avrg << " ns\n\n";
    return avrg;
}

// Verification function
void verify_accuracy_simd8(simd8func2f func,
                           const std::vector<float> &y_vals, const std::vector<float> &x_vals)
{

    double total_abs_error = 0.0;
    float max_error = 0.0f;
    size_t num_tests = y_vals.size();
    bool skip;
    for (size_t i = 0; i < num_tests; i++)
    {
        if (y_vals[i] == 0 && x_vals[i] == 0)
            continue;
        float actual = {
            std::atan2(y_vals[i], x_vals[i])};
        float approx;
        approx = func((&y_vals[i]), (&x_vals[i])).extract_lane(0);
        float error = 0;

        error += (std::abs(actual - approx));

        total_abs_error += error;
        if (error > max_error)
        {
            max_error = error;
        }
    }

    double average_error = total_abs_error / num_tests;

    std::cout << "--- Accuracy Verification (vs std::atan2) ---\n";
    std::cout << "Max Error:   " << max_error << " radians (" << max_error * 57.2958f << " degrees)\n";
    std::cout << "Average Error: " << average_error << " radians (" << average_error * 57.2958 << " degrees)\n\n";
}

#endif
#if defined(__SSE4_1__) || defined(_M_AMD64) || defined(_M_X64)
using simd4f = IAA03::simd4f;
using simd4func2f = simd4f (*)(simd4f, simd4f);

// Benchmarking function
double speed_test_simd4(const std::string &name, simd4func2f func, size_t number_of_tests,
                        const std::vector<float> &y_vals, const std::vector<float> &x_vals)
{

    auto start = std::chrono::high_resolution_clock::now();
    float sum_angles = 0.0f;
    float angles[4];
    for (size_t i = 0; i < y_vals.size(); i += 16)
    {
        func(&y_vals[i], &x_vals[i]).extract_to(angles);
        sum_angles += angles[0] + angles[1] + angles[2] + angles[3];
        func(&y_vals[i + 4], &x_vals[i + 4]).extract_to(angles);
        sum_angles += angles[0] + angles[1] + angles[2] + angles[3];
        func(&y_vals[i + 8], &x_vals[i + 8]).extract_to(angles);
        sum_angles += angles[0] + angles[1] + angles[2] + angles[3];
        func(&y_vals[i + 12], &x_vals[i + 12]).extract_to(angles);
        sum_angles += angles[0] + angles[1] + angles[2] + angles[3];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    global_volatile_sum = sum_angles;
    double avrg = (duration_ms.count() * number_of_tests / y_vals.size());
    std::cout << "--- Benchmarking " << name << " ---\n";
    std::cout << "Time taken for " << y_vals.size() << " calls: "
              << duration_ms.count() << " ms\n";
    std::cout << "Average time per call: "
              << avrg << " ns\n\n";
    return avrg;
}

// Verification function
void verify_accuracy_simd4(simd4func2f func, const std::vector<float> &y_vals, const std::vector<float> &x_vals)
{
    float max_error = 0, total_error = 0;
    size_t num_tests = y_vals.size();
    for (size_t i = 0; i < num_tests; i++)
    {

        if (y_vals[i] == 0 && x_vals[i] == 0)
            continue;
        float error = 0;

        error += (std::abs(std::atan2(y_vals[i], x_vals[i]) - func(simd4f(y_vals[i]), simd4f(x_vals[i])).extract_lane(0)));

        total_error += error;
        if (error > max_error)
        {
            max_error = error;
        }
    }

    double average_error = total_error / num_tests;

    std::cout << "--- Accuracy Verification (vs std::atan2) ---\n";
    std::cout << "Max Error:   " << max_error << " radians (" << max_error * 57.2958f << " degrees)\n";
    std::cout << "Average Error: " << average_error << " radians (" << average_error * 57.2958 << " degrees)\n\n";
}

void print_Limits_simd4(simd4func2f atan2_target)
{
    float pz = 0.0f;
    float nz = -0.0f;
    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();
    auto test_atan_simd4 = [&](float y, float x, const char *label)
    {
        simd4f vy(y), vx(x);
        float res = atan2_target(vy, vx).extract_lane(0);
        std::cout << std::left << std::setw(20) << label
                  << " atan2(" << std::setw(4) << y << "," << std::setw(4) << x << ") = "
                  << std::setw(10) << res
                  << " [Signbit: " << (((*(uint *)(&res)) & 0x80'00'00'00) > 0) << "]" << std::endl;
    };
    std::cout << "--- ORIGIN EDGES (The 4 Zeros) ---" << std::endl;
    test_atan_simd4(pz, pz, "Top-Right (Q1)");    // Expected: 0
    test_atan_simd4(pz, nz, "Top-Left (Q2)");     // Expected: PI (3.14159)
    test_atan_simd4(nz, nz, "Bottom-Left (Q3)");  // Expected: -PI (-3.14159)
    test_atan_simd4(nz, pz, "Bottom-Right (Q4)"); // Expected: -0

    std::cout << "\n--- AXIS EDGES ---" << std::endl;
    test_atan_simd4(1.0f, pz, "North Pole");  // Expected: PI/2 (1.5708)
    test_atan_simd4(-1.0f, pz, "South Pole"); // Expected: -PI/2 (-1.5708)
    test_atan_simd4(pz, -1.0f, "West Pole");  // Expected: PI (3.14159)

    std::cout << "\n--- INFINITY OCTANTS ---" << std::endl;
    test_atan_simd4(inf, inf, "45 deg");     // Expected: 0.7853
    test_atan_simd4(inf, -inf, "135 deg");   // Expected: 2.3561
    test_atan_simd4(-inf, -inf, "-135 deg"); // Expected: -2.3561
    test_atan_simd4(-inf, inf, "-45 deg");   // Expected: -0.7853

    std::cout << "\n--- EXTREME RATIOS ---" << std::endl;
    test_atan_simd4(1e-38f, -1.0f, "Near West (Top)");     // Extremely flat Q2
    test_atan_simd4(-1e-38f, -1.0f, "Near West (Bottom)"); // Extremely flat Q3
    test_atan_simd4(inf, 1.0f, "Extreme North");           // Parallel to Y
    // Testing the 45-degree transition (y almost equals x)
    test_atan_simd4(0.999999f, 1.000000f, "Near 45 deg");
    test_atan_simd4(1.000000f, 0.999999f, "Just past 45 deg");

    // Testing the very steep/very flat edges
    test_atan_simd4(1.0f, 1e-7f, "Near Vertical");
    test_atan_simd4(1e-7f, 1.0f, "Near Horizontal");
    // 1. The "Black Hole" (0/0)
    test_atan_simd4(0.0f, 0.0f, "The Origin");

    // 2. The "Wall" (1/0)
    test_atan_simd4(1.0f, 0.0f, "North");
    // 2.2. The "Wall" (1/0)
    test_atan_simd4(-1.0f, -1e-20f, "South");
    // 3. The "Void" (inf/inf)
    test_atan_simd4(inf, inf, "Infinite Diagonal");

    // 4. The "Ghost" (-0.0)
    test_atan_simd4(-0.0f, 1.0f, "Negative Zero Y");
}
#endif
using floatfunc2f = float (*)(float, float);

void generatePairs(const int num, std::vector<float> &x, std::vector<float> &y, float min = -10000.0f, float max = 10000.0f)
{
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    // Use a large range to cover various floating point magnitudes
    std::uniform_real_distribution<float> dist(min, max);
    x.resize(num);
    y.resize(num);
    for (size_t i = 0; i < num; ++i)
    {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }
}

void print_Limits(floatfunc2f atan2_target)
{
    auto test_atan = [&](float y, float x, const char *label)
    {
        float res = atan2_target(y, x);
        std::cout << std::left << std::setw(20) << label
                  << " atan2(" << std::setw(4) << y << "," << std::setw(4) << x << ") = "
                  << std::setw(10) << res
                  << " [Signbit: " << (((*(uint *)(&res)) & 0x80'00'00'00) > 0) << "]" << std::endl;
    };

    float pz = 0.0f;
    float nz = -0.0f;
    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();

    std::cout << "--- ORIGIN EDGES (The 4 Zeros) ---" << std::endl;
    test_atan(pz, pz, "Top-Right (Q1)");    // Expected: 0
    test_atan(pz, nz, "Top-Left (Q2)");     // Expected: PI (3.14159)
    test_atan(nz, nz, "Bottom-Left (Q3)");  // Expected: -PI (-3.14159)
    test_atan(nz, pz, "Bottom-Right (Q4)"); // Expected: -0

    std::cout << "\n--- AXIS EDGES ---" << std::endl;
    test_atan(1.0f, pz, "North Pole");  // Expected: PI/2 (1.5708)
    test_atan(-1.0f, pz, "South Pole"); // Expected: -PI/2 (-1.5708)
    test_atan(pz, -1.0f, "West Pole");  // Expected: PI (3.14159)

    std::cout << "\n--- INFINITY OCTANTS ---" << std::endl;
    test_atan(inf, inf, "45 deg");     // Expected: 0.7853
    test_atan(inf, -inf, "135 deg");   // Expected: 2.3561
    test_atan(-inf, -inf, "-135 deg"); // Expected: -2.3561
    test_atan(-inf, inf, "-45 deg");   // Expected: -0.7853

    std::cout << "\n--- EXTREME RATIOS ---" << std::endl;
    test_atan(1e-38f, -1.0f, "Near West (Top)");     // Extremely flat Q2
    test_atan(-1e-38f, -1.0f, "Near West (Bottom)"); // Extremely flat Q3
    test_atan(inf, 1.0f, "Extreme North");           // Parallel to Y
    // Testing the 45-degree transition (y almost equals x)
    test_atan(0.999999f, 1.000000f, "Near 45 deg");
    test_atan(1.000000f, 0.999999f, "Just past 45 deg");

    // Testing the very steep/very flat edges
    test_atan(1.0f, 1e-7f, "Near Vertical");
    test_atan(1e-7f, 1.0f, "Near Horizontal");
    // 1. The "Black Hole" (0/0)
    test_atan(0.0f, 0.0f, "The Origin");

    // 3. The "Void" (inf/inf)
    test_atan(inf, inf, "Infinite Diagonal");
}
void verify_accuracy(floatfunc2f func,
                     const std::vector<float> &y_vals, const std::vector<float> &x_vals)
{

    double total_abs_error = 0.0;
    float max_error = 0.0f;
    size_t num_tests = y_vals.size();

    for (size_t i = 0; i < num_tests; ++i)
    {
        float y = y_vals[i];
        float x = x_vals[i];

        // Skip the origin (0,0) as atan2 is mathematically undefined there
        if (x == 0.0f && y == 0.0f)
            continue;

        float actual = std::atan2(y, x);
        float approx = func(y, x);
        float error = std::abs(actual - approx);

        total_abs_error += error;
        if (error > max_error)
        {
            max_error = error;
        }
    }

    double average_error = total_abs_error / num_tests;

    std::cout << "--- Accuracy Verification (vs std::atan2) ---\n";
    std::cout << "Max Error:   " << max_error << " radians (" << max_error * 57.2958f << " degrees)\n";
    std::cout << "Average Error: " << average_error << " radians (" << average_error * 57.2958f << " degrees)\n\n";
}
double speed_test(const std::string &name, floatfunc2f func, size_t number_of_tests,
                  const std::vector<float> &y_vals, const std::vector<float> &x_vals)
{

    auto start = std::chrono::high_resolution_clock::now();
    float sum_angles = 0.0f;
    float angle;
    for (size_t i = 0; i < y_vals.size(); i += 4)
    {
        angle = func(y_vals[i + 0], x_vals[i + 0]);
        angle += func(y_vals[i + 1], x_vals[i + 1]);
        angle += func(y_vals[i + 2], x_vals[i + 2]);
        angle += func(y_vals[i + 3], x_vals[i + 3]);
        sum_angles += angle;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;

    global_volatile_sum = sum_angles;
    double avrg = (duration_ms.count() * number_of_tests / y_vals.size());
    std::cout << "--- Benchmarking " << name << " ---\n";
    std::cout << "Time taken for " << y_vals.size() << " calls: "
              << duration_ms.count() << " ms\n";
    std::cout << "Average time per call: "
              << avrg << " ns\n\n";
    return avrg;
}
// make sure to delete them later, it just generate 360*quality_base10 ,so use that as number_of_tests
void generate_atan2_test(float **x, float **y, float **results, size_t quality_base10 = 1)
{
    const float range = 360 * quality_base10;
    const float fraction = 1 / quality_base10;
    const float degree = (M_PI / 180.0f);

    *x = new float[range];
    *y = new float[range];
    *results = new float[range];
    for (int i = 0; i < range; ++i)
    {
        float angle_deg = (float)i * fraction;

        float angle_rad = angle_deg * degree;
        *x[i] = cosf(angle_rad);
        *y[i] = sinf(angle_rad);

        *results[i] = std::atan2((long double)*y[i], (long double)*x[i]);
    }
}

// inline float fastaccurateAtan(float x)
//{
//     constexpr float piq = PI / 4;
//     constexpr float c1 = 0.37;
//     constexpr float c2 = -7.5 * 0.009;
//     constexpr float c3 = -0.175 * 0.009;
//     constexpr float c4 = -0.65 * 0.0097;
//     float ix = 1 - x;
//     float u = x * 1.3385977256f, ui = 1 - u, ui3 = ui * ui * ui,
//           t = ix * 3.9533571094f, ti = 1 - t, ti2 = ti * ti,
//           p = x * 1.25f, sp2 = fmaf(-p, p, 1);
//
//     bool cond = x > 0.747050425165f;
//     // return (x+x * ix* c1 - u * ui3 * (!cond)*c2 - t * ti2 * (cond)* c3 - x * ix2 * sp2*c4)
//     //     *piq;
//     float xix = x * ix, core = fmaf(xix, c1, x);
//     float a = cond ? t : u, b = cond ? ti2 : ui3, c = cond ? c3 : c2;
//     return (
//                fmaf(xix * ix * sp2, c4,
//                     fmaf(a * b, c,
//                          core))) *
//            piq;
// }