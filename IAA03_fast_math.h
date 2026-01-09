/**
 * IAA03_fast_math.h
 * High-Performance Branchless Trigonometry Kernel
 * * Author: [Abdulrahman Alharbi / IAA03_Dev]
 * University: King Saud University (KSU)
 * Year: 2026
 * * Description:
 * A header-only C++ library utilizing Branchless Octant Folding,
 * ILP, and SIMD (AVX2/SSE4.1) to achieve up to 178x throughput
 * speedup over std::atan2.
 * - Also contain SSE and AVX2 wrappers and some float helpers
 * * License: MIT (or Apache 2.0)
 */

/* * USAGE:
 * #define IAA03_IMPLEMENT_MATH
 * #include "IAA03_fast_math.h"
 * * float result = IAA03::atan2_fast(y, x);
 * * float x[8],y[8],results[8]; IAA03::atan2_accurate(IAA03::simd8f(y),IAA03::simd8f(x)).extract_to(results);
 */

#if !defined(IAA03_FAST_MATH_H)
#define IAA03_FAST_MATH_H
#include <immintrin.h>
namespace IAA03
{

    constexpr double PI = 3.1415926535897935;
    constexpr double PIinv = 1.l / 3.1415926535897935;
    constexpr double deg = 57.29577951308232;
    constexpr double rad = 1.7453292519943295e-2;
    constexpr double PIhalf = PI / 2.0;
    constexpr double PIquadrant = PI / 4.0;
#ifndef uint
    typedef unsigned int uint;
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define IAA03_SIMD_LEVEL 3

#endif
#if defined(__AVX2__)

    struct simd8f
    {
        __m256 data;
        // 1. Default: Initialize all 8 lanes to 0.0f
        simd8f();
        // 2. Broadcast: Initialize all 8 lanes to the same value
        simd8f(float f);
        explicit simd8f(uint hex_float);
        // 3. Specific Lanes: Initialize each of the 8 lanes individually
        // Note: _mm256_set_ps takes arguments in REVERSE order (f8 is first lane in memory)
        simd8f(float f1, float f2, float f3, float f4,
               float f5, float f6, float f7, float f8);
        // 4. Pointer Load: Load 8 contiguous floats from memory
        simd8f(const float *fs);

        // 5. Internal: Initialize from an existing raw __m256 register
        simd8f(__m256 d);
        inline operator __m256() const noexcept;
        // Bitwise ABS
        inline simd8f abs() const noexcept; // Bitwise Infinity test: Exponent all 1s, Mantissa all 0s
        inline simd8f isinf() const noexcept;

        // Bitwise Unordered: The standard "Self-Compare" NaN test
        inline simd8f isnan() const noexcept;
        // Negation -x
        inline simd8f operator-() const noexcept;

        // Math Ops: +, -, *, /
        inline friend simd8f operator+(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator-(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator*(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator/(simd8f a, simd8f b) noexcept;

        // FMAF: (a * b) + c
        static inline simd8f fma(simd8f a, simd8f b, simd8f c) noexcept;
        inline friend simd8f operator<(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator>(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator<=(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator>=(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator==(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator!=(simd8f a, simd8f b) noexcept;

        // Bool Ops: AND, OR, XOR, NOT
        inline friend simd8f operator&(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator|(simd8f a, simd8f b) noexcept;
        inline friend simd8f operator^(simd8f a, simd8f b) noexcept;
        inline simd8f operator!() const noexcept;

        inline static simd8f andNot(simd8f a, simd8f b) noexcept;

        // Returns the raw sign bit (0x80000000)
        inline simd8f signbit() const noexcept;

        // Returns a TRUE MASK (-nan) if negative
        inline simd8f isneg() const noexcept;

        // Compares the bit-pattern of 'data' against the 256-bit integer pattern 'i'
        inline simd8f bittest_eq(__m256i i) const noexcept;
        inline simd8f copysign(simd8f value, simd8f target_sign) noexcept;
        inline simd8f copysign(simd8f target_sign) noexcept;

        // Select (The ?: operator)
        // selector is a mask from a comparison
        inline static simd8f select(simd8f mask, simd8f if_true, simd8f if_false) noexcept;
        inline float extract_lane(int lane);

        inline void extract_to(float *fs) noexcept;

        // Loads 8 floats
        inline void load(const float *fs) noexcept;

        inline void set_all_float(float f) noexcept;
    };

    inline simd8f atan2_fast(simd8f y, simd8f x) noexcept;
    inline simd8f atan2_accurate(simd8f y, simd8f x) noexcept;

#endif
#if defined(__SSE4_1__) || defined(_M_AMD64) || defined(_M_X64)
    struct simd4f
    {
        __m128 data;

        // Constant Load
        simd4f();
        simd4f(float f);
        simd4f(float f1, float f2, float f3, float f4);
        simd4f(const float *fs);
        simd4f(__m128 m);
        inline operator __m128() const noexcept;
        // Bitwise ABS
        inline simd4f abs() const;
        // Bitwise Infinity test
        inline simd4f isinf() const noexcept;
        // Bitwise Unordered( is NaN)
        inline simd4f isnan() const noexcept;
        // Negation -x
        inline simd4f operator-() const noexcept;
        // Math Ops: +, -, *, /
        inline friend simd4f operator+(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator-(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator*(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator/(simd4f a, simd4f b) noexcept;

        // FMAF: (a * b) + c
        static inline simd4f fma(simd4f a, simd4f b, simd4f c) noexcept;
        // Comparisons: <, >, <=, >=, ==, != (returns mask)
        inline friend simd4f operator<(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator>(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator<=(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator>=(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator==(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator!=(simd4f a, simd4f b) noexcept;

        // Bool Ops: AND, OR, XOR, NOT
        inline friend simd4f operator&(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator|(simd4f a, simd4f b) noexcept;
        inline friend simd4f operator^(simd4f a, simd4f b) noexcept;
        inline simd4f operator!() const noexcept;

        // ~a & b
        inline static simd4f andNot(simd4f a, simd4f b) noexcept;
        inline simd4f signbit() const noexcept;
        inline simd4f isneg() const noexcept;
        inline simd4f bittest_eq(__m128i i);
        inline static simd4f copysign(simd4f value, simd4f target_sign) noexcept;
        inline simd4f copysign(simd4f target_sign) noexcept;
        // Select (The ?: operator)
        // selector is a mask from a comparison
        inline static simd4f select(simd4f mask, simd4f if_true, simd4f if_false) noexcept;
        inline float extract_lane(int lane);
        inline float *extract_to(float *fs) noexcept;
        inline void setf(float f4, float f3, float f2, float f1) noexcept;
        inline void set_all_float(float f) noexcept;
        inline void set_int(int i) noexcept;
        inline void load(float *fs) noexcept;
    };
    inline simd4f atan2_fast(simd4f y, simd4f x) noexcept;
    inline simd4f atan2_accurate(simd4f y, simd4f x) noexcept;

#endif
    static constexpr inline float float_xor(float _f, uint mask) noexcept;
    static constexpr inline float float_or(float _f, uint mask) noexcept;
    static constexpr inline float float_and(float _f, uint mask) noexcept;
    static constexpr inline uint float_bits(float _f) noexcept;
    static constexpr inline bool float_signbit(float _f) noexcept;
    static constexpr inline float fast_fabs(float _f) noexcept;
    inline float atan2_fast(float _y, float _x) noexcept;
    inline float atan2_accurate(float y, float x) noexcept;
}
#endif

#if defined(IAA03_IMPLEMENT_MATH) && !defined(IAA03_IMPLEMENTED_MATH)

#define IAA03_IMPLEMENTED_MATH
#include <cmath>
namespace IAA03
{

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define IAA03_SIMD_LEVEL 3
#endif
#if defined(__AVX2__)

    // 1. Default: Initialize all 8 lanes to 0.0f
    simd8f::simd8f() : data(_mm256_setzero_ps()) {}

    // 2. Broadcast: Initialize all 8 lanes to the same value
    simd8f::simd8f(float f) : data(_mm256_set1_ps(f)) {}
    simd8f::simd8f(uint hex_float) : data(_mm256_castsi256_ps(_mm256_set1_epi32(hex_float))) {}

    // 3. Specific Lanes: Initialize each of the 8 lanes individually
    // Note: _mm256_set_ps takes arguments in REVERSE order (f8 is first lane in memory)
    simd8f::simd8f(float f1, float f2, float f3, float f4,
                   float f5, float f6, float f7, float f8)
        : data(_mm256_set_ps(f8, f7, f6, f5, f4, f3, f2, f1)) {}

    // 4. Pointer Load: Load 8 contiguous floats from memory
    simd8f::simd8f(const float *fs) : data(_mm256_loadu_ps(fs)) {}

    // 5. Internal: Initialize from an existing raw __m256 register
    simd8f::simd8f(__m256 d) : data(d) {}
    inline simd8f::operator __m256() const noexcept { return data; }
    // Bitwise ABS
    inline simd8f simd8f::abs() const noexcept
    {
        return _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x80'00'00'00u)), data);
    } // Bitwise Infinity test: Exponent all 1s, Mantissa all 0s
    inline simd8f simd8f::isinf() const noexcept
    {
        const __m256 signless_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        const __m256 inf_pattern = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000));
        return _mm256_cmp_ps(_mm256_and_ps(data, signless_mask), inf_pattern, _CMP_EQ_OQ);
    }

    // Bitwise Unordered: The standard "Self-Compare" NaN test
    inline simd8f simd8f::isnan() const noexcept
    {
        return _mm256_cmp_ps(data, data, _CMP_UNORD_Q);
    }
    // Negation -x
    inline simd8f simd8f::operator-() const noexcept
    {
        return _mm256_xor_ps(data, _mm256_castsi256_ps(_mm256_set1_epi32(0x80'00'00'00u)));
    }

    // Math Ops: +, -, *, /
    inline simd8f operator+(simd8f a, simd8f b) noexcept { return _mm256_add_ps(a.data, b.data); }
    inline simd8f operator-(simd8f a, simd8f b) noexcept { return _mm256_sub_ps(a.data, b.data); }
    inline simd8f operator*(simd8f a, simd8f b) noexcept { return _mm256_mul_ps(a.data, b.data); }
    inline simd8f operator/(simd8f a, simd8f b) noexcept { return _mm256_div_ps(a.data, b.data); }

    // FMAF: (a * b) + c
    inline simd8f simd8f::fma(simd8f a, simd8f b, simd8f c) noexcept
    {
        return _mm256_fmadd_ps(a.data, b.data, c.data);
    }
    inline simd8f operator<(simd8f a, simd8f b) noexcept { return _mm256_cmp_ps(a.data, b.data, _CMP_LT_OQ); }
    inline simd8f operator>(simd8f a, simd8f b) noexcept { return _mm256_cmp_ps(a.data, b.data, _CMP_GT_OQ); }
    inline simd8f operator<=(simd8f a, simd8f b) noexcept { return _mm256_cmp_ps(a.data, b.data, _CMP_LE_OQ); }
    inline simd8f operator>=(simd8f a, simd8f b) noexcept { return _mm256_cmp_ps(a.data, b.data, _CMP_GE_OQ); }
    inline simd8f operator==(simd8f a, simd8f b) noexcept { return _mm256_cmp_ps(a.data, b.data, _CMP_EQ_OQ); }
    inline simd8f operator!=(simd8f a, simd8f b) noexcept { return _mm256_cmp_ps(a.data, b.data, _CMP_NEQ_OQ); }

    // Bool Ops: AND, OR, XOR, NOT
    inline simd8f operator&(simd8f a, simd8f b) noexcept { return _mm256_and_ps(a.data, b.data); }
    inline simd8f operator|(simd8f a, simd8f b) noexcept { return _mm256_or_ps(a.data, b.data); }
    inline simd8f operator^(simd8f a, simd8f b) noexcept { return _mm256_xor_ps(a.data, b.data); }
    inline simd8f simd8f::operator!() const noexcept
    {
        // Logical NOT for masks: Flips all bits
        return _mm256_xor_ps(data, _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF)));
    }
    inline simd8f simd8f::andNot(simd8f a, simd8f b) noexcept { return _mm256_andnot_ps(a.data, b.data); }

    // Returns the raw sign bit (0x80000000)
    inline simd8f simd8f::signbit() const noexcept
    {
        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000u));
        return _mm256_and_ps(mask, data);
    }

    // Returns a TRUE MASK (-nan) if negative
    inline simd8f simd8f::isneg() const noexcept
    {
        return _mm256_cmp_ps(data, _mm256_setzero_ps(), _CMP_LT_OQ);
    } // Compares the bit-pattern of 'data' against the 256-bit integer pattern 'i'
    inline simd8f simd8f::bittest_eq(__m256i i) const noexcept
    {
        // 1. Cast 256-bit float register to 256-bit integer register (zero cost)
        __m256i data_as_int = _mm256_castps_si256(data);

        // 2. Perform 8-wide 32-bit integer comparison
        __m256i mask = _mm256_cmpeq_epi32(data_as_int, i);

        // 3. Cast back to float for mask usage (zero cost)
        return _mm256_castsi256_ps(mask);
    }
    inline simd8f simd8f::copysign(simd8f value, simd8f target_sign) noexcept
    {
        return _mm256_or_ps(target_sign.signbit(), value.abs());
    }
    inline simd8f simd8f::copysign(simd8f target_sign) noexcept
    {
        return _mm256_or_ps(target_sign.signbit(), abs());
    }

    // Select (The ?: operator)
    // selector is a mask from a comparison
    inline simd8f simd8f::select(simd8f mask, simd8f if_true, simd8f if_false) noexcept
    {
        return _mm256_blendv_ps(if_false.data, if_true.data, mask.data);
    }
    inline float simd8f::extract_lane(int lane)
    {
        alignas(32) float res[8]; // Must be 8 for AVX
        _mm256_storeu_ps(res, data);
        return res[lane];
    }

    inline void simd8f::extract_to(float *fs) noexcept
    {
        _mm256_storeu_ps(fs, data);
    }

    // Loads 8 floats
    inline void simd8f::load(const float *fs) noexcept
    {
        data = _mm256_loadu_ps(fs);
    }

    inline void simd8f::set_all_float(float f) noexcept
    {
        data = _mm256_set1_ps(f);
    }

    inline simd8f atan2_fast(simd8f y, simd8f x) noexcept
    {
        const simd8f PI(3.14159265f);
        const simd8f PI_HALF(1.57079632f);
        const simd8f PI_Quadrant(0.7853981633974484f);
        const simd8f ZERO;
        const simd8f ONE(1.0f);
        const simd8f c1 = 0.354;
        /*
        bool nx = float_signbit(x), ny = float_signbit(y);
        x = float_and(x, 0x7f'ff'ff'ff);
        y = float_and(y, 0x7f'ff'ff'ff);
        */
        simd8f nx = (x).signbit(),
               ny = (y).signbit();
        x = (x.abs());
        y = (y.abs());
        simd8f x_y = x > y;
        simd8f zy = y == ZERO;
        simd8f base = simd8f::select(x_y, x, y);
        simd8f value = simd8f::select(x_y, y, x);

        simd8f f = value / base;
        f = simd8f::select(value == base, ONE, simd8f::select(base == ZERO, ZERO, f));
        // return (x+x * ix* c1 - u * ui3 * (!cond)*c2 - t * ti2 * (cond)* c3 - x * ix2 * sp2*c4)
        //     *piq;
        simd8f ratio = simd8f::fma(f * (ONE - f), c1, f) * PI_Quadrant;

        simd8f angle = (simd8f::select(x_y ^ nx, (ratio), PIhalf - ratio));
        simd8f radians = simd8f::select(zy, simd8f::select((nx), PI, ZERO), (simd8f::select(nx, PIhalf, ZERO) + angle));
        return radians.copysign(ny);
    }
    inline simd8f atan2_accurate(simd8f y, simd8f x) noexcept
    {

        const simd8f piq = PIquadrant;
        const simd8f pih = PIhalf;
        const simd8f pi = PI;
        const simd8f c1 = 0.36989;
        const simd8f c2 = -7.5 * 0.009;
        const simd8f c3 = -0.175 * 0.0095;
        const simd8f c4 = -0.65 * 0.00964;
        const simd8f QNAN = 0.0 / 0.0;
        const simd8f zero;
        const simd8f one(1.0f);
        const simd8f sign_mask(0x80'00'00'00u);
        const simd8f abs_mask(0x7f'ff'ff'ffu);
        const simd8f pc(1.25f);
        const simd8f bc(3.9533571094f);
        const simd8f ac(1.3385977256f);
        const simd8f condc(0.747050425165f);
        simd8f nx = x.signbit();
        simd8f ny = y.signbit();
        simd8f zy = y == zero;
        x = (x & abs_mask);
        y = (y & abs_mask);
        simd8f x_y = x > y;
        simd8f same = x == y;
        simd8f base = simd8f::select(x_y, x, y);
        simd8f value = simd8f::select(x_y, y, x);

        simd8f f = simd8f::select(base == zero, zero, simd8f::select(same, one, value / base));
        simd8f cond = f > condc;
        simd8f fi = one - f;
        simd8f ffi = f * fi, core = simd8f::fma(ffi, c1, f);
        simd8f a = f * ac, ia = one - a, ia3 = ia * ia * ia;
        simd8f b = fi * bc, ib = one - b, ib2 = ib * ib;
        b = simd8f::fma(b * ib2, c3, core);
        a = simd8f::fma(a * ia3, c2, core);
        simd8f p = f * pc, sp2 = simd8f::fma(-p, p, one);
        simd8f c = simd8f::select(cond, b, a);
        a = ffi * fi;
        b = sp2 * c4;
        simd8f ratio = simd8f::fma(a, b, c) * piq;

        // return (x+x * ix* c1 - u * ui3 * (!cond)*c2 - t * ti2 * (cond)* c3 - x * ix2 * sp2*c4)
        //     *piq;

        simd8f angle = simd8f::select((x_y ^ nx), ratio, pih - ratio);
        simd8f radians = simd8f::select(zy, simd8f::select(nx, pi, zero), (simd8f::select(nx, pih, zero) + angle));
        radians = simd8f::select((radians != radians), QNAN, radians);
        return radians.copysign(ny);
    }
#endif
#if defined(__SSE4_1__) || defined(_M_AMD64) || defined(_M_X64)

    // Constant Load
    simd4f::simd4f() : data(_mm_setzero_ps()) {}
    simd4f::simd4f(float f) : data(_mm_set1_ps(f)) {}
    simd4f::simd4f(float f1, float f2, float f3, float f4) : data(_mm_set_ps(f4, f3, f2, f1)) {}
    simd4f::simd4f(const float *fs) : data(_mm_loadu_ps(fs)) {}
    simd4f::simd4f(__m128 m) : data(m) {}
    inline simd4f::operator __m128() const noexcept { return data; }
    // Bitwise ABS
    inline simd4f simd4f::abs() const
    {
        return _mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32(0x80'00'00'00u)), data);
    }
    // Bitwise Infinity test
    inline simd4f simd4f::isinf() const noexcept
    {
        simd4f signless = _mm_castsi128_ps(_mm_set1_epi32(0x7fff'ffff));
        simd4f inf = _mm_castsi128_ps(_mm_set1_epi32(0x7f80'00'00));
        return _mm_cmpeq_ps(_mm_and_ps(signless, data), inf);
    }
    // Bitwise Unordered( is NaN)
    inline simd4f simd4f::isnan() const noexcept
    {
        return _mm_cmpunord_ps(data, data);
    }
    // Negation -x
    inline simd4f simd4f::operator-() const noexcept
    {
        return _mm_xor_ps(data, _mm_castsi128_ps(_mm_set1_epi32(0x80'00'00'00u)));
    }

    // Math Ops: +, -, *, /
    inline simd4f operator+(simd4f a, simd4f b) noexcept { return _mm_add_ps(a.data, b.data); }
    inline simd4f operator-(simd4f a, simd4f b) noexcept { return _mm_sub_ps(a.data, b.data); }
    inline simd4f operator*(simd4f a, simd4f b) noexcept { return _mm_mul_ps(a.data, b.data); }
    inline simd4f operator/(simd4f a, simd4f b) noexcept { return _mm_div_ps(a.data, b.data); }

    // FMAF: (a * b) + c
    inline simd4f simd4f::fma(simd4f a, simd4f b, simd4f c) noexcept
    {
        return _mm_fmadd_ps(a.data, b.data, c.data);
    }

    // Comparisons: <, >, <=, >=, ==, != (returns mask)
    inline simd4f operator<(simd4f a, simd4f b) noexcept { return _mm_cmplt_ps(a.data, b.data); }
    inline simd4f operator>(simd4f a, simd4f b) noexcept { return _mm_cmpgt_ps(a.data, b.data); }
    inline simd4f operator<=(simd4f a, simd4f b) noexcept { return _mm_cmple_ps(a.data, b.data); }
    inline simd4f operator>=(simd4f a, simd4f b) noexcept { return _mm_cmpge_ps(a.data, b.data); }
    inline simd4f operator==(simd4f a, simd4f b) noexcept { return _mm_cmpeq_ps(a.data, b.data); }
    inline simd4f operator!=(simd4f a, simd4f b) noexcept { return _mm_cmpneq_ps(a.data, b.data); }

    // Bool Ops: AND, OR, XOR, NOT
    inline simd4f operator&(simd4f a, simd4f b) noexcept { return _mm_and_ps(a.data, b.data); }
    inline simd4f operator|(simd4f a, simd4f b) noexcept { return _mm_or_ps(a.data, b.data); }
    inline simd4f operator^(simd4f a, simd4f b) noexcept { return _mm_xor_ps(a.data, b.data); }
    // Logical NOT for masks: Flips all bits
    inline simd4f simd4f::operator!() const noexcept { return _mm_xor_ps(data, _mm_castsi128_ps(_mm_set1_epi32(0xFFFFFFFF))); }

    // ~a & b
    inline simd4f simd4f::andNot(simd4f a, simd4f b) noexcept { return _mm_andnot_ps(a.data, b.data); }
    inline simd4f simd4f::signbit() const noexcept
    {
        const static __m128 mask(_mm_castsi128_ps(_mm_set1_epi32(0x80'00'00'00u)));
        return _mm_and_ps(mask, data);
    }
    inline simd4f simd4f::isneg() const noexcept
    {
        __m128i i = _mm_set1_epi32(0x80'00'00'00);
        return _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(i, _mm_castps_si128(data)), i));
    }
    inline simd4f simd4f::bittest_eq(__m128i i)
    {
        return _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(data), i));
    }
    inline simd4f simd4f::copysign(simd4f value, simd4f target_sign) noexcept
    {
        return _mm_or_ps(target_sign.signbit(), value.abs());
    }
    inline simd4f simd4f::copysign(simd4f target_sign) noexcept
    {
        return _mm_or_ps(target_sign.signbit(), abs());
    }

    // Select (The ?: operator)
    // selector is a mask from a comparison
    inline simd4f simd4f::select(simd4f mask, simd4f if_true, simd4f if_false) noexcept
    {
        return _mm_blendv_ps(if_false.data, if_true.data, mask.data);
    }
    inline float simd4f::extract_lane(int lane)
    {
        float res[4];
        _mm_storeu_ps(res, data);
        return res[lane];
    }
    inline float *simd4f::extract_to(float *fs) noexcept
    {
        _mm_storeu_ps(fs, data);
        return fs;
    }
    inline void simd4f::setf(float f4, float f3, float f2, float f1) noexcept
    {
        data = _mm_set_ps(f4, f3, f2, f1);
    }
    inline void simd4f::set_all_float(float f) noexcept
    {
        data = _mm_set1_ps(f);
    }
    inline void simd4f::set_int(int i) noexcept
    {
        data = _mm_castsi128_ps(_mm_set1_epi32(i));
    }
    inline void simd4f::load(float *fs) noexcept
    {
        data = _mm_loadu_ps(const_cast<const float *>(fs));
    }
    inline simd4f atan2_fast(simd4f y, simd4f x) noexcept
    {

        const simd4f piq = PIquadrant;
        const simd4f pih = PIhalf;
        const simd4f pi = PI;
        const simd4f QNAN = 0.0 / 0.0;
        const simd4f zero;
        const simd4f one(1);
        const simd4f sign_mask(_mm_castsi128_ps(_mm_set1_epi32(0x80'00'00'00u)));
        const simd4f abs_mask(_mm_castsi128_ps(_mm_set1_epi32(0x7f'ff'ff'ffu)));
        const simd4f c1 = 0.354;
        simd4f nx = x.signbit(), ny = y.signbit();
        x = (x.abs());
        y = (y.abs());
        simd4f x_y = x > y;
        simd4f zy = y == zero;
        simd4f base = simd4f::select(x_y, x, y);
        simd4f value = simd4f::select(x_y, y, x);

        simd4f f = simd4f::select(base == zero, zero, simd4f::select(x == y, one, value / base));
        // return (x+x * ix* c1 - u * ui3 * (!cond)*c2 - t * ti2 * (cond)* c3 - x * ix2 * sp2*c4)
        //     *piq;
        simd4f ratio = simd4f::fma(f * (1 - f), c1, f) * piq;
        simd4f angle = simd4f::select((x_y ^ nx), ratio, pih - ratio);
        simd4f radians = simd4f::select(zy, simd4f::select(nx, pi, zero), (simd4f::select(nx, pih, zero) + angle));
        radians = simd4f::select((radians != radians), QNAN, radians);
        return radians.copysign(ny);
    }

    inline simd4f atan2_accurate(simd4f y, simd4f x) noexcept
    {

        const simd4f piq = PIquadrant;
        const simd4f pih = PIhalf;
        const simd4f pi = PI;
        const simd4f c1 = 0.36989;
        const simd4f c2 = -7.5 * 0.009;
        const simd4f c3 = -0.19 * 0.0095;
        const simd4f c4 = -0.65 * 0.00964;
        const simd4f QNAN = 0.0 / 0.0;
        const simd4f zero;
        const simd4f one(1);
        const simd4f sign_mask(_mm_castsi128_ps(_mm_set1_epi32(0x80'00'00'00u)));
        const simd4f abs_mask(_mm_castsi128_ps(_mm_set1_epi32(0x7f'ff'ff'ffu)));
        const simd4f pc(1.25f);
        const simd4f bc(3.9533571094f);
        const simd4f ac(1.3385977256f);
        const simd4f condc(0.747050425165f);
        simd4f nx = (x.signbit()),
               ny = (y.signbit());
        simd4f zy = y == zero;
        x = (x & abs_mask);
        y = (y & abs_mask);
        simd4f x_y = x > y;
        simd4f same = x == y;
        simd4f base = simd4f::select(x_y, x, y);
        simd4f value = simd4f::select(x_y, y, x);

        simd4f f = simd4f::select(base == zero, zero, simd4f::select(same, one, value / base));
        simd4f cond = f > condc;
        simd4f fi = one - f;
        simd4f ffi = f * fi, core = simd4f::fma(ffi, c1, f);
        simd4f a = f * ac, ia = one - a, ia3 = ia * ia * ia;
        simd4f b = fi * bc, ib = one - b, ib2 = ib * ib;
        b = simd4f::fma(b * ib2, c3, core);
        a = simd4f::fma(a * ia3, c2, core);
        simd4f p = f * pc, sp2 = simd4f::fma(-p, p, one);
        simd4f c = simd4f::select(cond, b, a);
        a = ffi * fi;
        b = sp2 * c4;
        simd4f ratio = simd4f::fma(a, b, c) * piq;

        // return (x+x * ix* c1 - u * ui3 * (!cond)*c2 - t * ti2 * (cond)* c3 - x * ix2 * sp2*c4)
        //     *piq;

        simd4f angle = simd4f::select((x_y ^ nx), ratio, pih - ratio);
        simd4f radians = simd4f::select(zy, simd4f::select(nx, pi, zero), (simd4f::select(nx, pih, zero) + angle));
        radians = simd4f::select((radians != radians), QNAN, radians);
        return radians.copysign(ny);
    }
#endif
    static constexpr inline float float_xor(float _f, uint mask) noexcept
    {
        float f = _f;
        uint u = 0;
        memcpy(&u, &f, 4);
        u ^= mask;
        memcpy(&f, &u, 4);
        return f;
    }
    static constexpr inline float float_or(float _f, uint mask) noexcept
    {
        float f = _f;
        uint u = 0;
        memcpy(&u, &f, 4);
        u |= mask;
        memcpy(&f, &u, 4);
        return f;
    }
    static constexpr inline float float_and(float _f, uint mask) noexcept
    {
        float f = _f;
        uint u = 0;
        memcpy(&u, &f, 4);
        u &= mask;
        memcpy(&f, &u, 4);
        return f;
    }
    static constexpr inline uint float_bits(float _f) noexcept
    {
        uint u = 0;
        memcpy(&u, &_f, 4);
        return u;
    }

    static constexpr inline bool float_signbit(float _f) noexcept
    {
        uint u = 0;
        memcpy(&u, &_f, 4);
        return (u & 0x80'00'00'00u) != 0;
    }
    static constexpr inline float fast_fabs(float _f) noexcept
    {
        float f = _f;
        uint u = 0;
        memcpy(&u, &f, 4);
        u &= 0x7f'ff'ff'ffu;
        memcpy(&f, &u, 4);
        return f;
    }
    inline float atan2_fast(float y, float x) noexcept
    {

        constexpr float piq = PI / 4;
        constexpr float c1 = 0.354;
        constexpr float QNAN = __builtin_nan("");
        bool nx = float_signbit(x), ny = float_signbit(y);
        x = float_and(x, 0x7f'ff'ff'ff);
        y = float_and(y, 0x7f'ff'ff'ff);
        bool x_y = x > y;
        bool same = x == y;
        float base = x_y ? x : y;
        float value = x_y ? y : x;

        float f = (base == 0) ? 0 : (same ? 1 : value / base);
        float fi = 1 - f;
        float ratio = (fmaf(f * fi, c1, f) * piq);
        bool zy = y == 0;
        //        float l = nx ? PIhalf + (zy ? PIhalf : 0) : 0;
        float p = nx ? (zy ? PI : PIhalf) : 0;
        float angle = (((x_y ^ nx) ? ratio : PIhalf - ratio));
        float radians = (zy ? 0 : angle) + p;
        return float_xor(radians, (ny) ? 0x8000'0000u : 0);
    }
    inline float atan2_accurate(float y, float x) noexcept
    {

        constexpr float piq = PI / 4;
        constexpr float c1 = 0.36989;
        constexpr float c2 = -7.5 * 0.009;
        constexpr float c3 = -0.175 * 0.0095;
        constexpr float c4 = -0.65 * 0.00964;
        constexpr float QNAN = __builtin_nan("");
        bool nx = float_signbit(x), ny = float_signbit(y);
        bool zy = y == 0;
        x = float_and(x, 0x7f'ff'ff'ff);
        y = float_and(y, 0x7f'ff'ff'ff);
        bool x_y = x > y;
        bool same = x == y;
        float base = x_y ? x : y;
        float value = x_y ? y : x;

        float f = (base == 0) ? 0 : (same ? 1 : value / base);
        bool cond = f > 0.747050425165f;
        float fi = 1 - f;
        float ffi = f * fi, core = fmaf(ffi, c1, f);
        float a = f * 1.3385977256f, ia = 1 - a, ia3 = ia * ia * ia;
        float b = fi * 3.9533571094f, ib = 1 - b, ib2 = ib * ib;
        b = fmaf(b * ib2, c3, core);
        a = fmaf(a * ia3, c2, core);
        float p = f * 1.25f, sp2 = fmaf(float_xor(p, 0x8000'0000), p, 1);
        float c = cond ? b : a;
        a = ffi * fi;
        b = sp2 * c4;
        float ratio = fmaf(a, b, c) * piq;

        // return (x+x * ix* c1 - u * ui3 * (!cond)*c2 - t * ti2 * (cond)* c3 - x * ix2 * sp2*c4)
        //     *piq;

        float angle = (((x_y ^ nx) ? ratio : PIhalf - ratio));
        float radians = (!zy ? ((nx ? PIhalf : 0) + angle) : ((nx) ? PI : 0));
        // nx PI | PIhalf ,PI if zy + angle
        radians = (radians != radians) ? QNAN : radians;
        return float_xor(radians, (ny) ? 0x80'00'00'00u : 0u);
    }
}
#endif