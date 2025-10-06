/**  RealFFT v1.2 C++ wrapper class written by Dmitry Boldyrev
 **
 **  File: const1.h
 **  Miscellaneous useful functions and definitions for SIMD support
 **
 **  GITHUB: https://github.com/mewza
 **  Email: subband@protonmail.com
 **
 **/

#pragma once

#include <signal.h>
#include <malloc/malloc.h>
#include <simd/simd.h>
#include <type_traits>
#include <algorithm>
#include <map>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <functional>

#define MSSFLOAT_IS_FLOAT (sizeof(mssFloat) == sizeof(float))

#define NOT_VECTOR(Z) (std::is_same_v<Z, float> || std::is_same_v<Z, double>)
#define IS_VECTOR(Z)  (!NOT_VECTOR(Z))

#ifndef D_ZFLOAT
#define D_ZFLOAT

#define DECL_ZFLOAT(TYPE) \
typedef TYPE zfloat; \
typedef simd_##TYPE##8 zfloat8; \
typedef simd_##TYPE##4 zfloat4; \
typedef simd_##TYPE##2 zfloat2; \
static inline zfloat8 make_zfloat8(zfloat4& a, zfloat4& b) { return simd_make_##TYPE##8(a,b); } \
static inline zfloat4 make_zfloat4(zfloat2& a, zfloat2& b) { return simd_make_##TYPE##4(a,b); } \
static inline zfloat2 make_zfloat2(zfloat& a, zfloat& b) { return simd_make_##TYPE##2(a,b); }

DECL_ZFLOAT(float)

#endif


#define ZFLOAT_IS_FLOAT constexpr(sizeof(zfloat) == sizeof(float))
#define FUNDEMENTAL(T) std::is_same_v<T, float> || std::is_same_v<T, double>

typedef float mssFloat;

typedef simd_float8 mssFloat8;
typedef simd_float4 mssFloat4;
typedef simd_float2 mssFloat2;

typedef simd_float2 float2v;
typedef simd_float4 float4v;
typedef simd_float8 float8v;

typedef simd_double2 double2v;
typedef simd_double4 double4v;
typedef simd_double8 double8v;

typedef simd_long8 long8v;
typedef simd_long4 long4v;
typedef simd_long2 long2v;

typedef simd_int2 int2v;
typedef simd_int4 int4v;
typedef simd_int8 int8v;

typedef std::conditional<sizeof(mssFloat) == sizeof(float), float2v, double2v>::type mssFloat2;
typedef std::conditional<sizeof(mssFloat) == sizeof(float), float4v, double4v>::type mssFloat4;
typedef std::conditional<sizeof(mssFloat) == sizeof(float), float8v, double8v>::type mssFloat8;

static __inline float8v make_mssFloat8(const float4v &a, const float4v &b) {
    return simd_make_float8(a,b);
}

static __inline double8v make_mssFloat8(const double4v &a, const double4v &b) {
    return simd_make_double8(a,b);
}

static __inline float4v make_mssFloat4(const float2v &a, const float2v &b) {
    return simd_make_float4(a,b);
}

static __inline double4v make_mssFloat4(const double2v &a, const double2v &b) {
    return simd_make_double4(a,b);
}

static __inline float2v make_mssFloat2(const float &a, const float &b) {
    return simd_make_float2(a,b);
}

static __inline double2v make_mssFloat2(const double &a, const double &b) {
    return simd_make_double2(a,b);
}

//typedef mssFloat cmplx[2];

#ifndef CMPLX_T_TYPE
#define CMPLX_T_TYPE

template <typename T> struct cmplxT;

#endif // CMPLX_T_TYPE

class const1 {
public:
    float v;
    constexpr const1 (float value) : v(value) {}
    inline operator mssFloat2() const {
        return mssFloat2{v, v};
    }
    inline operator mssFloat4() const {
        return mssFloat4{v, v, v, v};
    }
    inline operator mssFloat8() const {
        return mssFloat8{v, v, v, v, v, v, v, v};
    }
    inline operator float() const {
        return v;
    }
    inline constexpr const1 operator-() const { return const1(-v); }
};

inline constexpr const1 operator""_v(long double d) {
    return const1(double(d));
}

#define shufflevector __builtin_shufflevector
#define convertvector __builtin_convertvector

static __inline double F_ABS(double v) { return __builtin_fabs(v); }
static __inline float F_ABS(float v) { return __builtin_fabs(v); }
static __inline double8v F_ABS(double8v v) { return simd::fabs(v); }
static __inline double4v F_ABS(double4v v) { return simd::fabs(v); }
static __inline double2v F_ABS(double2v v) { return simd::fabs(v); }
static __inline float8v F_ABS(float8v v) { return simd::fabs(v); }
static __inline float4v F_ABS(float4v v) { return simd::fabs(v); }
static __inline float2v F_ABS(float2v v) { return simd::fabs(v); }

static __inline mssFloat2 F_COPYSIGN(mssFloat2 mag, mssFloat2 sign) { return simd::copysign(mag, sign); }
//static __inline mssFloat4 F_COPYSIGN(mssFloat4 mag, mssFloat4 sign) { return simd::copysign(mag, sign); }
//static __inline mssFloat8 F_COPYSIGN(mssFloat8 mag, mssFloat8 sign) { return simd::copysign(mag, sign); }
static __inline mssFloat8 F_COPYSIGN(float mag, mssFloat8 sign) { return simd::copysign((mssFloat8)const1(mag), sign); }
static __inline mssFloat4 F_COPYSIGN(float mag, mssFloat4 sign) { return simd::copysign((mssFloat4)const1(mag), sign); }
static __inline float4v F_COPYSIGN(float4v mag, float4v sign) { return simd::copysign(mag, sign); }
static __inline float8v F_COPYSIGN(float8v mag, float8v sign) { return simd::copysign(mag, sign); }
static __inline double4v F_COPYSIGN(double4v mag, double4v sign) { return simd::copysign(mag, sign); }
static __inline double8v F_COPYSIGN(double8v mag, double8v sign) { return simd::copysign(mag, sign); }

static __inline double8v F_MIN(double8v a, double8v b) {return simd::fmin(a,b); }
static __inline double4v F_MIN(double4v a, double4v b) { return simd::fmin(a,b); }
static __inline double2v F_MIN(double2v a, double2v b) { return simd::fmin(a,b); }
static __inline float8v F_MIN(float8v a, float8v b) { return simd::fmin(a,b); }
static __inline float4v F_MIN(float4v a, float4v b) { return simd::fmin(a,b); }
static __inline float2v F_MIN(float2v a, float2v b) { return simd::fmin(a,b); }
static __inline double F_MIN(float a, double b) { return __builtin_fmin(a, b); }
static __inline float F_MIN(float a, float b) { return __builtin_fmin(a, b); }
static __inline int F_MIN(int a, int b) { return __builtin_fmin(a, b); }

static __inline int8v F_MAX(int8v a, int8v b) {return simd::max(a,b); }
static __inline int4v F_MAX(int4v a, int4v b) {return simd::max(a,b); }
static __inline double8v F_MAX(double8v a, double8v b) {return simd::max(a,b); }
static __inline double4v F_MAX(double4v a, double4v b) { return simd::max(a,b); }
static __inline double2v F_MAX(double2v a, double2v b) { return simd::max(a,b); }
static __inline float8v F_MAX(float8v a, float8v b) {return simd::max(a,b); }
static __inline float4v F_MAX(float4v a, float4v b) { return simd::max(a,b); }
static __inline float2v F_MAX(float2v a, float2v b) { return simd::max(a,b); }
static __inline double F_MAX(double a, double b) { return __builtin_fmax(a, b); }
static __inline double F_MAX(double a, float b) { return __builtin_fmax(a, b); }
static __inline double F_MAX(float a, double b) { return __builtin_fmax(a, b); }
static __inline float F_MAX(float a, float b) { return __builtin_fmax(a, b); }
static __inline int F_MAX(int a, int b) { return std::max(a, b); }

static __inline mssFloat8 F_CLAMP(mssFloat8 x, mssFloat8 a, mssFloat8 b) { return F_MAX(a, F_MIN(x, b)); }
static __inline mssFloat4 F_CLAMP(mssFloat4 x, mssFloat4 a, mssFloat4 b) { return F_MAX(a, F_MIN(x, b)); }
static __inline mssFloat2 F_CLAMP(mssFloat2 x, mssFloat2 a, mssFloat2 b) { return F_MAX(a, F_MIN(x, b)); }
static __inline double F_CLAMP(double x, double a, double b) { return F_MIN(b, F_MAX(x, a)); }
static __inline double F_CLAMP(double x, float a, float b) { return F_MIN(b, F_MAX(x, a)); }
static __inline double F_CLAMP(double x, float a, double b) { return F_MIN(b, F_MAX(x, a)); }
static __inline double F_CLAMP(double x, double a, float b) { return F_MIN(b, F_MAX(x, a)); }
static __inline float F_CLAMP(float x, float a, float b) { return F_MIN(b, F_MAX(x, a)); }
static __inline float F_CLAMP(float x, double a, float b) { return F_MIN(b, F_MAX(x, a)); }
static __inline float F_CLAMP(float x, float a, double b) { return F_MIN(b, F_MAX(x, a)); }

static __inline double8v F_SIN(double8v s) { return simd::sin(s); }
static __inline double4v F_SIN(double4v s) { return simd::sin(s); }
static __inline double2v F_SIN(double2v s) { return simd::sin(s); }
static __inline float8v F_SIN(float8v s) { return simd::sin(s); }
static __inline float4v F_SIN(float4v s) { return simd::sin(s); }
static __inline float2v F_SIN(float2v s) { return simd::sin(s); }
static __inline mssFloat F_SIN(mssFloat s) { return __builtin_sin(s); }

static __inline mssFloat F_POW(mssFloat a, mssFloat b) { return __builtin_pow(a,b); }

static __inline float8v F_SQRT(float8v s) { return simd::sqrt(s); }
static __inline double8v F_SQRT(double8v s) { return simd::sqrt(s); }
static __inline float4v F_SQRT(float4v s) { return simd::sqrt(s); }
static __inline double4v F_SQRT(double4v s) { return simd::sqrt(s); }
static __inline float2v F_SQRT(float2v s) { return simd::sqrt(s); }
static __inline double2v F_SQRT(double2v s) { return simd::sqrt(s); }
static __inline double F_SQRT(double s) { return __builtin_sqrt(s); }
static __inline float F_SQRT(float s) { return __builtin_sqrt(s); }

static __inline mssFloat2 F_ATAN(mssFloat2 s) { return simd::atan(s); }
static __inline mssFloat4 F_ATAN(mssFloat4 s) { return simd::atan(s); }
static __inline mssFloat8 F_ATAN(mssFloat8 s) { return simd::atan(s); }

static __inline double F_ACOS(double s) { return acos(s); }
static __inline mssFloat2 F_ACOS(mssFloat2 s) { return simd::acos(s); }
static __inline mssFloat4 F_ACOS(mssFloat4 s) { return simd::acos(s); }
static __inline mssFloat8 F_ACOS(mssFloat8 s) { return simd::acos(s); }

static __inline size_t F_MAX(size_t a, size_t b) { return std::max(a, b); }
static __inline size_t F_MIN(size_t a, size_t b) { return std::min(a, b); }
static __inline int F_CLAMP(int x, int a, int b) { return std::clamp(x, a, b); }

static __inline double __builtin_reduce_avg(const mssFloat2 &a) {
    double x = simd_reduce_add(a) * 0.5;
    return (x != x) ? 0.0 : x;
}

static __inline double __builtin_reduce_avg(const mssFloat8 &a) {
    double x = simd_reduce_add(a) * 0.125;
    return (x != x) ? 0.0 : x;
}

static __inline double __builtin_reduce_avg(const mssFloat4 &a) {
    double x = simd_reduce_add(a) * 0.25;
    return (x != x) ? 0.0 : x;
}

static __inline mssFloat8 F_POW(mssFloat8 a, mssFloat8 b) { return simd::pow(a, b); }
static __inline mssFloat4 F_POW(mssFloat4 a, mssFloat4 b) { return simd::pow(a, b); }
static __inline mssFloat2 F_POW(mssFloat2 a, mssFloat2 b) { return simd::pow(a, b); }

static __inline double F_LOG(double g) { return __builtin_log(g); }
static __inline float F_LOG(float g) { return __builtin_log(g); }
static __inline mssFloat2 F_LOG(mssFloat2 g) { return simd::log(g); }
static __inline mssFloat4 F_LOG(mssFloat4 g) { return simd::log(g); }
static __inline mssFloat8 F_LOG(mssFloat8 g) { return simd::log(g); }

static __inline mssFloat2 F_EXP(mssFloat2 g) { return simd::exp(g); }
static __inline mssFloat4 F_EXP(mssFloat4 g) { return simd::exp(g); }
static __inline mssFloat8 F_EXP(mssFloat8 g) { return simd::exp(g); }
static __inline double F_EXP(double g) { return fast_expf(g); }
static __inline float F_EXP(float g) { return fast_expf(g); }

static __inline mssFloat8 F_ATAN2(mssFloat8 a, mssFloat8 b) { return simd::atan2(a, b); }
static __inline mssFloat4 F_ATAN2(mssFloat4 a, mssFloat4 b) { return simd::atan2(a, b); }
static __inline mssFloat2 F_ATAN2(mssFloat2 a, mssFloat2 b) { return simd::atan2(a, b); }
static __inline zfloat F_ATAN2(zfloat a, zfloat b) { return __builtin_atan2(a, b); }

static __inline double8v F_COS(double8v g) { return simd::cos(g); }
static __inline double4v F_COS(double4v g) { return simd::cos(g); }
static __inline double2v F_COS(double2v g) { return simd::cos(g); }
static __inline float8v F_COS(float8v g) { return simd::cos(g); }
static __inline float4v F_COS(float4v g) { return simd::cos(g); }
static __inline float2v F_COS(float2v g) { return simd::cos(g); }
static __inline double F_COS(double g) { return __builtin_cos(g); }
static __inline float F_COS(float g) { return __builtin_cos(g); }

static __inline double F_LOG10(double g) { return __builtin_log10(g); }
static __inline mssFloat2 F_LOG10(mssFloat2 g) { return simd::log10(g); }
static __inline mssFloat4 F_LOG10(mssFloat4 g) { return simd::log10(g); }
static __inline mssFloat8 F_LOG10(mssFloat8 g) { return simd::log10(g); }

static __inline mssFloat2 from_dB(mssFloat2 x) { return F_EXP( x * 0.1151292546497023 ); }
static __inline mssFloat8 from_dB(mssFloat8 x) { return F_EXP( x * 0.1151292546497023 ); }
static __inline double from_dB(double x) { return F_EXP( x * 0.1151292546497023 ); }

static __inline mssFloat2 to_dB(mssFloat2 g) { return (20.0f * F_LOG10(g)); }
static __inline mssFloat4 to_dB(mssFloat4 g) { return (20.0f * F_LOG10(g)); }
static __inline mssFloat8 to_dB(mssFloat8 g) {return (20.0f * F_LOG10(g)); }
static __inline float to_dB(float g) { return (20.0f * F_LOG10(g)); }

#define USE_NO_FINITE_MATH // -fno-finite-math-only

#if defined(USE_NO_FINITE_MATH)

static __inline bool ISNAN(float x)  { return __builtin_isnan(x); } //return x != x; }
static __inline bool ISNAN(double x) { return __builtin_isnan(x); } //return x != x; }

static __inline int8v ISNAN(float8v x) { return simd::isnan(x); }
static __inline int4v ISNAN(float4v x) { return simd::isnan(x); }
static __inline int2v ISNAN(float2v x) { return simd::isnan(x); }

static __inline long8v ISNAN(double8v x) { return simd::isnan(x); }
static __inline long4v ISNAN(double4v x) { return simd::isnan(x); }
static __inline long2v ISNAN(double2v x) { return simd::isnan(x); }

static __inline bool ISINF(double x) { return __builtin_isinf(x); } //return !ISNAN(x) && ISNAN(x - x); }
static __inline bool ISINF(float x)  { return __builtin_isinf(x); } //return !ISNAN(x) && ISNAN(x - x); }

static __inline int8v ISINF(float8v x) { return simd::isnan(x); }
static __inline int4v ISINF(float4v x) { return simd::isnan(x); }
static __inline int2v ISINF(float2v x) { return simd::isnan(x); }

static __inline long8v ISINF(double8v x) { return simd::isnan(x); }
static __inline long4v ISINF(double4v x) { return simd::isnan(x); }
static __inline long2v ISINF(double2v x) { return simd::isnan(x); }

__inline double F_RINT(double d) {
    d += 6755399441055744.0;
    return (double)reinterpret_cast<int&>(d);
}

static __inline float8v F_RINT(float8v x) { return simd::rint(x); }
static __inline float4v F_RINT(float4v x) { return simd::rint(x); }
static __inline float2v F_RINT(float2v x) { return simd::rint(x); }

static __inline double8v F_RINT(double8v x) { return simd::rint(x); }
static __inline double4v F_RINT(double4v x) { return simd::rint(x); }
static __inline double2v F_RINT(double2v x) { return simd::rint(x); }

static __inline bool ISNORM(double v) { return !ISINF(v) && !ISNAN(v) && F_ABS(v) >= 1.175494351E-38; }

#else

static __inline bool ISINF(float x) { return __builtin_isinf(x); }
static __inline bool ISINF(double x) { return __builtin_isinf(x); }

static __inline bool ISNAN(float x) { return __builtin_isnan(x); }
static __inline bool ISNAN(double x) { return __builtin_isnan(x); }
static __inline int8v ISNAN(mssFloat8 x) { return simd::isnan(x); }
static __inline int4v ISNAN(mssFloat4 x) { return simd::isnan(x); }
static __inline int2v ISNAN(mssFloat2 x) { return simd::isnan(x); }

static __inline int ISNORMf(float v) { return __builtin_isnormal(v); }
static __inline int ISNORMd(double v) { return __builtin_isnormal(v); }

#endif

#define IS_NORMAL_DECL(TYPE) \
__inline bool is_normal(const TYPE &val) \
{ \
    TYPE isnan = convertvector( (-ISNAN(val)) + (-ISINF(val)), TYPE ); \
    return (__builtin_reduce_max( isnan ) == 0.0); \
}

IS_NORMAL_DECL(float8v)
IS_NORMAL_DECL(float4v)
IS_NORMAL_DECL(float2v)
IS_NORMAL_DECL(double8v)
IS_NORMAL_DECL(double4v)
IS_NORMAL_DECL(double2v)

#define SANITIZE_DENORMAL_DECL(TYPE) \
__inline TYPE sanitize_denormal(const TYPE &val) \
{ \
    return (is_normal( val )) ? val : 0.0; \
}

SANITIZE_DENORMAL_DECL(float8v)
SANITIZE_DENORMAL_DECL(float4v)
SANITIZE_DENORMAL_DECL(float2v)
SANITIZE_DENORMAL_DECL(double8v)
SANITIZE_DENORMAL_DECL(double4v)
SANITIZE_DENORMAL_DECL(double2v)

template <typename T>
static __inline int sanitize_denormals( T *out, int count, const char *str = "" )
{
    int present = 0;
    
    while (count--) {
        T s = *out;
        if (is_normal(s))
            *out++ = s;
        else {
            *out++ = 0.0;
            present++;
        }
    }
 //   if (present > 0)
   //     LOGGER("%s: baddies found: %d", str, present);
    return present;
}


#define P_STR(x) #x
#define P_STRINGIFY(x) P_STR(x)

#define PRINT_D1(v) fprintf(stderr, "%s(D1): [%e]\n", P_STRINGIFY(v), v);
#define PRINT_D2(v) fprintf(stderr, "%s(D2): [%.3f %.3f]\n", P_STRINGIFY(v), v[0], v[1]);
#define PRINT_D8(v) fprintf(stderr, "%s(D8): [%e %e %e %e %e %e %e %e]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
#define PRINT_D4(v) fprintf(stderr, "%s(D4): [%e %e %e %e]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3]);
#define PRINT_I4(v) fprintf(stderr, "%s(I4): [%d %d %d %d]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3]);
#define PRINT_I8(v) fprintf(stderr, "%s(I8): [%d %d %d %d %d %d %d %d]\n", P_STRINGIFY(v), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

#define getName(var)            #var
#define CHECK_NULL(x)           if (!(x)) { fprintf(stderr, "CHECK_NULL: %s\n",getName(x)); raise(SIGINT); }

#define SQ2_2                   0.7071067811865475244
#define CMPLX_MAG(X,Y)          F_SQRT( (X) * (X) + (Y) * (Y) )

#ifndef D_CMPLX_T
#define D_CMPLX_T


template <typename T>
struct cmplxT {
    using T1 = SimdBase<T>;
    // Data members
    T re, im;
    
    // Constructors
    cmplxT() : re(0.0), im(0.0) {}
    cmplxT(T r, T i) : re(r), im(i) {}
    cmplxT(const cmplxT& v) : re(v.re), im(v.im) {}
    cmplxT(long double v) : re(v), im(v) {}
    cmplxT(const T1& v) : re(v), im(v) {}
    cmplxT(T r, double i) : re(r), im(i) {} // Added for yy[0], 0.0 case
    
    
    static inline cmplxT<T> polar(T mag, T phase) {
        return cmplxT<T>(mag * F_COS(phase), mag * F_SIN(phase));
    }
    
    // Conversion operator
    operator cmplxT<T>() const { return cmplxT(re, im); }
    
    // Assignment operators - don't use const return type
    inline cmplxT<T>& operator = (const cmplxT<T>& v) {
        re = v.re;
        im = v.im;
        return *this;
    }
    
    inline cmplxT<T>& operator = (const T1& v) {
        re = v;
        im = v;
        return *this;
    }
    
    // Binary arithmetic operators - should not modify *this
    inline cmplxT<T> operator * (const T& x) const {
        return cmplxT<T>(re * x, im * x);
    }
    
    inline cmplxT<T> operator * (const long double x) const {
        return cmplxT<T>(re * x, im * x);
    }
    
    inline cmplxT<T> operator * (const cmplxT<T>& x) const {
        return cmplxT<T>(re * x.re - im * x.im, re * x.im + im * x.re);
    }
    
    inline cmplxT<T> operator + (const cmplxT<T>& x) const {
        return cmplxT<T>(re + x.re, im + x.im);
    }
    
    inline cmplxT<T> operator - () const {
        return cmplxT<T>(-re, -im);
    }
    
    inline cmplxT<T> operator - (const cmplxT<T>& d) const {
        return *this = *this - d; //cmplxT<T>(re - d.re, im - d.im);
    }
    
    inline cmplxT<T> operator / (const cmplxT<T>& x) const {
        T denominator = (x.re * x.re + x.im * x.im);
        return cmplxT<T>(
            (re * x.re + im * x.im) / denominator,
            (im * x.re - re * x.im) / denominator
        );
    }
    
    inline cmplxT<T> operator & (const int8v& mask) const {
        return cmplxT<T>(shufflevector(re, mask), shufflevector(im, mask));
    }
    
    // Compound assignment operators - should not be const
    inline const cmplxT<T>& operator *= (const long double d) {
        return *this = *this * d;
    }
    
    inline const cmplxT<T>& operator *= (const T& d) {
        return *this = *this * d;
    }
    
    inline const cmplxT<T>& operator *= (const cmplxT<T>& x) {
        return *this = *this * x;
    }
    
    inline const cmplxT<T>& operator += (const cmplxT<T>& x) {
        return *this = *this + x;
    }
    
    inline const cmplxT<T>& operator -= (const cmplxT<T>& x) {
        return *this = *this - x;
    }
    
    // Index operator - must return by value, not reference
  //  inline cmplxT<T1> operator [](const int i) const {
  //      return cmplxT<T1>(re[i], im[i]);
  //  }
    
    // Utility methods
    inline T length() const { return F_SQRT(re * re + im * im); }
    inline T mag() const { return F_SQRT(re * re + im * im); }
    inline T phase() const { return F_ATAN2(im, re); }
    inline T real() const { return re; }
    inline T imag() const { return im; }
};

#endif //D_CMPLX_T


#define MALLOC_V4SF_ALIGNMENT   64
#define VALIGNED_ID             'VLGN'

static void *Valigned_malloc(size_t nb_bytes)
{
    void *p, *p0 = malloc(nb_bytes + MALLOC_V4SF_ALIGNMENT);
    if (!p0) return (void *) 0;
    p = (void *) (((size_t) p0 + MALLOC_V4SF_ALIGNMENT) & (~((size_t) (MALLOC_V4SF_ALIGNMENT-1))));
    *(uint32_t*)((void **) p - 1) = VALIGNED_ID;
    *((void **) p - 2) = p0;
    return p;
}

static void Valigned_free(void *p)
{
    if (p && *(uint32_t*)((void **) p - 1) == VALIGNED_ID) {
        void *ptr = *((void**) p - 2);
      //  int64_t sz = bigpool.malloc_size(ptr);
        free( ptr );
      //  if ((sz/1048576L) == 0)
      //      fprintf(stderr, "Valigned_free( %3lld kb)\n", (int64_t)(sz/1024));
      //  else
      //      fprintf(stderr, "Valigned_free( %3lld MB )\n", (int64_t)(sz/1048576L));
    }
}

static inline void *Valigned_ptr(void *p)
{
    return (p) ? *((void **) p - 1) : NULL;
}

template <typename T>
static inline T *callocT( int64_t k )
{
    T *p = (T*)Valigned_malloc (k * sizeof(T));
    if (!p)
        return NULL;
    memset (p, 0, k * sizeof (T));
    return p;
}

static inline void callocT_free( void *p )
{
    Valigned_free(p);
}

#define THROW_MEM(cond)  if (cond) throw (Converror (Converror::MEM_ALLOC));



inline float64x2x4_t vaddq_f64_4(float64x2x4_t a, float64x2x4_t b) {
    float64x2x4_t sum;
    sum.val[0] = vaddq_f64(a.val[0], b.val[0]);
    sum.val[1] = vaddq_f64(a.val[1], b.val[1]);
    sum.val[2] = vaddq_f64(a.val[2], b.val[2]);
    sum.val[3] = vaddq_f64(a.val[3], b.val[3]);
    return sum;
}

inline float64x2x4_t vsubq_f64_4(float64x2x4_t a, float64x2x4_t b) {
    float64x2x4_t sum;
    sum.val[0] = vsubq_f64(a.val[0], b.val[0]);
    sum.val[1] = vsubq_f64(a.val[1], b.val[1]);
    sum.val[2] = vsubq_f64(a.val[2], b.val[2]);
    sum.val[3] = vsubq_f64(a.val[3], b.val[3]);
    return sum;
}

inline float64x2x4_t vmulq_f64_4(float64x2x4_t a, float64x2x4_t b) {
    float64x2x4_t sum;
    sum.val[0] = vmulq_f64(a.val[0], b.val[0]);
    sum.val[1] = vmulq_f64(a.val[1], b.val[1]);
    sum.val[2] = vmulq_f64(a.val[2], b.val[2]);
    sum.val[3] = vmulq_f64(a.val[3], b.val[3]);
    return sum;
}

inline float64x2x4_t vmlaq_f64_4(float64x2x4_t a, float64x2x4_t b, float64x2x4_t c) {
    float64x2x4_t sum;
    sum.val[0] = vmlaq_f64(a.val[0], b.val[0], c.val[0]);
    sum.val[1] = vmlaq_f64(a.val[1], b.val[1], c.val[1]);
    sum.val[2] = vmlaq_f64(a.val[2], b.val[2], c.val[2]);
    sum.val[3] = vmlaq_f64(a.val[3], b.val[3], c.val[3]);
    return sum;
}

inline float64x2x4_t vnegq_f64_4(float64x2x4_t a) {
    a.val[0] = vnegq_f64(a.val[0]);
    a.val[1] = vnegq_f64(a.val[1]);
    a.val[2] = vnegq_f64(a.val[2]);
    a.val[3] = vnegq_f64(a.val[3]);
    return a;
}


inline float64x2x4_t vdupq_f64_4(double x) {
    float64x2_t val = vdupq_n_f64(x);
    float64x2x4_t sum;
    sum.val[0] = val;
    sum.val[1] = val;
    sum.val[2] = val;
    sum.val[3] = val;
    return sum;
}



inline float32x4x2_t vaddq_f32_4(float32x4x2_t a, float32x4x2_t b) {
    float32x4x2_t sum;
    sum.val[0] = vaddq_f32(a.val[0], b.val[0]);
    sum.val[1] = vaddq_f32(a.val[1], b.val[1]);
    sum.val[2] = vaddq_f32(a.val[2], b.val[2]);
    sum.val[3] = vaddq_f32(a.val[3], b.val[3]);
    return sum;
}

inline float32x4x2_t vsubq_f32_4(float32x4x2_t a, float32x4x2_t b) {
    float32x4x2_t sum;
    sum.val[0] = vsubq_f32(a.val[0], b.val[0]);
    sum.val[1] = vsubq_f32(a.val[1], b.val[1]);
    sum.val[2] = vsubq_f32(a.val[2], b.val[2]);
    sum.val[3] = vsubq_f32(a.val[3], b.val[3]);
    return sum;
}

inline float32x4x2_t vmulq_f32_4(float32x4x2_t a, float32x4x2_t b) {
    float32x4x2_t sum;
    sum.val[0] = vmulq_f32(a.val[0], b.val[0]);
    sum.val[1] = vmulq_f32(a.val[1], b.val[1]);
    sum.val[2] = vmulq_f32(a.val[2], b.val[2]);
    sum.val[3] = vmulq_f32(a.val[3], b.val[3]);
    return sum;
}

inline float32x4x2_t vmlaq_f32_4(float32x4x2_t a, float32x4x2_t b, float32x4x2_t c) {
    float32x4x2_t sum;
    sum.val[0] = vmlaq_f32(a.val[0], b.val[0], c.val[0]);
    sum.val[1] = vmlaq_f32(a.val[1], b.val[1], c.val[1]);
    sum.val[2] = vmlaq_f32(a.val[2], b.val[2], c.val[2]);
    sum.val[3] = vmlaq_f32(a.val[3], b.val[3], c.val[3]);
    return sum;
}

inline float32x4x2_t vnegq_f32_4(float32x4x2_t a) {
    a.val[0] = vnegq_f32(a.val[0]);
    a.val[1] = vnegq_f32(a.val[1]);
    a.val[2] = vnegq_f32(a.val[2]);
    a.val[3] = vnegq_f32(a.val[3]);
    return a;
}

inline float32x4x2_t vdupq_f32_4(float x) {
    float32x4_t val = vdupq_n_f32(x);
    float32x4x2_t sum;
    sum.val[0] = val;
    sum.val[1] = val;
    sum.val[2] = val;
    sum.val[3] = val;
    return sum;
}




