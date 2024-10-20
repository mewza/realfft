//  realfft.h - A highly optimized C++ SIMD vector templated class
//  --- 
//  FFTReal v1.1 (C) 2024 Dmitry Boldyrev <subband@gmail.com>
//  (C) 2024 Laurent de Soras <ldesoras@club-internet.fr>
//  Object Pascal port (C) 2024 Frederic Vanmol <frederic@fruityloops.com>

#pragma once
#include "const1.h"

#define USE_NEON

#ifdef USE_NEON
#include <simd/simd.h>
#endif


template <typename T, int N>
class FFTReal
{
    using T1 = SimdBase<T>;
    
    static constexpr unsigned floorlog2(unsigned x) {
        return (x == 1) ? 0 : 1 + floorlog2(x >> 1);
    }
    
    static const int nbr_bits = floorlog2( N );
    static const int N2 = N >> 1;
    
protected:
    
    alignas(64) T   buffer_ptr[ (N + 1) * 2 ];
    alignas(64) T   yy[ (N + 1) * 2 ];
    
public:
    
    // ========================================================================== //
    //      Description: Compute the real FFT of the array.                       //
    //                                                                            //
    //      Input parameters:                                                     //
    //        - f: pointer on the source array (time)                             //
    //                                                                            //
    //      Output parameters:                                                    //
    //        - x: pointer on the destination array (frequencies)                 //
    //             in [0...N(x)] = interleaved format R0,I0,R1,I1,R2,I2,          //
    // ========================================================================== //
    
    void real_fft(const T* x, cmplxT<T>* y, bool do_scale = false)
    {
        T1 mul = 1.0;
#ifdef USE_NEON
        if constexpr( std::is_same_v<T, simd_double8> ) {
            do_fft_neon_d8(x, yy);
        } else if constexpr( std::is_same_v<T, simd_float8> )
            do_fft_neon_f8(x, yy);
        else
            do_fft(x, yy);
#else
        do_fft(x, yy);
#endif
        if (do_scale) mul *= 1.0/(T1)N;
        
        for (int i=1; i < N2; i++) {
            y[i] = cmplxT<T>(yy[i], yy[i + N2]) * mul;
        }
        y[0] = cmplxT<T>(yy[0], 0.0) * mul;
    }
    
    // ========================================================================== //
    //      Description: Compute the inverse real FFT of the array. Notice that   //
    //                   IFFT (FFT (x)) = x * N (x). Data must be                 //
    //                   post-scaled.                                             //
    //                                                                            //
    //      Input parameters:                                                     //
    //        - f: pointer on the source array (frequencies).                     //
    //             in [0...N(x)] = interleaved format R0,I0,R1,I1,R2,I2,          //
    //                                                                            //
    //      Output parameters:                                                    //
    //        - x: pointer on the destination array (time).                       //
    // ========================================================================== //
    
    void real_ifft(const cmplxT<T>* x, T* y, bool do_scale = false)
    {
        for (int i=1; i < N2; i++) {
            yy[ i      ] = x[i].re;
            yy[ i + N2 ] = x[i].im;
        }
        yy[  0 ] = x[0].re;
        yy[ N2 ] = 0.0;
        
#ifdef USE_NEON
        if constexpr( std::is_same_v<T, simd_double8> ) {
            do_ifft_neon_d8(yy, y);
        } else if constexpr( std::is_same_v<T, simd_float8> )
            do_ifft_neon_f8(yy, y);
        else
            do_ifft(yy, y, do_scale);
#else
        do_ifft(yy, y, do_scale);
#endif
        do_ifft(yy, y, do_scale);
    }
    
    void do_fft(const T *x, T *f)
    {
        if (nbr_bits > 2)
        {
            T *sf, *df;
            
            if (nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }
            
            //  First and second pass at once
            
            auto lut_ptr = _bit_rev_lut.get_ptr();
            for (auto i = 0; i < N; i += 4)
            {
                auto df2 = &df [i];
                auto lut = &lut_ptr [i];
                
                auto x0 = x[ lut[0] ];
                auto x1 = x[ lut[1] ];
                auto x2 = x[ lut[2] ];
                auto x3 = x[ lut[3] ];
                
                df2[0] = x0 + x1 + x2 + x3;
                df2[1] = x0 - x1;
                df2[2] = x0 + x1 - x2 - x3;
                df2[3] = x2 - x3;
            }
            
            //  Third pass
            
            for (auto i = 0; i < N; i += 8)
            {
                auto sf2 = &sf [i];
                auto df2 = &df [i];
                
                sf2 [0] = df2 [0] + df2 [4];
                sf2 [4] = df2 [0] - df2 [4];
                sf2 [2] = df2 [2];
                sf2 [6] = df2 [6];
                
                T v = (df2 [5] - df2 [7]) * SQ2_2;
                sf2 [1] = df2 [1] + v;
                sf2 [3] = df2 [1] - v;
                
                v = (df2 [5] + df2 [7]) * SQ2_2;
                sf2 [5] = v + df2 [3];
                sf2 [7] = v - df2 [3];
            }
            
            //  Next pass
            
            for (auto pass = 3; pass < nbr_bits; ++pass)
            {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                auto cos_ptr = _trigo_lut.get_ptr(pass);
                
                for (auto i = 0; i < N; i += d_nbr_coef)
                {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;
                    
                    //  Extreme coefficients are always real
                    
                    dfr [0] = sf1r [0] + sf2r [0];
                    dfi [0] = sf1r [0] - sf2r [0];                  // dfr [nbr_coef] =
                    dfr [h_nbr_coef] = sf1r [h_nbr_coef];
                    dfi [h_nbr_coef] = sf2r [h_nbr_coef];
                    
                    //  Others are conjugate complex numbers
                    
                    auto sf1i = &sf1r [h_nbr_coef];
                    auto sf2i = &sf1i [nbr_coef];
                    
                    for (int j = 1; j < h_nbr_coef; ++j)
                    {
                        const T1 c = cos_ptr [j];                // cos (i*PI/nbr_coef);
                        const T1 s = cos_ptr [h_nbr_coef - j];   // sin (i*PI/nbr_coef);
                        
                        T v = sf2r [j] * c - sf2i [j] * s;
                        dfr [ j] = sf1r [j] + v;
                        dfi [-j] = sf1r [j] - v;                // dfr [nbr_coef - i] =
                        
                        v = sf2r [j] * s + sf2i [j] * c;
                        dfi [j] = v + sf1i [j];
                        dfi [nbr_coef - j] = v - sf1i [j];
                    }
                }
                
                //  Prepare to the next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        
        //  -- Special cases --
        
        //  4-point FFT
        else if (nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const T b_0 = x[0] + x[2];
            const T b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        //  2-point FFT
        else if (nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        //  1-point FFT
        else {
            f[0] = x[0];
        }
    }
    
    void do_ifft(const T *f, T *x, bool do_scale = false)
    {
        const T1 c2 = 2.0;
        
        T1 mul = 1.;
        if (do_scale) mul *= 1./(T1)N;
        
        //  General case
        
        if (nbr_bits > 2)
        {
            T * sf = (T*) f;
            T * df;
            T * df_temp;
            
            if (nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }
            
            // Do the transformation in several pass
            
            // First pass
            
            for (auto pass = nbr_bits - 1; pass >= 3; --pass)
            {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                auto cos_ptr = _trigo_lut.get_ptr (pass);
                
                for (auto i = 0; i < N; i += d_nbr_coef)
                {
                    auto sfr = &sf [i];
                    auto sfi = &sfr [nbr_coef];
                    auto df1r = &df [i];
                    auto df2r = &df1r [nbr_coef];
                    
                    // Extreme coefficients are always real
                    
                    df1r [0] = sfr [0] + sfr [nbr_coef];
                    df2r [0] = sfr [0] - sfr [nbr_coef];
                    df1r [h_nbr_coef] = sfr [h_nbr_coef] * c2;
                    df2r [h_nbr_coef] = sfi [h_nbr_coef] * c2;
                    
                    // Others are conjugate complex numbers
                    
                    auto df1i = &df1r [h_nbr_coef];
                    auto df2i = &df1i [nbr_coef ];
                    
                    for (auto i = 1; i < h_nbr_coef; ++i)
                    {
                        df1r [i] = sfr [i] + sfi [-i];          // + sfr [nbr_coef - i]
                        df1i [i] = sfi [i] - sfi [nbr_coef - i];
                        
                        auto c = cos_ptr [i];
                        auto s = cos_ptr [h_nbr_coef - i];
                        
                        auto vr = sfr [i] - sfi [-i];           // - sfr [nbr_coef - i];
                        auto vi = sfi [i] + sfi [nbr_coef - i];
                        
                        df2r [i] = vr * c + vi * s;
                        df2i [i] = vi * c - vr * s;
                    }
                }
                
                // Prepare to the next pass
                
                if (pass < nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }
            
            // Antepenultimate pass
            const T sq2_2 = SQ2_2;
            for (auto i = 0; i < N; i += 8)
            {
                auto df2 = &df [i];
                auto sf2 = &sf [i];
                
                auto vr = sf2 [1] - sf2 [3];
                auto vi = sf2 [5] + sf2 [7];
                
                df2 [0] = sf2 [0] + sf2 [4];
                df2 [1] = sf2 [1] + sf2 [3];
                df2 [2] = sf2 [2] * c2;
                df2 [3] = sf2 [5] - sf2 [7];
                df2 [4] = sf2 [0] - sf2 [4];
                df2 [5] = (vr + vi) * sq2_2;
                df2 [6] = sf2 [6] * c2;
                df2 [7] = (vi - vr) * sq2_2;
            }
            
            // Penultimate and last pass at once
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
            for (auto i = 0; i < N; i += 8)
            {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                {   auto b_0 = sf2[0] + sf2[2];
                    auto b_2 = sf2[0] - sf2[2];
                    auto b_1 = sf2[1] * c2;
                    auto b_3 = sf2[3] * c2;
                    
                    x[lut[0]] = (b_0 + b_1) * mul;
                    x[lut[1]] = (b_0 - b_1) * mul;
                    x[lut[2]] = (b_2 + b_3) * mul;
                    x[lut[3]] = (b_2 - b_3) * mul;
                }
                {   auto b_0 = sf2[4] + sf2[6];
                    auto b_2 = sf2[4] - sf2[6];
                    auto b_1 = sf2[5] * c2;
                    auto b_3 = sf2[7] * c2;
                    
                    x[lut[4]] = (b_0 + b_1) * mul;
                    x[lut[5]] = (b_0 - b_1) * mul;
                    x[lut[6]] = (b_2 + b_3) * mul;
                    x[lut[7]] = (b_2 - b_3) * mul;
                }
            }
        }
        
        //   Special cases
        
        // 4-point IFFT
        else if (nbr_bits == 2) {
            const T b_0 = f[0] + f[2];
            const T b_2 = f[0] - f[2];
            x[0] = (b_0 + f[1] * c2) * mul;
            x[2] = (b_0 - f[1] * c2) * mul;
            x[1] = (b_2 + f[3] * c2) * mul;
            x[3] = (b_2 - f[3] * c2) * mul;
        }
        // 2-point IFFT
        else if (nbr_bits == 1) {
            x[0] = (f[0] + f[1]) * mul;
            x[1] = (f[0] - f[1]) * mul;
        }
        // 1-point IFFT
        else {
            x[0] = f[0] * mul;
        }
    }
    
protected:
    
    
    inline void do_rescale(T *x, int len) const
    {
        const T1 mul = 1./(T1)N;
        for (auto i = 0; i < len; ++i)
            x[i] *= mul;
    }
    
    inline void do_rescale(cmplxT<T> *x, int len) const
    {
        const T1 mul = 1./(T1)N;
        
        for (auto i = 0; i < len; ++i)
            x[i] *= mul;
    }
    
#ifdef USE_NEON
    
    void do_fft_neon_d8(const simd_double8 *x, simd_double8 *f)
    {
        if (nbr_bits > 2) {
            simd_double8 *sf, *df;
            
            if (nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }
            
            // First stage: bit-reversal and initial butterfly computation
            auto bit_rev_lut_ptr = _bit_rev_lut.get_ptr();
            
            for (int i = 0; i < N; i += 4) {
                auto df2 = &df[i];
                auto lut = &bit_rev_lut_ptr[i];
                float64x2x4_t x0 = vld4q_f64((double*) &x[lut[0]]);
                float64x2x4_t x1 = vld4q_f64((double*) &x[lut[1]]);
                float64x2x4_t x2 = vld4q_f64((double*) &x[lut[2]]);
                float64x2x4_t x3 = vld4q_f64((double*) &x[lut[3]]);
                
                // Calculate the sums and differences
                float64x2x4_t sum_0 = vaddq_f64_4(x0, x1); // x0 + x1
                float64x2x4_t sum_2 = vaddq_f64_4(x2, x3); // x2 + x3
    
                float64x2x4_t diff_0 = vsubq_f64_4(x0, x1); // x0 - x1
                float64x2x4_t diff_2 = vsubq_f64_4(x2, x3); // x2 - x3
                
                vst4q_f64((double*)&df2[0], vaddq_f64_4(sum_0, sum_2)); // Total sum
                vst4q_f64((double*)&df2[1], diff_0); // x0 - x1
                vst4q_f64((double*)&df2[2], vsubq_f64_4(sum_0, sum_2)); // x0 + x1 - (x2 + x3)
                vst4q_f64((double*)&df2[3], diff_2); // x2 - x3
            }
            
            // Third pass
            float64x2x4_t _SQ2_2 = vdupq_f64_4(SQ2_2);
            
            for (auto i = 0; i < N; i += 8) {
                auto sf2 = &sf[i];
                auto df2 = &df[i];
                
                float64x2x4_t   f0 = vld4q_f64((double*)&df2[0]), f1 = vld4q_f64((double*)&df2[1]),
                                f2 = vld4q_f64((double*)&df2[2]), f3 = vld4q_f64((double*)&df2[3]),
                                f4 = vld4q_f64((double*)&df2[4]), f5 = vld4q_f64((double*)&df2[5]),
                                f6 = vld4q_f64((double*)&df2[6]), f7 = vld4q_f64((double*)&df2[7]);
                
                vst4q_f64((double*)&sf2[0], vaddq_f64_4(f0, f4));
                vst4q_f64((double*)&sf2[4], vsubq_f64_4(f0, f4));
                vst4q_f64((double*)&sf2[2], f2);
                vst4q_f64((double*)&sf2[6], f6);
                
                float64x2x4_t v1 = vsubq_f64_4(f5, f7);
                v1 = vmulq_f64_4(v1, _SQ2_2);
                vst4q_f64((double*)&sf2[1], vaddq_f64_4(f1, v1));
                vst4q_f64((double*)&sf2[3], vsubq_f64_4(f1, v1));
                
                float64x2x4_t v2 = vaddq_f64_4(f5, f7);
                v2 = vmulq_f64_4(v2, _SQ2_2);
                vst4q_f64((double*)&sf2[5], vaddq_f64_4(v2, f3));
                vst4q_f64((double*)&sf2[7], vsubq_f64_4(v2, f3));
            }
            
            for (auto pass = 3; pass < nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                const double *cos_ptr = _trigo_lut.get_ptr(pass);
                for (auto i = 0; i < N; i += d_nbr_coef) {
                    simd_double8 * const sf1r = sf + i;
                    simd_double8 * const sf2r = sf1r + nbr_coef;
                    simd_double8 * const dfr = df + i;
                    simd_double8 * const dfi = dfr + nbr_coef;
                    
                    float64x2x4_t sf1r0_0 = vld4q_f64((double*)&sf1r[0]);
                    float64x2x4_t sf2r0_0 = vld4q_f64((double*)&sf2r[0]);
                    
                    dfr[0] = sf1r[0] + sf2r[0];
                    dfi[0] = sf1r[0] - sf2r[0]; // dfr[nbr_coef] =
                    dfr[h_nbr_coef] = sf1r[h_nbr_coef];
                    dfi[h_nbr_coef] = sf2r[h_nbr_coef];
                    
                    // Extreme coefficients are always real
                    vst4q_f64((double*)&dfr[0], vaddq_f64_4(sf1r0_0, sf2r0_0));
                    vst4q_f64((double*)&dfi[0], vsubq_f64_4(sf1r0_0, sf2r0_0));
                    
                    dfr[h_nbr_coef] = sf1r[h_nbr_coef];
                    dfi[h_nbr_coef] = sf2r[h_nbr_coef];
                    
                    // Others are conjugate complex numbers
                    const simd_double8 * const sf1i = &sf1r[h_nbr_coef];
                    const simd_double8 * const sf2i = &sf1i[nbr_coef];
                    
                    for (int j = 1; j < h_nbr_coef; ++j)
                    {
                        // Load cosine and sine values into NEON registers
                        float64x2x4_t c = vdupq_f64_4(cos_ptr[j]); // Load the same value in both lanes
                        float64x2x4_t s = vdupq_f64_4(cos_ptr[h_nbr_coef-j]); // Same for sine
                        
                        // Calculate v using NEON operations
                        float64x2x4_t sf2r0_j = vld4q_f64((double*) &sf2r[j]);
                        float64x2x4_t sf2i0_j = vld4q_f64((double*) &sf2i[j]);
                        
                        // v = sf2r[j] * c - sf2i[j] * s
                        float64x2x4_t v_0 = vmlaq_f64_4(vnegq_f64_4(vmulq_f64_4(sf2i0_j, s)), sf2r0_j, c);
                       
                        float64x2x4_t sf1r0_j = vld4q_f64((double*) &sf1r[j]);
                        
                        vst4q_f64((double*) &dfr[ j], vaddq_f64_4(sf1r0_j, v_0));
                        vst4q_f64((double*) &dfi[-j], vsubq_f64_4(sf1r0_j, v_0)); // dfi[nbr_coef - j]
                        
                        float64x2x4_t sf1i0_j = vld4q_f64((double*) &sf1i[j]);
                        
                        // v = sf2r[j] * s + sf2i[j] * c
                        v_0 = vmlaq_f64_4(vmulq_f64_4(sf2i0_j, c), sf2r0_j, s);
                        
                        vst4q_f64((double*) &dfi[j], vaddq_f64_4(v_0, sf1i0_j));
                        vst4q_f64((double*) &dfi[nbr_coef-j], vsubq_f64_4(v_0, sf1i0_j));
                    }
                }
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        
        // -- Special cases --
        // 4-point FFT
        else if (nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_double8 b_0 = x[0] + x[2];
            const simd_double8 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        // 2-point FFT
        else if (nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        // 1-point FFT
        else {
            f[0] = x[0];
        }
    }
    
    
    void do_fft_neon_f8(const simd_float8 *x, simd_float8 *f)
    {
        if (nbr_bits > 2) {
            simd_float8 *sf, *df;
            
            if (nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }
            
            // First stage: bit-reversal and initial butterfly computation
            auto bit_rev_lut_ptr = _bit_rev_lut.get_ptr();
            
            for (int i = 0; i < N; i += 4) {
                auto df2 = &df[i];
                auto lut = &bit_rev_lut_ptr[i];
                float32x4x2_t x0 = vld2q_f32((float*) &x[lut[0]]);
                float32x4x2_t x1 = vld2q_f32((float*) &x[lut[1]]);
                float32x4x2_t x2 = vld2q_f32((float*) &x[lut[2]]);
                float32x4x2_t x3 = vld2q_f32((float*) &x[lut[3]]);
                
                // Calculate the sums and differences
                float32x4x2_t sum_0 = vaddq_f32_4(x0, x1); // x0 + x1
                float32x4x2_t sum_2 = vaddq_f32_4(x2, x3); // x2 + x3
    
                float32x4x2_t diff_0 = vsubq_f32_4(x0, x1); // x0 - x1
                float32x4x2_t diff_2 = vsubq_f32_4(x2, x3); // x2 - x3
                
                vst2q_f32((float*)&df2[0], vaddq_f32_4(sum_0, sum_2)); // Total sum
                vst2q_f32((float*)&df2[1], diff_0); // x0 - x1
                vst2q_f32((float*)&df2[2], vsubq_f32_4(sum_0, sum_2)); // x0 + x1 - (x2 + x3)
                vst2q_f32((float*)&df2[3], diff_2); // x2 - x3
            }
            
            // Third pass
            float32x4x2_t _SQ2_2 = vdupq_f32_4(SQ2_2);
            
            for (auto i = 0; i < N; i += 8) {
                auto sf2 = &sf[i];
                auto df2 = &df[i];
                
                float32x4x2_t   f0 = vld2q_f32((float*)&df2[0]), f1 = vld2q_f32((float*)&df2[1]),
                                f2 = vld2q_f32((float*)&df2[2]), f3 = vld2q_f32((float*)&df2[3]),
                                f4 = vld2q_f32((float*)&df2[4]), f5 = vld2q_f32((float*)&df2[5]),
                                f6 = vld2q_f32((float*)&df2[6]), f7 = vld2q_f32((float*)&df2[7]);
                
                vst2q_f32((float*)&sf2[0], vaddq_f32_4(f0, f4));
                vst2q_f32((float*)&sf2[4], vsubq_f32_4(f0, f4));
                vst2q_f32((float*)&sf2[2], f2);
                vst2q_f32((float*)&sf2[6], f6);
                
                float32x4x2_t v1 = vsubq_f32_4(f5, f7);
                v1 = vmulq_f32_4(v1, _SQ2_2);
                vst2q_f32((float*)&sf2[1], vaddq_f32_4(f1, v1));
                vst2q_f32((float*)&sf2[3], vsubq_f32_4(f1, v1));
                
                float32x4x2_t v2 = vaddq_f32_4(f5, f7);
                v2 = vmulq_f32_4(v2, _SQ2_2);
                vst2q_f32((float*)&sf2[5], vaddq_f32_4(v2, f3));
                vst2q_f32((float*)&sf2[7], vsubq_f32_4(v2, f3));
            }
            
            for (auto pass = 3; pass < nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                const auto cos_ptr = _trigo_lut.get_ptr(pass);
                for (auto i = 0; i < N; i += d_nbr_coef) {
                    simd_float8 * const sf1r = sf + i;
                    simd_float8 * const sf2r = sf1r + nbr_coef;
                    simd_float8 * const dfr = df + i;
                    simd_float8 * const dfi = dfr + nbr_coef;
                    
                    float32x4x2_t sf1r0_0 = vld2q_f32((float*)&sf1r[0]);
                    float32x4x2_t sf2r0_0 = vld2q_f32((float*)&sf2r[0]);
                    
                    dfr[0] = sf1r[0] + sf2r[0];
                    dfi[0] = sf1r[0] - sf2r[0]; // dfr[nbr_coef] =
                    dfr[h_nbr_coef] = sf1r[h_nbr_coef];
                    dfi[h_nbr_coef] = sf2r[h_nbr_coef];
                    
                    // Extreme coefficients are always real
                    vst2q_f32((float*)&dfr[0], vaddq_f32_4(sf1r0_0, sf2r0_0));
                    vst2q_f32((float*)&dfi[0], vsubq_f32_4(sf1r0_0, sf2r0_0));
                    
                    dfr[h_nbr_coef] = sf1r[h_nbr_coef];
                    dfi[h_nbr_coef] = sf2r[h_nbr_coef];
                    
                    // Others are conjugate complex numbers
                    const simd_float8 * const sf1i = &sf1r[h_nbr_coef];
                    const simd_float8 * const sf2i = &sf1i[nbr_coef];
                    
                    for (int j = 1; j < h_nbr_coef; ++j)
                    {
                        // Load cosine and sine values into NEON registers
                        float32x4x2_t c = vdupq_f32_4(cos_ptr[j]); // Load the same value in both lanes
                        float32x4x2_t s = vdupq_f32_4(cos_ptr[h_nbr_coef-j]); // Same for sine
                        
                        // Calculate v using NEON operations
                        float32x4x2_t sf2r0_j = vld2q_f32((float*) &sf2r[j]);
                        float32x4x2_t sf2i0_j = vld2q_f32((float*) &sf2i[j]);
                        
                        // v = sf2r[j] * c - sf2i[j] * s
                        float32x4x2_t v_0 = vmlaq_f32_4(vnegq_f32_4(vmulq_f32_4(sf2i0_j, s)), sf2r0_j, c);
                       
                        float32x4x2_t sf1r0_j = vld2q_f32((float*) &sf1r[j]);
                        
                        vst2q_f32((float*) &dfr[ j], vaddq_f32_4(sf1r0_j, v_0));
                        vst2q_f32((float*) &dfi[-j], vsubq_f32_4(sf1r0_j, v_0)); // dfi[nbr_coef - j]
                        
                        float32x4x2_t sf1i0_j = vld2q_f32((float*) &sf1i[j]);
                        
                        // v = sf2r[j] * s + sf2i[j] * c
                        v_0 = vmlaq_f32_4(vmulq_f32_4(sf2i0_j, c), sf2r0_j, s);
                        
                        vst2q_f32((float*) &dfi[j], vaddq_f32_4(v_0, sf1i0_j));
                        vst2q_f32((float*) &dfi[nbr_coef-j], vsubq_f32_4(v_0, sf1i0_j));
                    }
                }
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        
        // -- Special cases --
        // 4-point FFT
        else if (nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_float8 b_0 = x[0] + x[2];
            const simd_float8 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        // 2-point FFT
        else if (nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        // 1-point FFT
        else {
            f[0] = x[0];
        }
    }
    

    void do_ifft_neon_d8(const simd_double8 *f, simd_double8 *x)
    {
        const double c2 = 2.0;
        
        //  General case
        
        if (nbr_bits > 2)
        {
            simd_double8 * sf = (T*) f;
            simd_double8 * df;
            simd_double8 * df_temp;
            
            if (nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }
            
            // Do the transformation in several pass
            
            // First pass
            
            for (auto pass = nbr_bits - 1; pass >= 3; --pass)
            {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                auto cos_ptr = _trigo_lut.get_ptr (pass);
                
                for (auto i = 0; i < N; i += d_nbr_coef)
                {
                    auto sfr = &sf [i];
                    auto sfi = &sfr [nbr_coef];
                    auto df1r = &df [i];
                    auto df2r = &df1r [nbr_coef];
                    
                    // Extreme coefficients are always real
                    
                    df1r [0] = sfr [0] + sfr [nbr_coef];
                    df2r [0] = sfr [0] - sfr [nbr_coef];
                    df1r [h_nbr_coef] = sfr [h_nbr_coef] * c2;
                    df2r [h_nbr_coef] = sfi [h_nbr_coef] * c2;
                    
                    // Others are conjugate complex numbers
                    
                    auto df1i = &df1r [h_nbr_coef];
                    auto df2i = &df1i [nbr_coef ];
                    
                    for (auto i = 1; i < h_nbr_coef; ++i)
                    {
                        float64x2x4_t sfr_i    = vld4q_f64((double*) &sfr [i]);
                        float64x2x4_t sfi_mi   = vld4q_f64((double*) &sfi [-i]);
                        float64x2x4_t sfi_i    = vld4q_f64((double*) &sfi [i]);
                        float64x2x4_t sfi_nbr  = vld4q_f64((double*) &sfi [nbr_coef - i]);
                        
                        vst4q_f64((double*) &df1r [i], vaddq_f64_4(sfr_i, sfi_mi));
                        vst4q_f64((double*) &df1i [i], vsubq_f64_4(sfi_i, sfi_nbr));
                        
                        float64x2x4_t c = vdupq_f64_4(cos_ptr [i]);
                        float64x2x4_t s = vdupq_f64_4(cos_ptr [h_nbr_coef - i]);
                        
                        float64x2x4_t vr = vsubq_f64_4(sfr_i, sfi_mi); // - sfr [nbr_coef - i];
                        float64x2x4_t vi = vaddq_f64_4(sfi_i, sfi_nbr);
                        
                        vst4q_f64((double*) &df2r [i], vmlaq_f64_4(vmulq_f64_4(vr, c), vi, s));
                        vst4q_f64((double*) &df2i [i], vmlaq_f64_4(vnegq_f64_4(vmulq_f64_4(vr, s)), vi, c));
                    }
                }
                
                // Prepare to the next pass
                
                if (pass < nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }
            
            // Antepenultimate pass
            
            float64x2x4_t c2 = vdupq_f64_4(2.0);
            float64x2x4_t sq2_2 = vdupq_f64_4(SQ2_2);
            
            for (auto i = 0; i < N; i += 8)
            {
                auto sf2 = &sf [i], df2 = &df [i];
                
                float64x2x4_t   f0 = vld4q_f64((double*) &sf2 [0]), f1 = vld4q_f64((double*) &sf2 [1]),
                f2 = vld4q_f64((double*) &sf2 [2]), f3 = vld4q_f64((double*) &sf2 [3]),
                f4 = vld4q_f64((double*) &sf2 [4]), f5 = vld4q_f64((double*) &sf2 [5]),
                f6 = vld4q_f64((double*) &sf2 [6]), f7 = vld4q_f64((double*) &sf2 [7]);
                
                vst4q_f64((double*) &df2 [0], vaddq_f64_4(f0, f4));
                vst4q_f64((double*) &df2 [4], vsubq_f64_4(f0, f4));
                vst4q_f64((double*) &df2 [2], vmulq_f64_4(f2, c2));
                vst4q_f64((double*) &df2 [6], vmulq_f64_4(f6, c2));
                
                vst4q_f64((double*) &df2 [1], vaddq_f64_4(f1, f3));
                vst4q_f64((double*) &df2 [3], vsubq_f64_4(f5, f7));
                
                float64x2x4_t vr = vsubq_f64_4(f1, f3);
                float64x2x4_t vi = vaddq_f64_4(f5, f7);
                
                vst4q_f64((double*) &df2 [5], vmulq_f64_4(vaddq_f64_4(vr, vi), sq2_2));
                vst4q_f64((double*) &df2 [7], vmulq_f64_4(vsubq_f64_4(vi, vr), sq2_2));
            }
            
            // Penultimate and last pass at once
            
            const int * lut = _bit_rev_lut.get_ptr();
            const simd_double8 * sf2 = df;
            
            for (auto i = 0; i < N; i += 8)
            {
                float64x2x4_t   sf2_0 = vld4q_f64((double*) &sf2 [0]),
                                sf2_1 = vld4q_f64((double*) &sf2 [1]),
                                sf2_2 = vld4q_f64((double*) &sf2 [2]),
                                sf2_3 = vld4q_f64((double*) &sf2 [3]),
                                sf2_4 = vld4q_f64((double*) &sf2 [4]),
                                sf2_5 = vld4q_f64((double*) &sf2 [5]),
                                sf2_6 = vld4q_f64((double*) &sf2 [6]),
                                sf2_7 = vld4q_f64((double*) &sf2 [7]);
                
                
                float64x2x4_t   b_0 = vaddq_f64_4(sf2_0, sf2_2),
                                b_2 = vsubq_f64_4(sf2_0, sf2_2),
                                b_1 = vmulq_f64_4(sf2_1, c2),
                                b_3 = vmulq_f64_4(sf2_3, c2),
                                b_4 = vaddq_f64_4(sf2_4, sf2_6),
                                b_7 = vsubq_f64_4(sf2_4, sf2_6),
                                b_5 = vmulq_f64_4(sf2_5, c2),
                                b_6 = vmulq_f64_4(sf2_7, c2);
                
                vst4q_f64((double*) &x[lut[0]], vaddq_f64_4(b_0, b_1));
                vst4q_f64((double*) &x[lut[1]], vsubq_f64_4(b_0, b_1));
                vst4q_f64((double*) &x[lut[2]], vaddq_f64_4(b_2, b_3));
                vst4q_f64((double*) &x[lut[3]], vsubq_f64_4(b_2, b_3));
                
                vst4q_f64((double*) &x[lut[4]], vaddq_f64_4(b_4, b_5));
                vst4q_f64((double*) &x[lut[5]], vsubq_f64_4(b_4, b_5));
                vst4q_f64((double*) &x[lut[6]], vaddq_f64_4(b_6, b_7));
                vst4q_f64((double*) &x[lut[7]], vsubq_f64_4(b_6, b_7));
                
                sf2 += 8;
                lut += 8;
            }
        }
        
        //   Special cases
        
        // 4-point IFFT
        else if (nbr_bits == 2) {
            auto b_0 = f[0] + f[2];
            auto b_2 = f[0] - f[2];
            
            x[0] = b_0 + f[1] * c2;
            x[2] = b_0 - f[1] * c2;
            x[1] = b_2 + f[3] * c2;
            x[3] = b_2 - f[3] * c2;
        }
        // 2-point IFFT
        else if (nbr_bits == 1) {
            x[0] = f[0] + f[1];
            x[1] = f[0] - f[1];
        }
        // 1-point IFFT
        else {
            x[0] = f[0];
        }
    }
    
    
    void do_ifft_neon_f8(const simd_float8 *f, simd_float8 *x)
    {
        const float c2 = 2.0f;
        
        //  General case
        
        if (nbr_bits > 2)
        {
            simd_float8 * sf = (T*) f;
            simd_float8 * df;
            simd_float8 * df_temp;
            
            if (nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }
            
            // Do the transformation in several pass
            
            // First pass
            
            for (auto pass = nbr_bits - 1; pass >= 3; --pass)
            {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                auto cos_ptr = _trigo_lut.get_ptr (pass);
                
                for (auto i = 0; i < N; i += d_nbr_coef)
                {
                    auto sfr = &sf [i];
                    auto sfi = &sfr [nbr_coef];
                    auto df1r = &df [i];
                    auto df2r = &df1r [nbr_coef];
                    
                    // Extreme coefficients are always real
                    
                    df1r [0] = sfr [0] + sfr [nbr_coef];
                    df2r [0] = sfr [0] - sfr [nbr_coef];
                    df1r [h_nbr_coef] = sfr [h_nbr_coef] * c2;
                    df2r [h_nbr_coef] = sfi [h_nbr_coef] * c2;
                    
                    // Others are conjugate complex numbers
                    
                    auto df1i = &df1r [h_nbr_coef];
                    auto df2i = &df1i [nbr_coef ];
                    
                    for (auto i = 1; i < h_nbr_coef; ++i)
                    {
                        float32x4x2_t sfr_i    = vld2q_f32((float*) &sfr [i]);
                        float32x4x2_t sfi_mi   = vld2q_f32((float*) &sfi [-i]);
                        float32x4x2_t sfi_i    = vld2q_f32((float*) &sfi [i]);
                        float32x4x2_t sfi_nbr  = vld2q_f32((float*) &sfi [nbr_coef - i]);
                        
                        vst2q_f32((float*) &df1r [i], vaddq_f32_4(sfr_i, sfi_mi));
                        vst2q_f32((float*) &df1i [i], vsubq_f32_4(sfi_i, sfi_nbr));
                        
                        float32x4x2_t c = vdupq_f32_4(cos_ptr [i]);
                        float32x4x2_t s = vdupq_f32_4(cos_ptr [h_nbr_coef - i]);
                        
                        float32x4x2_t vr = vsubq_f32_4(sfr_i, sfi_mi); // - sfr [nbr_coef - i];
                        float32x4x2_t vi = vaddq_f32_4(sfi_i, sfi_nbr);
                        
                        vst2q_f32((float*) &df2r [i], vmlaq_f32_4(vmulq_f32_4(vr, c), vi, s));
                        vst2q_f32((float*) &df2i [i], vmlaq_f32_4(vnegq_f32_4(vmulq_f32_4(vr, s)), vi, c));
                    }
                }
                
                // Prepare to the next pass
                
                if (pass < nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }
            
            // Antepenultimate pass
            
            float32x4x2_t c2 = vdupq_f32_4(2.0);
            float32x4x2_t sq2_2 = vdupq_f32_4(SQ2_2);
            
            for (auto i = 0; i < N; i += 8)
            {
                auto sf2 = &sf [i], df2 = &df [i];
                
                float32x4x2_t   f0 = vld2q_f32((float*) &sf2 [0]), f1 = vld2q_f32((float*) &sf2 [1]),
                f2 = vld2q_f32((float*) &sf2 [2]), f3 = vld2q_f32((float*) &sf2 [3]),
                f4 = vld2q_f32((float*) &sf2 [4]), f5 = vld2q_f32((float*) &sf2 [5]),
                f6 = vld2q_f32((float*) &sf2 [6]), f7 = vld2q_f32((float*) &sf2 [7]);
                
                vst2q_f32((float*) &df2 [0], vaddq_f32_4(f0, f4));
                vst2q_f32((float*) &df2 [4], vsubq_f32_4(f0, f4));
                vst2q_f32((float*) &df2 [2], vmulq_f32_4(f2, c2));
                vst2q_f32((float*) &df2 [6], vmulq_f32_4(f6, c2));
                
                vst2q_f32((float*) &df2 [1], vaddq_f32_4(f1, f3));
                vst2q_f32((float*) &df2 [3], vsubq_f32_4(f5, f7));
                
                float32x4x2_t vr = vsubq_f32_4(f1, f3);
                float32x4x2_t vi = vaddq_f32_4(f5, f7);
                
                vst2q_f32((float*) &df2 [5], vmulq_f32_4(vaddq_f32_4(vr, vi), sq2_2));
                vst2q_f32((float*) &df2 [7], vmulq_f32_4(vsubq_f32_4(vi, vr), sq2_2));
            }
            
            // Penultimate and last pass at once
            
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
            
            for (auto i = 0; i < N; i += 8)
            {
                auto df2 = &df [i];
                auto lut = &lut_ptr [i];
                float32x4x2_t   sf2_0 = vld2q_f32((float*) &df2 [0]),
                                sf2_1 = vld2q_f32((float*) &df2 [1]),
                                sf2_2 = vld2q_f32((float*) &df2 [2]),
                                sf2_3 = vld2q_f32((float*) &df2 [3]),
                                sf2_4 = vld2q_f32((float*) &df2 [4]),
                                sf2_5 = vld2q_f32((float*) &df2 [5]),
                                sf2_6 = vld2q_f32((float*) &df2 [6]),
                                sf2_7 = vld2q_f32((float*) &df2 [7]);
                
                float32x4x2_t   b_0 = vaddq_f32_4(sf2_0, sf2_2),
                                b_2 = vsubq_f32_4(sf2_0, sf2_2),
                                b_1 = vmulq_f32_4(sf2_1, c2),
                                b_3 = vmulq_f32_4(sf2_3, c2),
                
                                b_4 = vaddq_f32_4(sf2_4, sf2_6),
                                b_7 = vsubq_f32_4(sf2_4, sf2_6),
                                b_5 = vmulq_f32_4(sf2_5, c2),
                                b_6 = vmulq_f32_4(sf2_7, c2);
                
                vst2q_f32((float*) &x[lut[0]], vaddq_f32_4(b_0, b_1));
                vst2q_f32((float*) &x[lut[1]], vsubq_f32_4(b_0, b_1));
                vst2q_f32((float*) &x[lut[2]], vaddq_f32_4(b_2, b_3));
                vst2q_f32((float*) &x[lut[3]], vsubq_f32_4(b_2, b_3));
                
                vst2q_f32((float*) &x[lut[4]], vaddq_f32_4(b_4, b_5));
                vst2q_f32((float*) &x[lut[5]], vsubq_f32_4(b_4, b_5));
                vst2q_f32((float*) &x[lut[6]], vaddq_f32_4(b_6, b_7));
                vst2q_f32((float*) &x[lut[7]], vsubq_f32_4(b_6, b_7));
            }
        }
        
        //   Special cases
        
        // 4-point IFFT
        else if (nbr_bits == 2) {
            auto b_0 = f[0] + f[2];
            auto b_2 = f[0] - f[2];
            
            x[0] = b_0 + f[1] * c2;
            x[2] = b_0 - f[1] * c2;
            x[1] = b_2 + f[3] * c2;
            x[3] = b_2 - f[3] * c2;
        }
        // 2-point IFFT
        else if (nbr_bits == 1) {
            x[0] = f[0] + f[1];
            x[1] = f[0] - f[1];
        }
        // 1-point IFFT
        else {
            x[0] = f[0];
        }
    }

#endif // USE_NEON

    
    // Bit-reversed look-up table nested class
    class BitReversedLUT
    {
    public:
        
        const int *get_ptr () const {
            return _ptr;
        }
        
        BitReversedLUT()
        {
            int cnt, br_index, bit;
        
            _ptr = new int [N];
            
            br_index = 0;
            _ptr[0] = 0;
            for (cnt=1; cnt < N; ++cnt)
            {
                // ++br_index (bit reversed)
                bit = N >> 1;
                while (((br_index ^= bit) & bit) == 0)
                    bit >>= 1;
                
                _ptr[cnt] = br_index;
            }
        }
        
        ~BitReversedLUT() {
            if (_ptr) delete[] _ptr;
        }
        
    private:
        int *_ptr = NULL;
    };
    
    // Trigonometric look-up table nested class
    class TrigoLUT
    {
    public:
        
        
        const T1 * get_baseptr() const {
            return _ptr;
        }
        
        const T1 * get_ptr(const int level) const
        {
            return (_ptr + (1 << (level - 1)) - 4);
        }
        
        // ========================================================================== //
        //      Input parameters:                                                     //
        //        - nbr_bits: number of bits of the array on which we want to do a    //
        //                    FFT. Range: > 0                                         //
        //      Throws: std::bad_alloc, anything                                      //
        // ========================================================================== //
        
        TrigoLUT()
        {
            if (nbr_bits > 3)
            {
                int total_len = (1 << (nbr_bits - 1)) - 4;
                _ptr = new T1 [total_len];
                
                for (int level=3; level < nbr_bits; ++level)
                {
                    const int level_len = 1 << (level - 1);
                    T1 * const level_ptr = const_cast<T1 *>(get_ptr(level));
                    const T1 mul = M_PI / T1(level_len << 1);
                    
                    for (int i=0; i < level_len; ++i)
                        level_ptr[i] = F_COS(i * mul);
                }
            }
        }
        ~TrigoLUT()
        {
            if (_ptr) delete [] _ptr;
        }
        
    private:
        T1 *_ptr = NULL;
    };
    
protected:
    
    const BitReversedLUT _bit_rev_lut;
    const TrigoLUT       _trigo_lut;
    
};



