//  realfft.h - A highly optimized C++ SIMD vector T templated FFT/iFFT transform
//  ---
//  FFTReal v1.4 (C) 2025 Dmitry Boldyrev <subband@gmail.com>
//  Pascal version (C) 2024 Laurent de Soras <ldesoras@club-internet.fr>
//  Object Pascal port (C) 2024 Frederic Vanmol <frederic@fruityloops.com>
//
//  NOTE: I decided to give away the highly optimized NEON code!  Enjoy, humanity,
//        and get well! 


#pragma once

#include <TargetConditionals.h>
#include <memory>
#include "const1.h"

#if !TARGET_OS_MACCATALYST && TARGET_CPU_ARM64 && defined(__ARM_NEON)

#include <arm_neon.h>
#define USE_NEON
#define NEW_NEON_OPT 1

#endif // NEON

template <typename T>
class FFTReal
{
    using T1 = SimdBase<T>;
    
    int _nbr_bits = 0;
    int _N2 = 0;
    int _N = 0;
    
    static constexpr unsigned floorlog2(unsigned x) {
        return (x == 1) ? 0 : 1 + floorlog2(x >> 1);
    }
    
protected:
    // Change fixed arrays to pointers
    T*_Nullable buffer_ptr = nullptr;
    T*_Nullable yy = nullptr;
    T*_Nullable xx = nullptr;
    
public:
    FFTReal(const int n) :
    _N(n),
    _N2(n/2),
    _nbr_bits(floorlog2(n))
    {
        buffer_ptr = alignedAlloc<T>((n + 1) * 2);
        yy = alignedAlloc<T>((n + 1) * 2);
        xx = alignedAlloc<T>(n);
        
        _trigo_lut = std::make_unique<trigo_lookup>(_nbr_bits);
        _bit_rev_lut = std::make_unique<bit_rev_lut>(_nbr_bits, _N);
        _twiddle_cache = std::make_unique<twiddle_cache>(_nbr_bits, _N2);
    }
    
    ~FFTReal() {
        alignedFree(buffer_ptr);
        alignedFree(yy);
        alignedFree(xx);
    }
    
    void reset() {
        if (buffer_ptr) {
            memset(buffer_ptr, 0, sizeof(T) * (_N + 1) * 2);
        }
        if (yy) {
            memset(yy, 0, sizeof(T) * (_N + 1) * 2);
        }
        if (xx) {
            memset(xx, 0, sizeof(T) * _N);
        }
    }
    
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
    
    void real_fft(const T*_Nonnull x, cmplxT<T>*_Nonnull y, bool do_scale = false)
    {
        T mul = 1.0;
        cmplxT<T> c;
        if (do_scale) {
            const T mul = 0.5;
            for (int i=0; i < _N; i++) {
                xx[i] = x[i] * mul;
            }
        } else {
            memcpy(xx, x, _N * sizeof(T));
        }
#if USE_NEON
        if constexpr( std::is_same_v<T, simd_double8> ) {
            do_fft_neon_d8(xx, yy);
        } else if constexpr( std::is_same_v<T, simd_float8> )
            do_fft_neon_f8(xx, yy);
        if constexpr( std::is_same_v<T, simd_double4> ) {
            do_fft_neon_d4(xx, yy);
        } else if constexpr( std::is_same_v<T, simd_float4> )
            do_fft_neon_f4(xx, yy);
        if constexpr( std::is_same_v<T, simd_double2> ) {
            do_fft_neon_d2(xx, yy);
        } else if constexpr( std::is_same_v<T, simd_float2> )
            do_fft_neon_f2(xx, yy);
        if constexpr( std::is_same_v<T, double> ) {
            do_fft_neon_d1(xx, yy);
        } else if constexpr( std::is_same_v<T, float> )
            do_fft_neon_f1(xx, yy);
        else
            do_fft(xx, yy);
#else
        do_fft(xx, yy);
#endif
        if (do_scale) mul *= 1./(T1)_N;
        
        y[0] = cmplxT<T>(yy[0], 0.0) * mul;
        for (int i=1; i < _N2; i++) {
            y[i] = cmplxT<T>(yy[i], yy[i + _N2]) * mul;
        }
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
    
    void real_ifft(const cmplxT<T>*_Nonnull x, T*_Nonnull y, bool do_scale = false)
    {
        for (int i=1; i < _N2; i++) {
            yy[ i       ] = x[i].re;
            yy[ i + _N2 ] = x[i].im;
        }
        yy[   0 ] = x[0].re;
        yy[ _N2 ] = 0.0;
        
#if USE_NEON
        if constexpr( std::is_same_v<T, simd_double8> ) {
            do_ifft_neon_d8(yy, y, do_scale);
        } else if constexpr( std::is_same_v<T, simd_float8> )
            do_ifft_neon_f8(yy, y, do_scale);
        if constexpr( std::is_same_v<T, simd_double4> ) {
            do_ifft_neon_d4(yy, y, do_scale);
        } else if constexpr( std::is_same_v<T, simd_float4> )
            do_ifft_neon_f4(yy, y, do_scale);
        if constexpr( std::is_same_v<T, simd_double2> ) {
            do_ifft_neon_d2(yy, y, do_scale);
        } else if constexpr( std::is_same_v<T, simd_float2> )
            do_ifft_neon_f2(yy, y, do_scale);
        if constexpr( std::is_same_v<T, double> ) {
            do_ifft_neon_d1(yy, y, do_scale);
        } else if constexpr( std::is_same_v<T, float> )
            do_ifft_neon_f1(yy, y, do_scale);
        else
            do_ifft(yy, y, do_scale);
#else
        do_ifft(yy, y, do_scale);
#endif
    }
    
    void do_fft(const T *_Nonnull x, T *_Nonnull f)
    {
        T1 c, s;
        if (_nbr_bits > 2) {
            T *sf, *df;
            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }
            
            //  First and second pass at once
            
            auto lut = _bit_rev_lut->get_ptr();
            for (auto i = 0; i < _N; i += 4)
            {
                auto df2 = &df [i];
                auto x0 = x[ lut[i] ];
                auto x1 = x[ lut[i+1] ];
                auto x2 = x[ lut[i+2] ];
                auto x3 = x[ lut[i+3] ];
                
                df2[0] = x0 + x1 + x2 + x3;
                df2[1] = x0 - x1;
                df2[2] = x0 + x1 - x2 - x3;
                df2[3] = x2 - x3;
            }
            
            //  Third pass
            
            for (auto i = 0; i < _N; i += 8)
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
            
            for (auto pass = 3; pass < _nbr_bits; ++pass)
            {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                // No longer using _trigo_lut directly
                // auto cos_ptr = _trigo_lut.get_ptr(pass);
                
                for (auto i = 0; i < _N; i += d_nbr_coef)
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
                        // Using twiddle_cache instead of direct access
                        // const T1 c = cos_ptr [j];                // cos (i*PI/nbr_coef);
                        // const T1 s = cos_ptr [h_nbr_coef - j];   // sin (i*PI/nbr_coef);
                        
                        _twiddle_cache->get_twiddle(pass, j, c, s);
                        
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
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const T b_0 = x[0] + x[2];
            const T b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        //  2-point FFT
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        //  1-point FFT
        else {
            f[0] = x[0];
        }
    }
    
    void do_ifft(const T *_Nonnull f, T *_Nonnull x, bool do_scale = false)
    {
        T1 c, s;
        const T1 c2 = 2.0;
        
        T1 mul = 1.;
        if (do_scale) mul *= 1./(T1)_N;
        
        //  General case
        
        if (_nbr_bits > 2)
        {
            T * sf = (T*) f;
            T * df;
            T * df_temp;
            
            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }
            
            // Do the transformation in several pass
            
            // First pass
            
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass)
            {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                // No longer using _trigo_lut directly
                // auto cos_ptr = _trigo_lut.get_ptr (pass);
                
                for (auto i = 0; i < _N; i += d_nbr_coef)
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
                    
                    for (auto j = 1; j < h_nbr_coef; ++j)
                    {
                        df1r [j] = sfr [j] + sfi [-j];  // + sfr [nbr_coef - j]
                        df1i [j] = sfi [j] - sfi [nbr_coef - j];
                        
                        // Using twiddle_cache instead of direct access
                        // auto c = cos_ptr [j];
                        // auto s = cos_ptr [h_nbr_coef - j];
                        
                        _twiddle_cache->get_twiddle(pass, j, c, s);
                        
                        auto vr = sfr [j] - sfi [-j];           // - sfr [nbr_coef - j];
                        auto vi = sfi [j] + sfi [nbr_coef - j];
                        
                        df2r [j] = vr * c + vi * s;
                        df2i [j] = vi * c - vr * s;
                    }
                }
                
                // Prepare to the next pass
                if (pass < _nbr_bits - 1) {
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
            for (auto i = 0; i < _N; i += 8)
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
            auto lut_ptr = _bit_rev_lut->get_ptr();
            for (auto i = 0; i < _N; i += 8)
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
        else if (_nbr_bits == 2) {
            const T b_0 = f[0] + f[2];
            const T b_2 = f[0] - f[2];
            x[0] = (b_0 + f[1] * c2) * mul;
            x[2] = (b_0 - f[1] * c2) * mul;
            x[1] = (b_2 + f[3] * c2) * mul;
            x[3] = (b_2 - f[3] * c2) * mul;
        }
        // 2-point IFFT
        else if (_nbr_bits == 1) {
            x[0] = (f[0] + f[1]) * mul;
            x[1] = (f[0] - f[1]) * mul;
        }
        // 1-point IFFT
        else {
            x[0] = f[0] * mul;
        }
    }

    FFTReal(FFTReal&& other) noexcept :
        _nbr_bits(other._nbr_bits),
        _N2(other._N2),
        _N(other._N),
        buffer_ptr(other.buffer_ptr),
        yy(other.yy),
        xx(other.xx),
        _trigo_lut(std::move(other._trigo_lut)),
        _bit_rev_lut(std::move(other._bit_rev_lut)),
        _twiddle_cache(std::move(other._twiddle_cache))
    {
        other.buffer_ptr = nullptr;
        other.yy = nullptr;
        other.xx = nullptr;
    }
    
    // Implement move assignment
    FFTReal& operator=(FFTReal&& other) noexcept {
        if (this != &other) {
            alignedFree(buffer_ptr);
            alignedFree(yy);
            alignedFree(xx);
            
            _nbr_bits = other._nbr_bits;
            _N2 = other._N2;
            _N = other._N;
            buffer_ptr = other.buffer_ptr;
            yy = other.yy;
            xx = other.xx;
            _trigo_lut = std::move(other._trigo_lut);
            _bit_rev_lut = std::move(other._bit_rev_lut);
            _twiddle_cache = std::move(other._twiddle_cache);
            
            other.buffer_ptr = nullptr;
            other.yy = nullptr;
            other.xx = nullptr;
        }
        return *this;
    }
    
    // Delete copy constructor and assignment to avoid double-free issues
    FFTReal(const FFTReal&) = delete;
    FFTReal& operator=(const FFTReal&) = delete;
   
    inline void do_rescale(T *_Nonnull x) const {
        const T1 mul = 1./(T1)_N;
        for (auto i = 0; i < _N; ++i)
            x[i] *= mul;
    }
    
    inline void do_rescale(cmplxT<T> *_Nonnull x) const {
        const T mul = 1./(T1)_N;
        x[0] = cmplxT<T>(x[0].re, 0.0) * mul;
        for (auto i = 1; i < _N2; ++i)
            x[i] *= mul;
    }
    
    template <typename U>
    static U*_Nullable alignedAlloc(size_t size) {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(size * sizeof(U), 64);
#else
        posix_memalign(&ptr, 64, size * sizeof(U));
#endif
        return static_cast<U*>(ptr);
    }
    
    // Helper function for aligned deallocation
    static void alignedFree(void*_Nonnull ptr) {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
    }
    
protected:
    
    class trigo_lookup {
    protected:
        int*_Nullable offsets = nullptr;
        T1*_Nullable cos_data = nullptr;
        int nbr_bits = 0;
    public:
        trigo_lookup(int bits) : nbr_bits(bits) {
            int total_coef = 0;
            
            // Allocate with alignment
            offsets = FFTReal::alignedAlloc<int>(nbr_bits + 1);
            
            // Calculate total size needed
            for (int pass = 0; pass < nbr_bits; pass++) {
                offsets[pass] = total_coef;
                int nbr_coef = 1 << pass;
                total_coef += nbr_coef;
            }
            offsets[nbr_bits] = total_coef;
            
            cos_data = FFTReal::alignedAlloc<T1>(total_coef);
            
            // Calculate and store trig values in flat array
            for (int pass = 0; pass < nbr_bits; pass++) {
                int nbr_coef = 1 << pass;
                int offset = offsets[pass];
                
                // Calculate and store the cosine values
                for (int i = 0; i < nbr_coef; i++) {
                    cos_data[offset + i] = F_COS((i * M_PI) / nbr_coef);
                }
            }
        }
        
        ~trigo_lookup() {
            FFTReal::alignedFree(offsets);
            FFTReal::alignedFree(cos_data);
        }
        
        // Get pointer to the cosine values for a specific pass
        inline const T1*_Nullable get_ptr(int pass) const {
            return &cos_data[ offsets[pass] ];
        }
        
        trigo_lookup(trigo_lookup&& other) noexcept :
            nbr_bits(other.nbr_bits),
            offsets(other.offsets),
            cos_data(other.cos_data)
        {
            other.nbr_bits = 0;
            other.offsets = nullptr;
            other.cos_data = nullptr;
        }
        
        // Implement move assignment
        trigo_lookup& operator=(trigo_lookup&& other) noexcept {
            if (this != &other) {
                alignedFree(offsets);
                alignedFree(cos_data);
                
                nbr_bits = other.nbr_bits;
                offsets = std::move(other.offsets);
                cos_data = std::move(other.cos_data);
                
                nbr_bits = 0;
                other.offsets = nullptr;
                other.cos_data = nullptr;
            }
            return *this;
        }
        
        // Delete copy constructor and assignment to avoid double-free issues
        trigo_lookup(const trigo_lookup&) = delete;
        trigo_lookup& operator=(const trigo_lookup&) = delete;
    };

    class bit_rev_lut {
    protected:
        int *_Nullable indices{nullptr};
        int N{0}, nbr_bits{0};
    public:
        bit_rev_lut(int bits, int n) : nbr_bits(bits), N(n) {
            indices = FFTReal::alignedAlloc<int>(N);
            for (int i = 0; i < N; i++) {
                int rev = 0;
                for (int j = 0; j < nbr_bits; j++) {
                    if (i & (1 << j)) rev |= (1 << (nbr_bits - 1 - j));
                }
                indices[i] = rev;
            }
        }
        inline const int*_Nullable get_ptr() const {
            return indices;
        }
        
        bit_rev_lut(bit_rev_lut&& other) noexcept :
            nbr_bits(other.nbr_bits),
            N(other.N),
            indices(std::move(other.indices))
        {
            other.nbr_bits = 0;
            other.N = 0;
            other.indices = nullptr;
        }
        
        // Implement move assignment
        bit_rev_lut& operator=(bit_rev_lut&& other) noexcept {
            if (this != &other) {
                alignedFree(indices);
                
                nbr_bits = other.nbr_bits;
                N = other.N;
                indices = std::move(other.indices);
                
                nbr_bits = 0;
                N = 0;
                other.indices = nullptr;
            }
            return *this;
        }
        
        // Delete copy constructor and assignment to avoid double-free issues
        bit_rev_lut(const bit_rev_lut&) = delete;
        bit_rev_lut& operator=(const bit_rev_lut&) = delete;
        
        ~bit_rev_lut() {
            FFTReal::alignedFree(indices);
        }
    };
    
    class twiddle_cache {
    protected:
        T1*_Nullable*_Nullable cos_data = nullptr;
        T1*_Nullable*_Nullable sin_data = nullptr;
        int nbr_bits = 0;
        int N2 = 0;
    public:
        twiddle_cache(int bits, int n2): nbr_bits(bits), N2(n2) {
            // Allocate 2D arrays using heap
            cos_data = FFTReal::alignedAlloc<T1*>(nbr_bits);
            sin_data = FFTReal::alignedAlloc<T1*>(nbr_bits);
            
            for (int pass = 0; pass < nbr_bits; pass++) {
                cos_data[pass] = FFTReal::alignedAlloc<T1>(N2);
                sin_data[pass] = FFTReal::alignedAlloc<T1>(N2);
                
                int nbr_coef = 1 << pass;
                int h_nbr_coef = nbr_coef >> 1;
                
                // Skip passes with no coefficients
                if (h_nbr_coef <= 0) continue;
                
                // Initialize for this pass
                for (int j = 0; j < h_nbr_coef && j < N2; j++) {
                    T1 angle = (j * M_PI) / nbr_coef;
                    cos_data[pass][j] = F_COS(angle);
                    sin_data[pass][j] = F_SIN(angle); // F_COS(M_PI/2 - angle);
                }
            }
        }
        
        ~twiddle_cache() {
            if (cos_data) {
                for (int i = 0; i < nbr_bits; i++) {
                    FFTReal::alignedFree(cos_data[i]);
                    FFTReal::alignedFree(sin_data[i]);
                }
                FFTReal::alignedFree(cos_data);
                FFTReal::alignedFree(sin_data);
            }
        }
        
        twiddle_cache(twiddle_cache&& other) noexcept :
            nbr_bits(other.nbr_bits),
            N2(other.N2),
            cos_data(std::move(other.cos_data)),
            sin_data(std::move(other.sin_data))
        {
            other.nbr_bits = 0;
            other.N2 = 0;
            other.offsets = nullptr;
            other.cos_data = nullptr;
        }
        
        // Implement move assignment
        twiddle_cache& operator=(twiddle_cache&& other) noexcept {
            if (this != &other) {
                alignedFree(cos_data);
                alignedFree(sin_data);
                
                nbr_bits = other.nbr_bits;
                N2 = other.N2;
                cos_data = std::move(other.cos_data);
                sin_data = std::move(other.sin_data);
                
                nbr_bits = 0;
                N2 = 0;
                other.cos_data = nullptr;
                other.sin_data = nullptr;
            }
            return *this;
        }
        
        // Delete copy constructor and assignment to avoid double-free issues
        twiddle_cache(const twiddle_cache&) = delete;
        twiddle_cache& operator=(const twiddle_cache&) = delete;
        
        inline void get_twiddle(int pass, int j, T1& cos_val, T1& sin_val) const
        {
            if (pass >= 0 && pass < nbr_bits && j >= 0 && j < N2) {
                cos_val = cos_data[pass][j];
                sin_val = sin_data[pass][j];
            }
        }
    };
    
    // Bit-reversal lookup table for FFT without std::vector
   
    std::unique_ptr<trigo_lookup>   _trigo_lut;
    std::unique_ptr<bit_rev_lut>    _bit_rev_lut;
    std::unique_ptr<twiddle_cache>  _twiddle_cache;

#if USE_NEON
    
#if NEW_NEON_OPT
    typedef union {
        float32x4_t v[2];   // Two NEON registers (4 floats each)
        float       f[8];         // Array of 8 floats for easy access
    } simd_float8;
    
    typedef union {
        float64x2_t v[4];   // Four NEON registers (2 doubles each)
        double      f[8];        // Array of 8 doubles for easy access
    } simd_double8;
    
    // Helper function for prefetching - critical for cache optimization
    inline void prefetch_read(const void *_Nonnull ptr) {
        __builtin_prefetch(ptr, 0, 3);  // 0=read, 3=high temporal locality
    }

    inline void prefetch_write(void *_Nonnull ptr) {
        __builtin_prefetch(ptr, 1, 3);  // 1=write, 3=high temporal locality
    }
    
    // Define likely/unlikely macros for better readability
    #define likely(x)   __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
    
    void do_fft_neon_f1(const float *_Nonnull x, float *_Nonnull f)
    {
        float c, s;
        if (likely(_nbr_bits > 2)) {
            float *sf, *df;

            // Initial prefetch for memory
            __builtin_prefetch(x, 0, 3);
            __builtin_prefetch(_bit_rev_lut->get_ptr(), 0, 3);

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }

            // First/second pass with bit-reversal
            constexpr int PREFETCH_DISTANCE = 8;
            auto lut = _bit_rev_lut->get_ptr();
            
            // Process data in blocks of 4 for NEON optimization
#pragma unroll 2
            for (auto i = 0; i < _N; i += 4) {
                if (likely(i + PREFETCH_DISTANCE < _N)) {
                    __builtin_prefetch(&lut[i + PREFETCH_DISTANCE], 0, 3);
                    __builtin_prefetch(&x[lut[i + PREFETCH_DISTANCE]], 0, 3);
                }

                // Load 4 inputs with bit-reversed indices using NEON
                float32x4_t x0123 = { x[lut[i]], x[lut[i+1]], x[lut[i+2]], x[lut[i+3]] };
                
                // Process 4 elements at a time for first pass butterfly
                float v0 = vgetq_lane_f32(x0123, 0);
                float v1 = vgetq_lane_f32(x0123, 1);
                float v2 = vgetq_lane_f32(x0123, 2);
                float v3 = vgetq_lane_f32(x0123, 3);
                
                // Compute butterfly operations
                df[i]   = v0 + v1 + v2 + v3;
                df[i+1] = v0 - v1;
                df[i+2] = v0 + v1 - v2 - v3;
                df[i+3] = v2 - v3;
            }

            // Third pass
            const float SQ2_2_val = SQ2_2;
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&df[i + 8], 0, 3);
                }
                
                // Load 8 elements in blocks of 4 for NEON processing
                float32x4_t df0_3 = vld1q_f32(&df[i]);
                float32x4_t df4_7 = vld1q_f32(&df[i+4]);
                
                // Extract individual elements
                float df0 = vgetq_lane_f32(df0_3, 0);
                float df1 = vgetq_lane_f32(df0_3, 1);
                float df2 = vgetq_lane_f32(df0_3, 2);
                float df3 = vgetq_lane_f32(df0_3, 3);
                float df4 = vgetq_lane_f32(df4_7, 0);
                float df5 = vgetq_lane_f32(df4_7, 1);
                float df6 = vgetq_lane_f32(df4_7, 2);
                float df7 = vgetq_lane_f32(df4_7, 3);
                
                // Compute butterfly operations
                sf[i]   = df0 + df4;
                sf[i+4] = df0 - df4;
                sf[i+2] = df2;
                sf[i+6] = df6;
                
                // Optimized SQ2_2 calculations
                float v = (df5 - df7) * SQ2_2_val;
                sf[i+1] = df1 + v;
                sf[i+3] = df1 - v;
                
                v = (df5 + df7) * SQ2_2_val;
                sf[i+5] = v + df3;
                sf[i+7] = v - df3;
            }

            // Later passes with twiddle factors
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;

                    // Prefetch for next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients
                    float sf1r_0 = sf1r[0];
                    float sf2r_0 = sf2r[0];
                    
                    dfr[0] = sf1r_0 + sf2r_0;
                    dfi[0] = sf1r_0 - sf2r_0;
                    
                    dfr[h_nbr_coef] = sf1r[h_nbr_coef];
                    dfi[h_nbr_coef] = sf2r[h_nbr_coef];

                    // Process other conjugate complex numbers
                    auto sf1i = &sf1r[h_nbr_coef];
                    auto sf2i = &sf1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    float c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache->get_twiddle(pass, 1, c_next, s_next);
                    }

                    // Process conjugate numbers in blocks for NEON when possible
                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        c = c_next;
                        s = s_next;
                        
                        // Prefetch next data and twiddle factors
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sf1r[j + 1], 0, 3);
                            __builtin_prefetch(&sf2r[j + 1], 0, 3);
                            _twiddle_cache->get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load individual scalar values
                        float sf1r_j = sf1r[j];
                        float sf2r_j = sf2r[j];
                        float sf2i_j = sf2i[j];
                        float sf1i_j = sf1i[j];
                        
                        // Calculate v = sf2r[j] * c - sf2i[j] * s
                        float v = sf2r_j * c - sf2i_j * s;
                        
                        dfr[j] = sf1r_j + v;
                        dfi[-j] = sf1r_j - v;

                        // Calculate v = sf2r[j] * s + sf2i[j] * c
                        v = sf2r_j * s + sf2i_j * c;
                        
                        dfi[j] = v + sf1i_j;
                        dfi[nbr_coef - j] = v - sf1i_j;
                    }
                }
                
                // Prepare for next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        // Special cases for small FFTs - direct scalar operations
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const float b_0 = x[0] + x[2];
            const float b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        else {
            f[0] = x[0];
        }
    }
    
    void do_fft_neon_d1(const double *_Nonnull x, double *_Nonnull f)
    {
        double c, s;
        if (likely(_nbr_bits > 2)) {
            double *sf, *df;

            // Initial prefetch
            __builtin_prefetch(x, 0, 3);
            __builtin_prefetch(_bit_rev_lut.get_ptr(), 0, 3);

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }

            // First and second pass with bit-reversal
            constexpr int PREFETCH_DISTANCE = 8;
            auto lut = _bit_rev_lut.get_ptr();
            
            // Process elements in pairs for NEON optimization with doubles
            #pragma unroll 2
            for (auto i = 0; i < _N; i += 4) {
                if (likely(i + PREFETCH_DISTANCE < _N)) {
                    __builtin_prefetch(&lut[i + PREFETCH_DISTANCE], 0, 3);
                    __builtin_prefetch(&x[lut[i + PREFETCH_DISTANCE]], 0, 3);
                }

                // Load individual bit-reversed elements
                double x0 = x[lut[i]];
                double x1 = x[lut[i+1]];
                double x2 = x[lut[i+2]];
                double x3 = x[lut[i+3]];
                
                // Compute butterfly operations directly
                df[i]   = x0 + x1 + x2 + x3;
                df[i+1] = x0 - x1;
                df[i+2] = x0 + x1 - x2 - x3;
                df[i+3] = x2 - x3;
            }

            // Third pass with SQ2_2 optimizations
            const double SQ2_2_val = SQ2_2;
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&df[i + 8], 0, 3);
                }

                // Load elements in batches of 2 for NEON processing
                double df0 = df[i];
                double df1 = df[i+1];
                double df2 = df[i+2];
                double df3 = df[i+3];
                double df4 = df[i+4];
                double df5 = df[i+5];
                double df6 = df[i+6];
                double df7 = df[i+7];
                
                // Compute butterfly values
                sf[i]   = df0 + df4;
                sf[i+4] = df0 - df4;
                sf[i+2] = df2;
                sf[i+6] = df6;
                
                // SQ2_2 calculations
                double v = (df5 - df7) * SQ2_2_val;
                sf[i+1] = df1 + v;
                sf[i+3] = df1 - v;
                
                v = (df5 + df7) * SQ2_2_val;
                sf[i+5] = v + df3;
                sf[i+7] = v - df3;
            }

            // Next passes with twiddle factors
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;

                    // Prefetch for next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients
                    double sf1r_0 = sf1r[0];
                    double sf2r_0 = sf2r[0];
                    
                    dfr[0] = sf1r_0 + sf2r_0;
                    dfi[0] = sf1r_0 - sf2r_0;
                    
                    dfr[h_nbr_coef] = sf1r[h_nbr_coef];
                    dfi[h_nbr_coef] = sf2r[h_nbr_coef];

                    // Process conjugate complex numbers
                    auto sf1i = &sf1r[h_nbr_coef];
                    auto sf2i = &sf1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    double c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        c = c_next;
                        s = s_next;
                        
                        // Prefetch next iteration data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sf1r[j + 1], 0, 3);
                            __builtin_prefetch(&sf2r[j + 1], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load scalar values
                        double sf1r_j = sf1r[j];
                        double sf2r_j = sf2r[j];
                        double sf2i_j = sf2i[j];
                        
                        // Calculate v = sf2r[j] * c - sf2i[j] * s
                        double v = sf2r_j * c - sf2i_j * s;
                        
                        dfr[j] = sf1r_j + v;
                        dfi[-j] = sf1r_j - v;

                        // Calculate v = sf2r[j] * s + sf2i[j] * c
                        double sf1i_j = sf1i[j];
                        v = sf2r_j * s + sf2i_j * c;
                        
                        dfi[j] = v + sf1i_j;
                        dfi[nbr_coef - j] = v - sf1i_j;
                    }
                }
                
                // Prepare for next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        // Special cases for small FFTs - direct scalar operations
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const double b_0 = x[0] + x[2];
            const double b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        else {
            f[0] = x[0];
        }
    }
    
    void do_ifft_neon_f1(const float *_Nonnull f, float *_Nonnull x, bool do_scale = false)
    {
        const float c2 = 2.0f;
        
        // Initialize scaling factor
        float mul = 1.0f;
        if (unlikely(do_scale)) {
            mul = 1.0f / (float)_N;
        }

        if (likely(_nbr_bits > 2)) {
            float *sf = (float*)f;
            float *df;
            float *df_temp;

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }

            // First pass with NEON optimizations where possible
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sfr = &sf[i];
                    auto sfi = &sfr[nbr_coef];
                    auto df1r = &df[i];
                    auto df2r = &df1r[nbr_coef];

                    // Prefetch next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients
                    float sfr_0 = sfr[0];
                    float sfr_nbr_coef = sfr[nbr_coef];
                    
                    df1r[0] = sfr_0 + sfr_nbr_coef;
                    df2r[0] = sfr_0 - sfr_nbr_coef;
                    
                    // Process h_nbr_coef coefficients with c2 multiply
                    df1r[h_nbr_coef] = sfr[h_nbr_coef] * c2;
                    df2r[h_nbr_coef] = sfi[h_nbr_coef] * c2;

                    // Process conjugate complex numbers
                    auto df1i = &df1r[h_nbr_coef];
                    auto df2i = &df1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    float c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache->get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        float c = c_next;
                        float s = s_next;
                        
                        // Prefetch next iteration data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sfr[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[-(j + 1)], 0, 3);
                            _twiddle_cache->get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load scalar values
                        float sfr_j = sfr[j];
                        float sfi_neg_j = sfi[-j];
                        float sfi_j = sfi[j];
                        float sfi_nbr_j = sfi[nbr_coef - j];

                        // Calculate df1r and df1i
                        df1r[j] = sfr_j + sfi_neg_j;
                        df1i[j] = sfi_j - sfi_nbr_j;

                        // Calculate vr and vi
                        float vr = sfr_j - sfi_neg_j;
                        float vi = sfi_j + sfi_nbr_j;

                        // Calculate df2r and df2i
                        df2r[j] = vr * c + vi * s;
                        df2i[j] = vi * c - vr * s;
                    }
                }

                // Prepare for next pass
                if (pass < _nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }

            // Antepenultimate pass with SQ2_2 optimizations
            const float SQ2_2_val = SQ2_2;
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&sf[i + 8], 0, 3);
                }

                // Load scalar values
                float f0 = sf[i];
                float f4 = sf[i+4];
                
                float sum04 = f0 + f4;
                float diff04 = f0 - f4;
                
                // Store sum and difference
                df[i]   = sum04;
                df[i+4] = diff04;
                
                // Process c2 multiplications
                float f2 = sf[i+2];
                float f6 = sf[i+6];
                
                df[i+2] = f2 * c2;
                df[i+6] = f6 * c2;
                
                // Load additional data
                float f1 = sf[i+1];
                float f3 = sf[i+3];
                float f5 = sf[i+5];
                float f7 = sf[i+7];
                
                // Simple additions and subtractions
                df[i+1] = f1 + f3;
                df[i+3] = f5 - f7;
                
                // SQ2_2 calculations
                float vr = f1 - f3;
                float vi = f5 + f7;
                
                df[i+5] = (vr + vi) * SQ2_2_val;
                df[i+7] = (vi - vr) * SQ2_2_val;
            }

            // Penultimate and last pass with bit-reversal
            auto lut_ptr = _bit_rev_lut->get_ptr();
            
            for (auto i = 0; i < _N; i += 8) {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                // Prefetch output locations
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&x[lut[8]], 1, 0);
                }

                // Process first 4 outputs
                {
                    float sf0 = sf2[0];
                    float sf1 = sf2[1];
                    float sf2val = sf2[2];
                    float sf3 = sf2[3];

                    // Butterfly calculation
                    float b_0 = sf0 + sf2val;
                    float b_2 = sf0 - sf2val;
                    float b_1 = sf1 * c2;
                    float b_3 = sf3 * c2;

                    // Apply scaling and store with bit-reversal
                    x[lut[0]] = (b_0 + b_1) * mul;
                    x[lut[1]] = (b_0 - b_1) * mul;
                    x[lut[2]] = (b_2 + b_3) * mul;
                    x[lut[3]] = (b_2 - b_3) * mul;
                }

                // Process second 4 outputs
                {
                    float sf4 = sf2[4];
                    float sf5 = sf2[5];
                    float sf6 = sf2[6];
                    float sf7 = sf2[7];

                    // Butterfly calculation
                    float b_0 = sf4 + sf6;
                    float b_2 = sf4 - sf6;
                    float b_1 = sf5 * c2;
                    float b_3 = sf7 * c2;

                    // Apply scaling and store with bit-reversal
                    x[lut[4]] = (b_0 + b_1) * mul;
                    x[lut[5]] = (b_0 - b_1) * mul;
                    x[lut[6]] = (b_2 + b_3) * mul;
                    x[lut[7]] = (b_2 - b_3) * mul;
                }
            }
        }
        // Special cases for small IFFTs with direct scalar operations
        else if (unlikely(_nbr_bits == 2)) {
            // 4-point IFFT
            const float b_0 = f[0] + f[2];
            const float b_2 = f[0] - f[2];
            const float f1_c2 = f[1] * c2;
            const float f3_c2 = f[3] * c2;
            
            x[0] = (b_0 + f1_c2) * mul;
            x[2] = (b_0 - f1_c2) * mul;
            x[1] = (b_2 + f3_c2) * mul;
            x[3] = (b_2 - f3_c2) * mul;
        }
        else if (unlikely(_nbr_bits == 1)) {
            // 2-point IFFT
            x[0] = (f[0] + f[1]) * mul;
            x[1] = (f[0] - f[1]) * mul;
        }
        else {
            // 1-point IFFT
            x[0] = f[0] * mul;
        }
    }
    
    void do_ifft_neon_d1(const double *_Nonnull f, double *_Nonnull x, bool do_scale = false)
    {
        // Initialize constants
        const double c2 = 2.0;
        
        // Initialize scaling factor
        double mul = 1.0;
        if (unlikely(do_scale)) {
            mul = 1.0 / (double)_N;
        }

        if (likely(_nbr_bits > 2)) {
            double *sf = (double*)f;
            double *df;
            double *df_temp;

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }

            // First pass with NEON optimizations where possible
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sfr = &sf[i];
                    auto sfi = &sfr[nbr_coef];
                    auto df1r = &df[i];
                    auto df2r = &df1r[nbr_coef];

                    // Prefetch next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients with scalar operations
                    double sfr_0 = sfr[0];
                    double sfr_nbr_coef = sfr[nbr_coef];
                    
                    df1r[0] = sfr_0 + sfr_nbr_coef;
                    df2r[0] = sfr_0 - sfr_nbr_coef;
                    
                    // Process h_nbr_coef with c2 multiply
                    df1r[h_nbr_coef] = sfr[h_nbr_coef] * c2;
                    df2r[h_nbr_coef] = sfi[h_nbr_coef] * c2;

                    // Process conjugate complex numbers
                    auto df1i = &df1r[h_nbr_coef];
                    auto df2i = &df1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    double c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        double c = c_next;
                        double s = s_next;
                        
                        // Prefetch next iteration data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sfr[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[-(j + 1)], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load scalar values
                        double sfr_j = sfr[j];
                        double sfi_neg_j = sfi[-j];
                        double sfi_j = sfi[j];
                        double sfi_nbr_j = sfi[nbr_coef - j];

                        // Calculate df1r and df1i values
                        df1r[j] = sfr_j + sfi_neg_j;
                        df1i[j] = sfi_j - sfi_nbr_j;

                        // Calculate vr and vi
                        double vr = sfr_j - sfi_neg_j;
                        double vi = sfi_j + sfi_nbr_j;

                        // Calculate with scalar operations
                        df2r[j] = vr * c + vi * s;
                        df2i[j] = vi * c - vr * s;
                    }
                }

                // Prepare for next pass
                if (pass < _nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }

            // Antepenultimate pass with SQ2_2 optimizations
            const double SQ2_2_val = SQ2_2;
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&sf[i + 8], 0, 3);
                }

                // Load scalar values
                double f0 = sf[i];
                double f4 = sf[i+4];
                
                double sum04 = f0 + f4;
                double diff04 = f0 - f4;
                
                // Store sum and difference
                df[i]   = sum04;
                df[i+4] = diff04;
                
                // Process c2 multiplications
                double f2 = sf[i+2];
                double f6 = sf[i+6];
                
                df[i+2] = f2 * c2;
                df[i+6] = f6 * c2;
                
                // Load additional data
                double f1 = sf[i+1];
                double f3 = sf[i+3];
                double f5 = sf[i+5];
                double f7 = sf[i+7];
                
                // Simple additions and subtractions
                df[i+1] = f1 + f3;
                df[i+3] = f5 - f7;
                
                // SQ2_2 calculations
                double vr = f1 - f3;
                double vi = f5 + f7;
                
                df[i+5] = (vr + vi) * SQ2_2_val;
                df[i+7] = (vi - vr) * SQ2_2_val;
            }

            // Penultimate and last pass with bit-reversal
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
            for (auto i = 0; i < _N; i += 8) {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                // Prefetch output locations
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&x[lut[8]], 1, 0);
                }

                // Process first 4 outputs
                {
                    double sf0 = sf2[0];
                    double sf1 = sf2[1];
                    double sf2val = sf2[2];
                    double sf3 = sf2[3];

                    // Calculate butterfly values
                    double b_0 = sf0 + sf2val;
                    double b_2 = sf0 - sf2val;
                    double b_1 = sf1 * c2;
                    double b_3 = sf3 * c2;

                    // Apply scaling and store with bit-reversal
                    x[lut[0]] = (b_0 + b_1) * mul;
                    x[lut[1]] = (b_0 - b_1) * mul;
                    x[lut[2]] = (b_2 + b_3) * mul;
                    x[lut[3]] = (b_2 - b_3) * mul;
                }

                // Process second 4 outputs
                {
                    double sf4 = sf2[4];
                    double sf5 = sf2[5];
                    double sf6 = sf2[6];
                    double sf7 = sf2[7];

                    // Calculate butterfly values
                    double b_0 = sf4 + sf6;
                    double b_2 = sf4 - sf6;
                    double b_1 = sf5 * c2;
                    double b_3 = sf7 * c2;

                    // Apply scaling and store with bit-reversal
                    x[lut[4]] = (b_0 + b_1) * mul;
                    x[lut[5]] = (b_0 - b_1) * mul;
                    x[lut[6]] = (b_2 + b_3) * mul;
                    x[lut[7]] = (b_2 - b_3) * mul;
                }
            }
        }
        // Special cases for small IFFTs
        else if (unlikely(_nbr_bits == 2)) {
            // 4-point IFFT
            const double b_0 = f[0] + f[2];
            const double b_2 = f[0] - f[2];
            const double f1_c2 = f[1] * c2;
            const double f3_c2 = f[3] * c2;
            
            x[0] = (b_0 + f1_c2) * mul;
            x[2] = (b_0 - f1_c2) * mul;
            x[1] = (b_2 + f3_c2) * mul;
            x[3] = (b_2 - f3_c2) * mul;
        }
        else if (unlikely(_nbr_bits == 1)) {
            // 2-point IFFT
            x[0] = (f[0] + f[1]) * mul;
            x[1] = (f[0] - f[1]) * mul;
        }
        else {
            // 1-point IFFT
            x[0] = f[0] * mul;
        }
    }
    
    void do_fft_neon_f2(const simd_float2 *_Nonnull x, simd_float2 *_Nonnull f)
    {
        float c, s;
        if (likely(_nbr_bits > 2)) {
            simd_float2 *sf, *df;
            
            // First/second pass with bit-reversal
            constexpr int PREFETCH_DISTANCE = 8;
            auto lut = _bit_rev_lut->get_ptr();
            
            // Initial prefetch
            __builtin_prefetch(x, 0, 3);
            __builtin_prefetch(lut, 0, 3);

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }

            #pragma unroll 2
            for (auto i = 0; i < _N; i += 4) {
                if (likely(i + PREFETCH_DISTANCE < _N)) {
                    __builtin_prefetch(&lut[i + PREFETCH_DISTANCE], 0, 3);
                    __builtin_prefetch(&x[lut[i + PREFETCH_DISTANCE]], 0, 3);
                }

                // For simd_float2, load 2 floats at a time with proper casting
                float32x2_t x0 = vld1_f32((float*)&x[lut[i]]);
                float32x2_t x1 = vld1_f32((float*)&x[lut[i+1]]);
                float32x2_t x2 = vld1_f32((float*)&x[lut[i+2]]);
                float32x2_t x3 = vld1_f32((float*)&x[lut[i+3]]);
                
                // Compute butterfly pattern with 2-element vectors
                float32x2_t sum01 = vadd_f32(x0, x1);
                float32x2_t diff01 = vsub_f32(x0, x1);
                float32x2_t sum23 = vadd_f32(x2, x3);
                float32x2_t diff23 = vsub_f32(x2, x3);
                
                // Store with proper casting
                float* df_ptr = (float*)&df[i];
                vst1_f32(df_ptr, vadd_f32(sum01, sum23));
                vst1_f32(df_ptr + 2, diff01);
                vst1_f32(df_ptr + 4, vsub_f32(sum01, sum23));
                vst1_f32(df_ptr + 6, diff23);
            }

            // Third pass with SQ2_2 optimizations
            const float32x2_t sq2_constants = {-SQ2_2, SQ2_2};
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&df[i + 8], 0, 3);
                }

                // Load with proper casting
                float32x2_t df0 = vld1_f32((float*)&df[i]);
                float32x2_t df4 = vld1_f32((float*)&df[i+4]);
                
                float32x2_t sum04 = vadd_f32(df0, df4);
                float32x2_t diff04 = vsub_f32(df0, df4);
                
                float32x2_t df2val = vld1_f32((float*)&df[i+2]);
                float32x2_t df6val = vld1_f32((float*)&df[i+6]);
                
                // Store early results with proper casting
                vst1_f32((float*)&sf[i], sum04);
                vst1_f32((float*)&sf[i+4], diff04);
                vst1_f32((float*)&sf[i+2], df2val);
                vst1_f32((float*)&sf[i+6], df6val);
                
                // Load more data with proper casting
                float32x2_t df1 = vld1_f32((float*)&df[i+1]);
                float32x2_t df3 = vld1_f32((float*)&df[i+3]);
                float32x2_t df5 = vld1_f32((float*)&df[i+5]);
                float32x2_t df7 = vld1_f32((float*)&df[i+7]);

                // Calculate butterfly with optimized operations
                // For 2-element vectors, use appropriate intrinsics
                float32x2_t v1 = vmul_n_f32(df5, vget_lane_f32(sq2_constants, 0));
                v1 = vmla_n_f32(v1, df7, vget_lane_f32(sq2_constants, 0));
                
                float32x2_t v2 = vmul_n_f32(df5, vget_lane_f32(sq2_constants, 1));
                v2 = vmla_n_f32(v2, df7, vget_lane_f32(sq2_constants, 1));

                // Store final results with proper casting
                vst1_f32((float*)&sf[i+1], vadd_f32(df1, v1));
                vst1_f32((float*)&sf[i+3], vsub_f32(df1, v1));
                vst1_f32((float*)&sf[i+5], vadd_f32(v2, df3));
                vst1_f32((float*)&sf[i+7], vsub_f32(v2, df3));
            }

            // Later passes with twiddle factors
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;

                    // Prefetch for next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients with proper casting
                    float32x2_t sf1r_0 = vld1_f32((float*)&sf1r[0]);
                    float32x2_t sf2r_0 = vld1_f32((float*)&sf2r[0]);
                    
                    vst1_f32((float*)&dfr[0], vadd_f32(sf1r_0, sf2r_0));
                    vst1_f32((float*)&dfi[0], vsub_f32(sf1r_0, sf2r_0));
                    
                    vst1_f32((float*)&dfr[h_nbr_coef], vld1_f32((float*)&sf1r[h_nbr_coef]));
                    vst1_f32((float*)&dfi[h_nbr_coef], vld1_f32((float*)&sf2r[h_nbr_coef]));

                    // Process other conjugate complex numbers
                    auto sf1i = &sf1r[h_nbr_coef];
                    auto sf2i = &sf1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    float c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache->get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        c = c_next;
                        s = s_next;
                        
                        // Prefetch next data and twiddle factors
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sf1r[j + 1], 0, 3);
                            __builtin_prefetch(&sf2r[j + 1], 0, 3);
                            _twiddle_cache->get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load data with proper casting
                        float32x2_t sf1r_j = vld1_f32((float*)&sf1r[j]);
                        float32x2_t sf2r_j = vld1_f32((float*)&sf2r[j]);
                        float32x2_t sf2i_j = vld1_f32((float*)&sf2i[j]);
                        
                        // Calculate sf2r*c - sf2i*s for 2-element vectors
                        float32x2_t sf2r_c = vmul_n_f32(sf2r_j, c);
                        float32x2_t sf2i_s = vmul_n_f32(sf2i_j, s);
                        float32x2_t v = vsub_f32(sf2r_c, sf2i_s);
                        
                        // Store results with proper casting
                        vst1_f32((float*)&dfr[j], vadd_f32(sf1r_j, v));
                        vst1_f32((float*)&dfi[-j], vsub_f32(sf1r_j, v));

                        // Load additional data with proper casting
                        float32x2_t sf1i_j = vld1_f32((float*)&sf1i[j]);
                        
                        // Calculate sf2r*s + sf2i*c for 2-element vectors
                        float32x2_t sf2r_s = vmul_n_f32(sf2r_j, s);
                        float32x2_t sf2i_c = vmul_n_f32(sf2i_j, c);
                        v = vadd_f32(sf2r_s, sf2i_c);
                        
                        vst1_f32((float*)&dfi[j], vadd_f32(v, sf1i_j));
                        vst1_f32((float*)&dfi[nbr_coef - j], vsub_f32(v, sf1i_j));
                    }
                }
                
                // Prepare for next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        // Special cases for small FFTs - these use direct element assignments
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_float2 b_0 = x[0] + x[2];
            const simd_float2 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        else {
            f[0] = x[0];
        }
    }
    
    void do_fft_neon_d2(const simd_double2 *_Nonnull x, simd_double2 *_Nonnull f)
    {
        double c, s;
        if (likely(_nbr_bits > 2)) {
            simd_double2 *sf, *df;

            // Initial prefetch
            __builtin_prefetch(x, 0, 3);
            __builtin_prefetch(_bit_rev_lut->get_ptr(), 0, 3);

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }

            // First and second pass with bit-reversal
            constexpr int PREFETCH_DISTANCE = 8;
            auto lut_ptr = _bit_rev_lut->get_ptr();
            
#pragma unroll 2
            for (auto i = 0; i < _N; i += 4) {
                if (likely(i + PREFETCH_DISTANCE < _N)) {
                    __builtin_prefetch(&lut_ptr[i + PREFETCH_DISTANCE], 0, 3);
                    __builtin_prefetch(&x[lut_ptr[i + PREFETCH_DISTANCE]], 0, 3);
                }
                
                auto df2 = &df[i];
                auto lut = &lut_ptr[i];

                // For simd_double2, load 2 doubles at a time
                float64x2_t x0 = vld1q_f64((double*)&x[lut[0]]);
                float64x2_t x1 = vld1q_f64((double*)&x[lut[1]]);
                float64x2_t x2 = vld1q_f64((double*)&x[lut[2]]);
                float64x2_t x3 = vld1q_f64((double*)&x[lut[3]]);
                
                // Compute butterflies with double precision
                float64x2_t sum01 = vaddq_f64(x0, x1);
                float64x2_t diff01 = vsubq_f64(x0, x1);
                float64x2_t sum23 = vaddq_f64(x2, x3);
                float64x2_t diff23 = vsubq_f64(x2, x3);
                
                // Store results with proper casting
                double* df_ptr = (double*)df2;
                vst1q_f64(df_ptr,     vaddq_f64(sum01, sum23));
                vst1q_f64(df_ptr + 2, diff01);
                vst1q_f64(df_ptr + 4, vsubq_f64(sum01, sum23));
                vst1q_f64(df_ptr + 6, diff23);
            }

            // Third pass with SQ2_2 optimization
            const float64x1_t neg_sq2_2 = vdup_n_f64(-SQ2_2);
            const float64x1_t pos_sq2_2 = vdup_n_f64(SQ2_2);
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&df[i + 8], 0, 3);
                }

                // Load data with proper casting
                double* df_ptr = (double*)&df[i];
                float64x2_t df0 = vld1q_f64(df_ptr);
                float64x2_t df4 = vld1q_f64(df_ptr + 8);
                
                // Compute sums and differences
                float64x2_t sum04 = vaddq_f64(df0, df4);
                float64x2_t diff04 = vsubq_f64(df0, df4);
                
                // Load more data
                float64x2_t df2val = vld1q_f64(df_ptr + 4);
                float64x2_t df6val = vld1q_f64(df_ptr + 12);
                
                // Store interim results
                double* sf_ptr = (double*)&sf[i];
                vst1q_f64(sf_ptr,      sum04);
                vst1q_f64(sf_ptr + 8,  diff04);
                vst1q_f64(sf_ptr + 4,  df2val);
                vst1q_f64(sf_ptr + 12, df6val);
                
                // Load additional data
                float64x2_t df1 = vld1q_f64(df_ptr + 2);
                float64x2_t df3 = vld1q_f64(df_ptr + 6);
                float64x2_t df5 = vld1q_f64(df_ptr + 10);
                float64x2_t df7 = vld1q_f64(df_ptr + 14);

                // Calculate v1 with scalar multiplication
                float64x2_t neg_sq2_2_vec = vdupq_n_f64(vget_lane_f64(neg_sq2_2, 0));
                float64x2_t v1 = vmulq_f64(df5, neg_sq2_2_vec);
                float64x2_t df7_scaled = vmulq_f64(df7, neg_sq2_2_vec);
                v1 = vaddq_f64(v1, df7_scaled);

                // Calculate v2 with scalar multiplication
                float64x2_t pos_sq2_2_vec = vdupq_n_f64(vget_lane_f64(pos_sq2_2, 0));
                float64x2_t v2 = vmulq_f64(df5, pos_sq2_2_vec);
                float64x2_t df7_scaled2 = vmulq_f64(df7, pos_sq2_2_vec);
                v2 = vaddq_f64(v2, df7_scaled2);

                // Store final results
                vst1q_f64(sf_ptr + 2,  vaddq_f64(df1, v1));
                vst1q_f64(sf_ptr + 6,  vsubq_f64(df1, v1));
                vst1q_f64(sf_ptr + 10, vaddq_f64(v2, df3));
                vst1q_f64(sf_ptr + 14, vsubq_f64(v2, df3));
            }

            // Later passes with twiddle factors
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;

                    // Prefetch for next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients with proper casting
                    double* sf1r_ptr = (double*)sf1r;
                    double* sf2r_ptr = (double*)sf2r;
                    double* dfr_ptr = (double*)dfr;
                    double* dfi_ptr = (double*)dfi;
                    
                    float64x2_t sf1r_0 = vld1q_f64(sf1r_ptr);
                    float64x2_t sf2r_0 = vld1q_f64(sf2r_ptr);
                    
                    vst1q_f64(dfr_ptr, vaddq_f64(sf1r_0, sf2r_0));
                    vst1q_f64(dfi_ptr, vsubq_f64(sf1r_0, sf2r_0));
                    
                    // Handle h_nbr_coef coefficients
                    double* sf1r_h_ptr = (double*)&sf1r[h_nbr_coef];
                    double* sf2r_h_ptr = (double*)&sf2r[h_nbr_coef];
                    double* dfr_h_ptr = (double*)&dfr[h_nbr_coef];
                    double* dfi_h_ptr = (double*)&dfi[h_nbr_coef];
                    
                    vst1q_f64(dfr_h_ptr, vld1q_f64(sf1r_h_ptr));
                    vst1q_f64(dfi_h_ptr, vld1q_f64(sf2r_h_ptr));

                    // Process conjugate complex numbers
                    auto sf1i = &sf1r[h_nbr_coef];
                    auto sf2i = &sf1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    double c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache->get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        c = c_next;
                        s = s_next;
                        
                        // Prefetch next iteration data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sf1r[j + 1], 0, 3);
                            __builtin_prefetch(&sf2r[j + 1], 0, 3);
                            _twiddle_cache->get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load data with proper casting
                        double* sf1r_j_ptr = (double*)&sf1r[j];
                        double* sf2r_j_ptr = (double*)&sf2r[j];
                        double* sf2i_j_ptr = (double*)&sf2i[j];
                        double* sf1i_j_ptr = (double*)&sf1i[j];
                        double* dfr_j_ptr = (double*)&dfr[j];
                        double* dfi_neg_j_ptr = (double*)&dfi[-j];
                        double* dfi_j_ptr = (double*)&dfi[j];
                        double* dfi_nbr_j_ptr = (double*)&dfi[nbr_coef - j];
                        
                        float64x2_t sf1r_j = vld1q_f64(sf1r_j_ptr);
                        float64x2_t sf2r_j = vld1q_f64(sf2r_j_ptr);
                        float64x2_t sf2i_j = vld1q_f64(sf2i_j_ptr);

                        // Calculate sf2r*c - sf2i*s with scalar multiplication
                        float64x2_t c_vec = vdupq_n_f64(c);
                        float64x2_t s_vec = vdupq_n_f64(s);
                        
                        float64x2_t sf2r_c = vmulq_f64(sf2r_j, c_vec);
                        float64x2_t sf2i_s = vmulq_f64(sf2i_j, s_vec);
                        float64x2_t v = vsubq_f64(sf2r_c, sf2i_s);
                        
                        // Store results
                        vst1q_f64(dfr_j_ptr, vaddq_f64(sf1r_j, v));
                        vst1q_f64(dfi_neg_j_ptr, vsubq_f64(sf1r_j, v));

                        // Calculate sf2r*s + sf2i*c
                        float64x2_t sf1i_j = vld1q_f64(sf1i_j_ptr);
                        
                        float64x2_t sf2r_s = vmulq_f64(sf2r_j, s_vec);
                        float64x2_t sf2i_c = vmulq_f64(sf2i_j, c_vec);
                        v = vaddq_f64(sf2r_s, sf2i_c);
                        
                        vst1q_f64(dfi_j_ptr, vaddq_f64(v, sf1i_j));
                        vst1q_f64(dfi_nbr_j_ptr, vsubq_f64(v, sf1i_j));
                    }
                }
                
                // Prepare for next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        // Special cases for small FFTs - direct element assignment
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_double2 b_0 = x[0] + x[2];
            const simd_double2 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        else {
            f[0] = x[0];
        }
    }
    
    void do_fft_neon_f4(const simd_float4 *_Nonnull x, simd_float4 *_Nonnull f)
    {
        float c, s;
        if (likely(_nbr_bits > 2)) {
            simd_float4 *sf, *df;

            // Initial prefetch
            __builtin_prefetch(x, 0, 3);
            __builtin_prefetch(_bit_rev_lut.get_ptr(), 0, 3);

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }

            // First/second pass with bit-reversal
            constexpr int PREFETCH_DISTANCE = 8;
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
#pragma unroll 2
            for (auto i = 0; i < _N; i += 4) {
                if (likely(i + PREFETCH_DISTANCE < _N)) {
                    __builtin_prefetch(&lut_ptr[i + PREFETCH_DISTANCE], 0, 3);
                    __builtin_prefetch(&x[lut_ptr[i + PREFETCH_DISTANCE]], 0, 3);
                }

                // Load input data in bit-reversed order with correct casting
                float32x4_t x0 = vld1q_f32((float*)&x[lut_ptr[i]]);
                float32x4_t x1 = vld1q_f32((float*)&x[lut_ptr[i+1]]);
                float32x4_t x2 = vld1q_f32((float*)&x[lut_ptr[i+2]]);
                float32x4_t x3 = vld1q_f32((float*)&x[lut_ptr[i+3]]);
                
                // Compute butterfly pattern
                float32x4_t sum01 = vaddq_f32(x0, x1);
                float32x4_t diff01 = vsubq_f32(x0, x1);
                float32x4_t sum23 = vaddq_f32(x2, x3);
                float32x4_t diff23 = vsubq_f32(x2, x3);
                
                // Store with proper casting
                float* df_ptr = (float*)&df[i];
                vst1q_f32(df_ptr, vaddq_f32(sum01, sum23));
                vst1q_f32(df_ptr + 4, diff01);
                vst1q_f32(df_ptr + 8, vsubq_f32(sum01, sum23));
                vst1q_f32(df_ptr + 12, diff23);
            }

            // Third pass with SQ2_2 optimizations
            static const float32x2_t sq2_constants = {-SQ2_2, SQ2_2};
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&df[i + 8], 0, 3);
                }

                // Load with proper casting
                float32x4_t df0 = vld1q_f32((float*)&df[i]);
                float32x4_t df4 = vld1q_f32((float*)&df[i+4]);
                
                float32x4_t sum04 = vaddq_f32(df0, df4);
                float32x4_t diff04 = vsubq_f32(df0, df4);
                
                float32x4_t df2val = vld1q_f32((float*)&df[i+2]);
                float32x4_t df6val = vld1q_f32((float*)&df[i+6]);
                
                // Store early results with proper casting
                vst1q_f32((float*)&sf[i], sum04);
                vst1q_f32((float*)&sf[i+4], diff04);
                vst1q_f32((float*)&sf[i+2], df2val);
                vst1q_f32((float*)&sf[i+6], df6val);
                
                // Load more data with proper casting
                float32x4_t df1 = vld1q_f32((float*)&df[i+1]);
                float32x4_t df3 = vld1q_f32((float*)&df[i+3]);
                float32x4_t df5 = vld1q_f32((float*)&df[i+5]);
                float32x4_t df7 = vld1q_f32((float*)&df[i+7]);

                // Calculate butterfly with FMA operations
                float32x4_t v1 = vfmaq_lane_f32(
                    vfmaq_lane_f32(vdupq_n_f32(0.0f), df5, sq2_constants, 0),
                    df7, sq2_constants, 0
                );
                
                float32x4_t v2 = vfmaq_lane_f32(
                    vfmaq_lane_f32(vdupq_n_f32(0.0f), df5, sq2_constants, 1),
                    df7, sq2_constants, 1
                );

                // Store final results with proper casting
                vst1q_f32((float*)&sf[i+1], vaddq_f32(df1, v1));
                vst1q_f32((float*)&sf[i+3], vsubq_f32(df1, v1));
                vst1q_f32((float*)&sf[i+5], vaddq_f32(v2, df3));
                vst1q_f32((float*)&sf[i+7], vsubq_f32(v2, df3));
            }

            // Later passes with twiddle factors
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;

                    // Prefetch for next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients with proper casting
                    float32x4_t sf1r_0 = vld1q_f32((float*)&sf1r[0]);
                    float32x4_t sf2r_0 = vld1q_f32((float*)&sf2r[0]);
                    
                    vst1q_f32((float*)&dfr[0], vaddq_f32(sf1r_0, sf2r_0));
                    vst1q_f32((float*)&dfi[0], vsubq_f32(sf1r_0, sf2r_0));
                    
                    vst1q_f32((float*)&dfr[h_nbr_coef], vld1q_f32((float*)&sf1r[h_nbr_coef]));
                    vst1q_f32((float*)&dfi[h_nbr_coef], vld1q_f32((float*)&sf2r[h_nbr_coef]));

                    // Process other conjugate complex numbers
                    auto sf1i = &sf1r[h_nbr_coef];
                    auto sf2i = &sf1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    float c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        c = c_next;
                        s = s_next;
                        
                        // Prefetch next data and twiddle factors
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sf1r[j + 1], 0, 3);
                            __builtin_prefetch(&sf2r[j + 1], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Create vectorized twiddle factors
                        float32x4_t c_vec = vdupq_n_f32(c);
                        float32x4_t s_vec = vdupq_n_f32(s);

                        // Load data with proper casting
                        float32x4_t sf1r_j = vld1q_f32((float*)&sf1r[j]);
                        float32x4_t sf2r_j = vld1q_f32((float*)&sf2r[j]);
                        float32x4_t sf2i_j = vld1q_f32((float*)&sf2i[j]);
                        
                        // Calculate sf2r*c - sf2i*s
                        float32x4_t sf2r_c = vmulq_f32(sf2r_j, c_vec);
                        float32x4_t sf2i_s = vmulq_f32(sf2i_j, s_vec);
                        float32x4_t v = vsubq_f32(sf2r_c, sf2i_s);
                        
                        // Store results with proper casting
                        vst1q_f32((float*)&dfr[j], vaddq_f32(sf1r_j, v));
                        vst1q_f32((float*)&dfi[-j], vsubq_f32(sf1r_j, v));

                        // Load additional data with proper casting
                        float32x4_t sf1i_j = vld1q_f32((float*)&sf1i[j]);
                        
                        // Calculate sf2r*s + sf2i*c
                        v = vfmaq_f32(vmulq_f32(sf2r_j, s_vec), sf2i_j, c_vec);
                        
                        vst1q_f32((float*)&dfi[j], vaddq_f32(v, sf1i_j));
                        vst1q_f32((float*)&dfi[nbr_coef - j], vsubq_f32(v, sf1i_j));
                    }
                }
                
                // Prepare for next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        // Special cases for small FFTs - these use direct element assignments
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_float4 b_0 = x[0] + x[2];
            const simd_float4 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        else {
            f[0] = x[0];
        }
    }
    
    void do_fft_neon_d4(const simd_double4 *_Nonnull x, simd_double4 *_Nonnull f)
    {
        double c, s;
        if (likely(_nbr_bits > 2)) {
            simd_double4 *sf, *df;

            // Initial prefetch
            __builtin_prefetch(x, 0, 3);
            __builtin_prefetch(_bit_rev_lut.get_ptr(), 0, 3);

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }

            // First and second pass with bit-reversal
            constexpr int PREFETCH_DISTANCE = 8;
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
#pragma unroll 2
            for (auto i = 0; i < _N; i += 4) {
                if (likely(i + PREFETCH_DISTANCE < _N)) {
                    __builtin_prefetch(&lut_ptr[i + PREFETCH_DISTANCE], 0, 3);
                    __builtin_prefetch(&x[lut_ptr[i + PREFETCH_DISTANCE]], 0, 3);
                }
                
                auto df2 = &df[i];
                auto lut = &lut_ptr[i];

                // Proper casting for simd_double4 vector
                double* x0_ptr = (double*)&x[lut[0]];
                double* x1_ptr = (double*)&x[lut[1]];
                double* x2_ptr = (double*)&x[lut[2]];
                double* x3_ptr = (double*)&x[lut[3]];
                
                // For simd_double4, load two doubles at a time
                float64x2_t x0_low = vld1q_f64(x0_ptr);
                float64x2_t x0_high = vld1q_f64(x0_ptr + 2);
                float64x2_t x1_low = vld1q_f64(x1_ptr);
                float64x2_t x1_high = vld1q_f64(x1_ptr + 2);
                float64x2_t x2_low = vld1q_f64(x2_ptr);
                float64x2_t x2_high = vld1q_f64(x2_ptr + 2);
                float64x2_t x3_low = vld1q_f64(x3_ptr);
                float64x2_t x3_high = vld1q_f64(x3_ptr + 2);
                
                // Compute butterflies
                float64x2_t sum01_low = vaddq_f64(x0_low, x1_low);
                float64x2_t sum01_high = vaddq_f64(x0_high, x1_high);
                float64x2_t diff01_low = vsubq_f64(x0_low, x1_low);
                float64x2_t diff01_high = vsubq_f64(x0_high, x1_high);
                float64x2_t sum23_low = vaddq_f64(x2_low, x3_low);
                float64x2_t sum23_high = vaddq_f64(x2_high, x3_high);
                float64x2_t diff23_low = vsubq_f64(x2_low, x3_low);
                float64x2_t diff23_high = vsubq_f64(x2_high, x3_high);
                
                // Store results with proper casting
                double* df_ptr = (double*)df2;
                vst1q_f64(df_ptr,      vaddq_f64(sum01_low, sum23_low));
                vst1q_f64(df_ptr + 2,  vaddq_f64(sum01_high, sum23_high));
                vst1q_f64(df_ptr + 4,  diff01_low);
                vst1q_f64(df_ptr + 6,  diff01_high);
                vst1q_f64(df_ptr + 8,  vsubq_f64(sum01_low, sum23_low));
                vst1q_f64(df_ptr + 10, vsubq_f64(sum01_high, sum23_high));
                vst1q_f64(df_ptr + 12, diff23_low);
                vst1q_f64(df_ptr + 14, diff23_high);
            }

            // Third pass with optimized SQ2_2 calculations
            const float64x1_t neg_sq2_2 = vdup_n_f64(-SQ2_2);
            const float64x1_t pos_sq2_2 = vdup_n_f64(SQ2_2);
            
            for (auto i = 0; i < _N; i += 8) {
                auto sf2 = &sf[i];
                auto df2 = &df[i];
                
                // Prefetch
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&df[i + 8], 0, 3);
                }

                // Load with proper casting
                double* df_ptr = (double*)df2;
                float64x2_t df0_low = vld1q_f64(df_ptr);
                float64x2_t df0_high = vld1q_f64(df_ptr + 2);
                float64x2_t df4_low = vld1q_f64(df_ptr + 16);
                float64x2_t df4_high = vld1q_f64(df_ptr + 18);
                
                // Compute sums and differences
                float64x2_t sum04_low = vaddq_f64(df0_low, df4_low);
                float64x2_t sum04_high = vaddq_f64(df0_high, df4_high);
                float64x2_t diff04_low = vsubq_f64(df0_low, df4_low);
                float64x2_t diff04_high = vsubq_f64(df0_high, df4_high);

                // Load more data
                float64x2_t df2val_low = vld1q_f64(df_ptr + 8);
                float64x2_t df2val_high = vld1q_f64(df_ptr + 10);
                float64x2_t df6val_low = vld1q_f64(df_ptr + 24);
                float64x2_t df6val_high = vld1q_f64(df_ptr + 26);
                
                // Store with proper casting
                double* sf_ptr = (double*)sf2;
                vst1q_f64(sf_ptr,      sum04_low);
                vst1q_f64(sf_ptr + 2,  sum04_high);
                vst1q_f64(sf_ptr + 16, diff04_low);
                vst1q_f64(sf_ptr + 18, diff04_high);
                vst1q_f64(sf_ptr + 8,  df2val_low);
                vst1q_f64(sf_ptr + 10, df2val_high);
                vst1q_f64(sf_ptr + 24, df6val_low);
                vst1q_f64(sf_ptr + 26, df6val_high);

                // Load additional data
                float64x2_t df1_low = vld1q_f64(df_ptr + 4);
                float64x2_t df1_high = vld1q_f64(df_ptr + 6);
                float64x2_t df3_low = vld1q_f64(df_ptr + 12);
                float64x2_t df3_high = vld1q_f64(df_ptr + 14);
                float64x2_t df5_low = vld1q_f64(df_ptr + 20);
                float64x2_t df5_high = vld1q_f64(df_ptr + 22);
                float64x2_t df7_low = vld1q_f64(df_ptr + 28);
                float64x2_t df7_high = vld1q_f64(df_ptr + 30);

                // Calculate v1 with proper operations for doubles
                float64x2_t v1_low = vmulq_lane_f64(df5_low, neg_sq2_2, 0);
                float64x2_t v1_high = vmulq_lane_f64(df5_high, neg_sq2_2, 0);
               
                // Create a vector with the scalar value in both lanes
                float64x2_t neg_sq2_2_vec = vdupq_n_f64(vget_lane_f64(neg_sq2_2, 0));

                // Multiply and add separately
                float64x2_t df7_low_scaled = vmulq_f64(df7_low, neg_sq2_2_vec);
                float64x2_t df7_high_scaled = vmulq_f64(df7_high, neg_sq2_2_vec);
                v1_low = vaddq_f64(v1_low, df7_low_scaled);
                v1_high = vaddq_f64(v1_high, df7_high_scaled);
                
                // Calculate v2 - similar pattern as v1
                float64x2_t v2_low = vmulq_lane_f64(df5_low, pos_sq2_2, 0);
                float64x2_t v2_high = vmulq_lane_f64(df5_high, pos_sq2_2, 0);
                float64x2_t pos_sq2_2_vec = vdupq_n_f64(vget_lane_f64(pos_sq2_2, 0));
                float64x2_t df7_low_scaled2 = vmulq_f64(df7_low, pos_sq2_2_vec);
                float64x2_t df7_high_scaled2 = vmulq_f64(df7_high, pos_sq2_2_vec);
                v2_low = vaddq_f64(v2_low, df7_low_scaled2);
                v2_high = vaddq_f64(v2_high, df7_high_scaled2);
                
                // Store final results
                vst1q_f64(sf_ptr + 4,  vaddq_f64(df1_low, v1_low));
                vst1q_f64(sf_ptr + 6,  vaddq_f64(df1_high, v1_high));
                vst1q_f64(sf_ptr + 12, vsubq_f64(df1_low, v1_low));
                vst1q_f64(sf_ptr + 14, vsubq_f64(df1_high, v1_high));
                
                vst1q_f64(sf_ptr + 20, vaddq_f64(v2_low, df3_low));
                vst1q_f64(sf_ptr + 22, vaddq_f64(v2_high, df3_high));
                vst1q_f64(sf_ptr + 28, vsubq_f64(v2_low, df3_low));
                vst1q_f64(sf_ptr + 30, vsubq_f64(v2_high, df3_high));
            }

            // Next passes
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;

                    // Prefetch
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients
                    double* sf1r_ptr = (double*)sf1r;
                    double* sf2r_ptr = (double*)sf2r;
                    double* dfr_ptr = (double*)dfr;
                    double* dfi_ptr = (double*)dfi;
                    
                    float64x2_t sf1r_0_low = vld1q_f64(sf1r_ptr);
                    float64x2_t sf1r_0_high = vld1q_f64(sf1r_ptr + 2);
                    float64x2_t sf2r_0_low = vld1q_f64(sf2r_ptr);
                    float64x2_t sf2r_0_high = vld1q_f64(sf2r_ptr + 2);
                    
                    vst1q_f64(dfr_ptr, vaddq_f64(sf1r_0_low, sf2r_0_low));
                    vst1q_f64(dfr_ptr + 2, vaddq_f64(sf1r_0_high, sf2r_0_high));
                    vst1q_f64(dfi_ptr, vsubq_f64(sf1r_0_low, sf2r_0_low));
                    vst1q_f64(dfi_ptr + 2, vsubq_f64(sf1r_0_high, sf2r_0_high));
                    
                    // Calculate pointer to h_nbr_coef
                    double* sf1r_h_ptr = (double*)&sf1r[h_nbr_coef];
                    double* sf2r_h_ptr = (double*)&sf2r[h_nbr_coef];
                    double* dfr_h_ptr = (double*)&dfr[h_nbr_coef];
                    double* dfi_h_ptr = (double*)&dfi[h_nbr_coef];
                    
                    vst1q_f64(dfr_h_ptr, vld1q_f64(sf1r_h_ptr));
                    vst1q_f64(dfr_h_ptr + 2, vld1q_f64(sf1r_h_ptr + 2));
                    vst1q_f64(dfi_h_ptr, vld1q_f64(sf2r_h_ptr));
                    vst1q_f64(dfi_h_ptr + 2, vld1q_f64(sf2r_h_ptr + 2));

                    // Process conjugate complex numbers
                    auto sf1i = &sf1r[h_nbr_coef];
                    auto sf2i = &sf1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    double c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        c = c_next;
                        s = s_next;
                        
                        // Prefetch next iteration data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sf1r[j + 1], 0, 3);
                            __builtin_prefetch(&sf2r[j + 1], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Create scalar factors
                        float64x1_t cos_val = vdup_n_f64(c);
                        float64x1_t sin_val = vdup_n_f64(s);

                        // Load data with proper casting
                        double* sf1r_j_ptr = (double*)&sf1r[j];
                        double* sf2r_j_ptr = (double*)&sf2r[j];
                        double* sf2i_j_ptr = (double*)&sf2i[j];
                        double* sf1i_j_ptr = (double*)&sf1i[j];
                        double* dfr_j_ptr = (double*)&dfr[j];
                        double* dfi_neg_j_ptr = (double*)&dfi[-j];
                        double* dfi_j_ptr = (double*)&dfi[j];
                        double* dfi_nbr_j_ptr = (double*)&dfi[nbr_coef - j];
                        
                        float64x2_t sf1r_j_low = vld1q_f64(sf1r_j_ptr);
                        float64x2_t sf1r_j_high = vld1q_f64(sf1r_j_ptr + 2);
                        float64x2_t sf2r_j_low = vld1q_f64(sf2r_j_ptr);
                        float64x2_t sf2r_j_high = vld1q_f64(sf2r_j_ptr + 2);
                        float64x2_t sf2i_j_low = vld1q_f64(sf2i_j_ptr);
                        float64x2_t sf2i_j_high = vld1q_f64(sf2i_j_ptr + 2);

                        // Calculate sf2r*c - sf2i*s (in separate operations for doubles)
                        float64x2_t sf2r_c_low = vmulq_lane_f64(sf2r_j_low, cos_val, 0);
                        float64x2_t sf2r_c_high = vmulq_lane_f64(sf2r_j_high, cos_val, 0);
                        float64x2_t sf2i_s_low = vmulq_lane_f64(sf2i_j_low, sin_val, 0);
                        float64x2_t sf2i_s_high = vmulq_lane_f64(sf2i_j_high, sin_val, 0);
                        
                        float64x2_t v_low = vsubq_f64(sf2r_c_low, sf2i_s_low);
                        float64x2_t v_high = vsubq_f64(sf2r_c_high, sf2i_s_high);
                        
                        // Store results
                        vst1q_f64(dfr_j_ptr, vaddq_f64(sf1r_j_low, v_low));
                        vst1q_f64(dfr_j_ptr + 2, vaddq_f64(sf1r_j_high, v_high));
                        vst1q_f64(dfi_neg_j_ptr, vsubq_f64(sf1r_j_low, v_low));
                        vst1q_f64(dfi_neg_j_ptr + 2, vsubq_f64(sf1r_j_high, v_high));

                        // Calculate sf2r*s + sf2i*c
                        float64x2_t sf1i_j_low = vld1q_f64(sf1i_j_ptr);
                        float64x2_t sf1i_j_high = vld1q_f64(sf1i_j_ptr + 2);
                        
                        float64x2_t sf2r_s_low = vmulq_lane_f64(sf2r_j_low, sin_val, 0);
                        float64x2_t sf2r_s_high = vmulq_lane_f64(sf2r_j_high, sin_val, 0);
                        float64x2_t sf2i_c_low = vmulq_lane_f64(sf2i_j_low, cos_val, 0);
                        float64x2_t sf2i_c_high = vmulq_lane_f64(sf2i_j_high, cos_val, 0);
                        
                        v_low = vaddq_f64(sf2r_s_low, sf2i_c_low);
                        v_high = vaddq_f64(sf2r_s_high, sf2i_c_high);
                        
                        vst1q_f64(dfi_j_ptr, vaddq_f64(v_low, sf1i_j_low));
                        vst1q_f64(dfi_j_ptr + 2, vaddq_f64(v_high, sf1i_j_high));
                        vst1q_f64(dfi_nbr_j_ptr, vsubq_f64(v_low, sf1i_j_low));
                        vst1q_f64(dfi_nbr_j_ptr + 2, vsubq_f64(v_high, sf1i_j_high));
                    }
                }
                
                // Prepare for next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        // Special case handling same as original
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_double4 b_0 = x[0] + x[2];
            const simd_double4 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        else {
            f[0] = x[0];
        }
    }
    
    void do_fft_neon_f8(const simd_float8 *_Nonnull x, simd_float8 *_Nonnull f)
    {
        float c, s;
        if (likely(_nbr_bits > 2)) {
            simd_float8 *sf, *df;
            
            // OPTIMIZATION: Prefetch critical memory blocks at start
            __builtin_prefetch(&x[0], 0, 3);
            __builtin_prefetch(&_bit_rev_lut.get_ptr()[0], 0, 3);

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }

            // First and second pass - BIT REVERSAL OPTIMIZATION
            constexpr int PREFETCH_DISTANCE = 8;
            auto lut = _bit_rev_lut.get_ptr();
            
            #pragma unroll 2
            for (auto i = 0; i < _N; i += 4) {
                if (likely(i + PREFETCH_DISTANCE < _N)) {
                    __builtin_prefetch(&lut[i + PREFETCH_DISTANCE], 0, 3);
                    __builtin_prefetch(&x[lut[i + PREFETCH_DISTANCE]], 0, 3);
                }
                
                auto df2 = &df[i];
                auto lut0 = lut[i];
                auto lut1 = lut[i+1];
                auto lut2 = lut[i+2];
                auto lut3 = lut[i+3];
 
                float32x4_t x0_low = vld1q_f32(&x[lut0].f[0]);
                float32x4_t x0_high = vld1q_f32(&x[lut0].f[4]);
                float32x4_t x1_low = vld1q_f32(&x[lut1].f[0]);
                float32x4_t x1_high = vld1q_f32(&x[lut1].f[4]);
                float32x4_t x2_low = vld1q_f32(&x[lut2].f[0]);
                float32x4_t x2_high = vld1q_f32(&x[lut2].f[4]);
                float32x4_t x3_low = vld1q_f32(&x[lut3].f[0]);
                float32x4_t x3_high = vld1q_f32(&x[lut3].f[4]);
                
                // Compute all sums and differences in parallel
                float32x4_t sum01_low = vaddq_f32(x0_low, x1_low);
                float32x4_t diff01_low = vsubq_f32(x0_low, x1_low);
                float32x4_t sum23_low = vaddq_f32(x2_low, x3_low);
                float32x4_t diff23_low = vsubq_f32(x2_low, x3_low);
                
                float32x4_t sum01_high = vaddq_f32(x0_high, x1_high);
                float32x4_t diff01_high = vsubq_f32(x0_high, x1_high);
                float32x4_t sum23_high = vaddq_f32(x2_high, x3_high);
                float32x4_t diff23_high = vsubq_f32(x2_high, x3_high);
                
                float32x4_t sum0123_low = vaddq_f32(sum01_low, sum23_low);
                float32x4_t sum0123_high = vaddq_f32(sum01_high, sum23_high);
                float32x4_t diff0123_low = vsubq_f32(sum01_low, sum23_low);
                float32x4_t diff0123_high = vsubq_f32(sum01_high, sum23_high);

                // Store results with cache line optimization
                vst1q_f32(&df2[0].f[0], sum0123_low);
                vst1q_f32(&df2[0].f[4], sum0123_high);
                vst1q_f32(&df2[1].f[0], diff01_low);
                vst1q_f32(&df2[1].f[4], diff01_high);
                vst1q_f32(&df2[2].f[0], diff0123_low);
                vst1q_f32(&df2[2].f[4], diff0123_high);
                vst1q_f32(&df2[3].f[0], diff23_low);
                vst1q_f32(&df2[3].f[4], diff23_high);
            }

            // Third pass with software pipelining
            static const float32x2_t sq2_constants = {-SQ2_2, SQ2_2};
            
            // Preload first block
            auto df_first = &df[0];
            float32x4_t df0_low_next = vld1q_f32(&df_first[0].f[0]);
            float32x4_t df0_high_next = vld1q_f32(&df_first[0].f[4]);
            float32x4_t df4_low_next = vld1q_f32(&df_first[4].f[0]);
            float32x4_t df4_high_next = vld1q_f32(&df_first[4].f[4]);
            
            for (auto i = 0; i < _N; i += 8) {
                auto sf2 = &sf[i];
                auto df2 = &df[i];
                
                // Software pipelining - use preloaded data
                float32x4_t df0_low = df0_low_next;
                float32x4_t df0_high = df0_high_next;
                float32x4_t df4_low = df4_low_next;
                float32x4_t df4_high = df4_high_next;
                
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&df[i+8], 0, 3);
                    __builtin_prefetch(&df[i+12], 0, 3);
                    
                    if (likely(i + 16 < _N)) {
                        df0_low_next = vld1q_f32(&df[i+8].f[0]);
                        df0_high_next = vld1q_f32(&df[i+8].f[4]);
                        df4_low_next = vld1q_f32(&df[i+12].f[0]);
                        df4_high_next = vld1q_f32(&df[i+12].f[4]);
                    }
                }
                
                // Compute sums and store immediately to keep registers free
                float32x4_t sum04_low = vaddq_f32(df0_low, df4_low);
                float32x4_t sum04_high = vaddq_f32(df0_high, df4_high);
                float32x4_t diff04_low = vsubq_f32(df0_low, df4_low);
                float32x4_t diff04_high = vsubq_f32(df0_high, df4_high);
                
                // Store first part while loading more data
                vst1q_f32(&sf2[0].f[0], sum04_low);
                float32x4_t df2val_low = vld1q_f32(&df2[2].f[0]);
                vst1q_f32(&sf2[0].f[4], sum04_high);
                float32x4_t df2val_high = vld1q_f32(&df2[2].f[4]);
                
                vst1q_f32(&sf2[4].f[0], diff04_low);
                float32x4_t df6val_low = vld1q_f32(&df2[6].f[0]);
                vst1q_f32(&sf2[4].f[4], diff04_high);
                float32x4_t df6val_high = vld1q_f32(&df2[6].f[4]);
                
                // Store simple copies while loading more data
                vst1q_f32(&sf2[2].f[0], df2val_low);
                float32x4_t df1_low = vld1q_f32(&df2[1].f[0]);
                vst1q_f32(&sf2[2].f[4], df2val_high);
                float32x4_t df1_high = vld1q_f32(&df2[1].f[4]);
                
                vst1q_f32(&sf2[6].f[0], df6val_low);
                float32x4_t df3_low = vld1q_f32(&df2[3].f[0]);
                vst1q_f32(&sf2[6].f[4], df6val_high);
                float32x4_t df3_high = vld1q_f32(&df2[3].f[4]);
                
                // Load remaining data
                float32x4_t df5_low = vld1q_f32(&df2[5].f[0]);
                float32x4_t df5_high = vld1q_f32(&df2[5].f[4]);
                float32x4_t df7_low = vld1q_f32(&df2[7].f[0]);
                float32x4_t df7_high = vld1q_f32(&df2[7].f[4]);

                // Optimized butterfly with lane-specific FMA
                float32x4_t v1_low = vfmaq_lane_f32(vfmaq_lane_f32(vdupq_n_f32(0), df5_low, sq2_constants, 0),
                                                  df7_low, sq2_constants, 0);
                float32x4_t v1_high = vfmaq_lane_f32(vfmaq_lane_f32(vdupq_n_f32(0), df5_high, sq2_constants, 0),
                                                   df7_high, sq2_constants, 0);

                float32x4_t v2_low = vfmaq_lane_f32(vfmaq_lane_f32(vdupq_n_f32(0), df5_low, sq2_constants, 1),
                                                  df7_low, sq2_constants, 1);
                float32x4_t v2_high = vfmaq_lane_f32(vfmaq_lane_f32(vdupq_n_f32(0), df5_high, sq2_constants, 1),
                                                   df7_high, sq2_constants, 1);

                // Final stores
                vst1q_f32(&sf2[1].f[0], vaddq_f32(df1_low, v1_low));
                vst1q_f32(&sf2[1].f[4], vaddq_f32(df1_high, v1_high));
                vst1q_f32(&sf2[3].f[0], vsubq_f32(df1_low, v1_low));
                vst1q_f32(&sf2[3].f[4], vsubq_f32(df1_high, v1_high));
                
                vst1q_f32(&sf2[5].f[0], vaddq_f32(v2_low, df3_low));
                vst1q_f32(&sf2[5].f[4], vaddq_f32(v2_high, df3_high));
                vst1q_f32(&sf2[7].f[0], vsubq_f32(v2_low, df3_low));
                vst1q_f32(&sf2[7].f[4], vsubq_f32(v2_high, df3_high));
            }

            // Next passes optimization with improved twiddle factor handling
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

#pragma loop_count min(1), max(10), avg(6)
                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;

                    // Prefetch outer loop data
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                        __builtin_prefetch(&sf[i + d_nbr_coef + nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients first (always real)
                    float32x4_t sf1r_0_low = vld1q_f32(&sf1r[0].f[0]);
                    float32x4_t sf1r_0_high = vld1q_f32(&sf1r[0].f[4]);
                    float32x4_t sf2r_0_low = vld1q_f32(&sf2r[0].f[0]);
                    float32x4_t sf2r_0_high = vld1q_f32(&sf2r[0].f[4]);
                    
                    vst1q_f32(&dfr[0].f[0], vaddq_f32(sf1r_0_low, sf2r_0_low));
                    vst1q_f32(&dfr[0].f[4], vaddq_f32(sf1r_0_high, sf2r_0_high));
                    vst1q_f32(&dfi[0].f[0], vsubq_f32(sf1r_0_low, sf2r_0_low));
                    vst1q_f32(&dfi[0].f[4], vsubq_f32(sf1r_0_high, sf2r_0_high));
                    
                    float32x4_t sf1r_h_low = vld1q_f32(&sf1r[h_nbr_coef].f[0]);
                    float32x4_t sf1r_h_high = vld1q_f32(&sf1r[h_nbr_coef].f[4]);
                    float32x4_t sf2r_h_low = vld1q_f32(&sf2r[h_nbr_coef].f[0]);
                    float32x4_t sf2r_h_high = vld1q_f32(&sf2r[h_nbr_coef].f[4]);
                    
                    vst1q_f32(&dfr[h_nbr_coef].f[0], sf1r_h_low);
                    vst1q_f32(&dfr[h_nbr_coef].f[4], sf1r_h_high);
                    vst1q_f32(&dfi[h_nbr_coef].f[0], sf2r_h_low);
                    vst1q_f32(&dfi[h_nbr_coef].f[4], sf2r_h_high);

                    // Process conjugate complex numbers with twiddle factors
                    auto sf1i = &sf1r[h_nbr_coef];
                    auto sf2i = &sf1i[nbr_coef];
                    
                    // Preload twiddle values for inner loop
                    float c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Get current twiddle factors from preload
                        c = c_next;
                        s = s_next;
                        
                        // Prefetch next twiddle factors
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sf1r[j + 1], 0, 3);
                            __builtin_prefetch(&sf2r[j + 1], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Precompute twiddle factor vectors
                        float32x4_t c_vec = vdupq_n_f32(c);
                        float32x4_t s_vec = vdupq_n_f32(s);

                        // Load data
                        float32x4_t sf2r_j_low = vld1q_f32(&sf2r[j].f[0]);
                        float32x4_t sf2r_j_high = vld1q_f32(&sf2r[j].f[4]);
                        float32x4_t sf2i_j_low = vld1q_f32(&sf2i[j].f[0]);
                        float32x4_t sf2i_j_high = vld1q_f32(&sf2i[j].f[4]);
                        float32x4_t sf1r_j_low = vld1q_f32(&sf1r[j].f[0]);
                        float32x4_t sf1r_j_high = vld1q_f32(&sf1r[j].f[4]);
                        
                        // Compute first part: v = sf2r[j] * c - sf2i[j] * s
                        float32x4_t v_low = vfmsq_f32(vmulq_f32(sf2r_j_low, c_vec), sf2i_j_low, s_vec);
                        float32x4_t v_high = vfmsq_f32(vmulq_f32(sf2r_j_high, c_vec), sf2i_j_high, s_vec);
                        
                        // Store computed values
                        vst1q_f32(&dfr[j].f[0], vaddq_f32(sf1r_j_low, v_low));
                        vst1q_f32(&dfr[j].f[4], vaddq_f32(sf1r_j_high, v_high));
                        vst1q_f32(&dfi[-j].f[0], vsubq_f32(sf1r_j_low, v_low));
                        vst1q_f32(&dfi[-j].f[4], vsubq_f32(sf1r_j_high, v_high));

                        // Compute second part: v = sf2r[j] * s + sf2i[j] * c
                        float32x4_t sf1i_j_low = vld1q_f32(&sf1i[j].f[0]);
                        float32x4_t sf1i_j_high = vld1q_f32(&sf1i[j].f[4]);
                        
                        v_low = vfmaq_f32(vmulq_f32(sf2r_j_low, s_vec), sf2i_j_low, c_vec);
                        v_high = vfmaq_f32(vmulq_f32(sf2r_j_high, s_vec), sf2i_j_high, c_vec);
                        
                        vst1q_f32(&dfi[j].f[0], vaddq_f32(v_low, sf1i_j_low));
                        vst1q_f32(&dfi[j].f[4], vaddq_f32(v_high, sf1i_j_high));
                        vst1q_f32(&dfi[nbr_coef - j].f[0], vsubq_f32(v_low, sf1i_j_low));
                        vst1q_f32(&dfi[nbr_coef - j].f[4], vsubq_f32(v_high, sf1i_j_high));
                    }
                }
                // Prepare for next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        // Special cases for small FFTs remain unchanged
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_float8 b_0 = x[0] + x[2];
            const simd_float8 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        else {
            f[0] = x[0];
        }
    }
    
    void do_fft_neon_d8(const simd_double8 *_Nonnull x, simd_double8 *_Nonnull f)
    {
        if (likely(_nbr_bits > 2)) {
            simd_double8 *sf, *df;
            
            // Initial prefetch for better memory performance
            __builtin_prefetch(&x[0], 0, 3);
            __builtin_prefetch(&_bit_rev_lut.get_ptr()[0], 0, 3);

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }

            // First/second pass with bit-reversal and butterfly
            constexpr int PREFETCH_DISTANCE = 8;
            auto lut = _bit_rev_lut.get_ptr();
            
#pragma unroll 2
            for (auto i = 0; i < _N; i+=4) {
                if (likely(i + PREFETCH_DISTANCE < _N)) {
                    __builtin_prefetch(&lut[i + PREFETCH_DISTANCE], 0, 3);
                    __builtin_prefetch(&x[lut[i + PREFETCH_DISTANCE]], 0, 3);
                }
                
                auto df2 = &df[i];
                auto lut0 = &lut[i];
                auto lut1 = &lut[i+1];
                auto lut2 = &lut[i+2];
                auto lut3 = &lut[i+3];

                // Load with prefetch assistance
                float64x2_t x0_low  = vld1q_f64(&x[lut0].f[0]);
                float64x2_t x0_high = vld1q_f64(&x[lut0].f[2]);
                float64x2_t x1_low  = vld1q_f64(&x[lut1].f[0]);
                float64x2_t x1_high = vld1q_f64(&x[lut1].f[2]);
                float64x2_t x2_low  = vld1q_f64(&x[lut2].f[0]);
                float64x2_t x2_high = vld1q_f64(&x[lut2].f[2]);
                float64x2_t x3_low  = vld1q_f64(&x[lut3].f[0]);
                float64x2_t x3_high = vld1q_f64(&x[lut3].f[2]);
                
                // Compute sums and differences
                float64x2_t sum01_low = vaddq_f64(x0_low, x1_low);
                float64x2_t diff01_low = vsubq_f64(x0_low, x1_low);
                float64x2_t sum23_low = vaddq_f64(x2_low, x3_low);
                float64x2_t diff23_low = vsubq_f64(x2_low, x3_low);
                
                float64x2_t sum01_high = vaddq_f64(x0_high, x1_high);
                float64x2_t diff01_high = vsubq_f64(x0_high, x1_high);
                float64x2_t sum23_high = vaddq_f64(x2_high, x3_high);
                float64x2_t diff23_high = vsubq_f64(x2_high, x3_high);
                
                // Compute final results for butterfly
                float64x2_t sum0123_low = vaddq_f64(sum01_low, sum23_low);
                float64x2_t sum0123_high = vaddq_f64(sum01_high, sum23_high);
                float64x2_t diff0123_low = vsubq_f64(sum01_low, sum23_low);
                float64x2_t diff0123_high = vsubq_f64(sum01_high, sum23_high);

                // Store with optimized pattern for cache line utilization
                vst1q_f64(&df2[0].f[0], sum0123_low);
                vst1q_f64(&df2[0].f[2], sum0123_high);
                vst1q_f64(&df2[1].f[0], diff01_low);
                vst1q_f64(&df2[1].f[2], diff01_high);
                vst1q_f64(&df2[2].f[0], diff0123_low);
                vst1q_f64(&df2[2].f[2], diff0123_high);
                vst1q_f64(&df2[3].f[0], diff23_low);
                vst1q_f64(&df2[3].f[2], diff23_high);
            }

            // Third pass with software pipelining
            // Precomputed constants
            const float64x1_t neg_sq2_2 = vdup_n_f64(-SQ2_2);
            const float64x1_t pos_sq2_2 = vdup_n_f64(SQ2_2);
            
            // Preload first data block
            auto df_first = &df[0];
            float64x2_t df0_low_next = vld1q_f64(&df_first[0].f[0]);
            float64x2_t df0_high_next = vld1q_f64(&df_first[0].f[2]);
            float64x2_t df4_low_next = vld1q_f64(&df_first[4].f[0]);
            float64x2_t df4_high_next = vld1q_f64(&df_first[4].f[2]);
            
            for (auto i = 0; i < _N; i += 8) {
                auto sf2 = &sf[i];
                auto df2 = &df[i];
                
                // Software pipelining - use preloaded data
                float64x2_t df0_low = df0_low_next;
                float64x2_t df0_high = df0_high_next;
                float64x2_t df4_low = df4_low_next;
                float64x2_t df4_high = df4_high_next;
                
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&df[i+8], 0, 3);
                    __builtin_prefetch(&df[i+12], 0, 3);
                    
                    if (likely(i + 16 < _N)) {
                        df0_low_next = vld1q_f64(&df[i+8].f[0]);
                        df0_high_next = vld1q_f64(&df[i+8].f[2]);
                        df4_low_next = vld1q_f64(&df[i+12].f[0]);
                        df4_high_next = vld1q_f64(&df[i+12].f[2]);
                    }
                }
                
                // Compute and store in interleaved pattern for performance
                float64x2_t sum04_low = vaddq_f64(df0_low, df4_low);
                float64x2_t diff04_low = vsubq_f64(df0_low, df4_low);
                float64x2_t sum04_high = vaddq_f64(df0_high, df4_high);
                float64x2_t diff04_high = vsubq_f64(df0_high, df4_high);

                // Load more data in interleaved pattern
                float64x2_t df2val_low = vld1q_f64(&df2[2].f[0]);
                float64x2_t df2val_high = vld1q_f64(&df2[2].f[2]);
                float64x2_t df6val_low = vld1q_f64(&df2[6].f[0]);
                float64x2_t df6val_high = vld1q_f64(&df2[6].f[2]);
                
                float64x2_t df1_low = vld1q_f64(&df2[1].f[0]);
                float64x2_t df1_high = vld1q_f64(&df2[1].f[2]);
                float64x2_t df3_low = vld1q_f64(&df2[3].f[0]);
                float64x2_t df3_high = vld1q_f64(&df2[3].f[2]);
                float64x2_t df5_low = vld1q_f64(&df2[5].f[0]);
                float64x2_t df5_high = vld1q_f64(&df2[5].f[2]);
                float64x2_t df7_low = vld1q_f64(&df2[7].f[0]);
                float64x2_t df7_high = vld1q_f64(&df2[7].f[2]);

                // Store early results while continuing computation
                vst1q_f64(&sf2[0].f[0], sum04_low);
                vst1q_f64(&sf2[0].f[2], sum04_high);
                vst1q_f64(&sf2[4].f[0], diff04_low);
                vst1q_f64(&sf2[4].f[2], diff04_high);
                vst1q_f64(&sf2[2].f[0], df2val_low);
                vst1q_f64(&sf2[2].f[2], df2val_high);
                vst1q_f64(&sf2[6].f[0], df6val_low);
                vst1q_f64(&sf2[6].f[2], df6val_high);

                // Optimized FMA operations for double precision
                float64x2_t v1_low = vfmaq_lane_f64(vfmaq_lane_f64(vdupq_n_f64(0), df5_low, neg_sq2_2, 0), df7_low, neg_sq2_2, 0);
                float64x2_t v1_high = vfmaq_lane_f64(vfmaq_lane_f64(vdupq_n_f64(0), df5_high, neg_sq2_2, 0), df7_high, neg_sq2_2, 0);

                float64x2_t v2_low = vfmaq_lane_f64(vfmaq_lane_f64(vdupq_n_f64(0), df5_low, pos_sq2_2, 0), df7_low, pos_sq2_2, 0);
                float64x2_t v2_high = vfmaq_lane_f64(vfmaq_lane_f64(vdupq_n_f64(0), df5_high, pos_sq2_2, 0), df7_high, pos_sq2_2, 0);

                // Final stores
                vst1q_f64(&sf2[1].f[0], vaddq_f64(df1_low, v1_low));
                vst1q_f64(&sf2[1].f[2], vaddq_f64(df1_high, v1_high));
                vst1q_f64(&sf2[3].f[0], vsubq_f64(df1_low, v1_low));
                vst1q_f64(&sf2[3].f[2], vsubq_f64(df1_high, v1_high));
                
                vst1q_f64(&sf2[5].f[0], vaddq_f64(v2_low, df3_low));
                vst1q_f64(&sf2[5].f[2], vaddq_f64(v2_high, df3_high));
                vst1q_f64(&sf2[7].f[0], vsubq_f64(v2_low, df3_low));
                vst1q_f64(&sf2[7].f[2], vsubq_f64(v2_high, df3_high));
            }

            // Later passes optimized
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sf1r = sf + i;
                    auto sf2r = sf1r + nbr_coef;
                    auto dfr = df + i;
                    auto dfi = dfr + nbr_coef;

                    // Prefetch for reduced memory latency
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                        __builtin_prefetch(&sf[i + d_nbr_coef + nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients
                    float64x2_t sf1r_0_low = vld1q_f64(&sf1r[0].f[0]);
                    float64x2_t sf1r_0_high = vld1q_f64(&sf1r[0].f[2]);
                    float64x2_t sf2r_0_low = vld1q_f64(&sf2r[0].f[0]);
                    float64x2_t sf2r_0_high = vld1q_f64(&sf2r[0].f[2]);
                    
                    vst1q_f64(&dfr[0].f[0], vaddq_f64(sf1r_0_low, sf2r_0_low));
                    vst1q_f64(&dfr[0].f[2], vaddq_f64(sf1r_0_high, sf2r_0_high));
                    vst1q_f64(&dfi[0].f[0], vsubq_f64(sf1r_0_low, sf2r_0_low));
                    vst1q_f64(&dfi[0].f[2], vsubq_f64(sf1r_0_high, sf2r_0_high));
                    
                    vst1q_f64(&dfr[h_nbr_coef].f[0], vld1q_f64(&sf1r[h_nbr_coef].f[0]));
                    vst1q_f64(&dfr[h_nbr_coef].f[2], vld1q_f64(&sf1r[h_nbr_coef].f[2]));
                    vst1q_f64(&dfi[h_nbr_coef].f[0], vld1q_f64(&sf2r[h_nbr_coef].f[0]));
                    vst1q_f64(&dfi[h_nbr_coef].f[2], vld1q_f64(&sf2r[h_nbr_coef].f[2]));

                    // Process other conjugate complex numbers
                    auto sf1i = &sf1r[h_nbr_coef];
                    auto sf2i = &sf1i[nbr_coef];

                    // Preload first twiddle factors
                    double c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        double c = c_next;
                        double s = s_next;
                        
                        // Prefetch next twiddle factors and data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sf1r[j + 1], 0, 3);
                            __builtin_prefetch(&sf2r[j + 1], 0, 3);
                            __builtin_prefetch(&sf1i[j + 1], 0, 3);
                            __builtin_prefetch(&sf2i[j + 1], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Create vectorized twiddle factors
                        float64x1_t cos_val = vdup_n_f64(c);
                        float64x1_t sin_val = vdup_n_f64(s);

                        // Load data
                        float64x2_t sf1r_j_low = vld1q_f64(&sf1r[j].f[0]);
                        float64x2_t sf1r_j_high = vld1q_f64(&sf1r[j].f[2]);
                        float64x2_t sf2r_j_low = vld1q_f64(&sf2r[j].f[0]);
                        float64x2_t sf2r_j_high = vld1q_f64(&sf2r[j].f[2]);
                        float64x2_t sf2i_j_low = vld1q_f64(&sf2i[j].f[0]);
                        float64x2_t sf2i_j_high = vld1q_f64(&sf2i[j].f[2]);

                        // Compute v = sf2r[j] * c - sf2i[j] * s
        
                        // First calculate the products separately
                        float64x2_t sf2r_c_low = vmulq_lane_f64(sf2r_j_low, cos_val, 0);
                        float64x2_t sf2r_c_high = vmulq_lane_f64(sf2r_j_high, cos_val, 0);
                        float64x2_t sf2i_s_low = vmulq_lane_f64(sf2i_j_low, sin_val, 0);
                        float64x2_t sf2i_s_high = vmulq_lane_f64(sf2i_j_high, sin_val, 0);

                        // Then do the subtraction: sf2r * c - sf2i * s
                        float64x2_t v_low = vsubq_f64(sf2r_c_low, sf2i_s_low);
                        float64x2_t v_high = vsubq_f64(sf2r_c_high, sf2i_s_high);
                        
                        // Store results
                        vst1q_f64(&dfr[j].f[0], vaddq_f64(sf1r_j_low, v_low));
                        vst1q_f64(&dfr[j].f[2], vaddq_f64(sf1r_j_high, v_high));
                        vst1q_f64(&dfi[-j].f[0], vsubq_f64(sf1r_j_low, v_low));
                        vst1q_f64(&dfi[-j].f[2], vsubq_f64(sf1r_j_high, v_high));

                        // Compute v = sf2r[j] * s + sf2i[j] * c
                        float64x2_t sf1i_j_low = vld1q_f64(&sf1i[j].f[0]);
                        float64x2_t sf1i_j_high = vld1q_f64(&sf1i[j].f[2]);
                        
                        v_low = vfmaq_lane_f64(vmulq_lane_f64(sf2r_j_low, sin_val, 0), sf2i_j_low, cos_val, 0);
                        v_high = vfmaq_lane_f64(vmulq_lane_f64(sf2r_j_high, sin_val, 0), sf2i_j_high, cos_val, 0);
                        
                        vst1q_f64(&dfi[j].f[0], vaddq_f64(v_low, sf1i_j_low));
                        vst1q_f64(&dfi[j].f[2], vaddq_f64(v_high, sf1i_j_high));
                        vst1q_f64(&dfi[nbr_coef - j].f[0], vsubq_f64(v_low, sf1i_j_low));
                        vst1q_f64(&dfi[nbr_coef - j].f[2], vsubq_f64(v_high, sf1i_j_high));
                    }
                }
                // Prepare for next pass
                auto tmp = df;
                df = sf;
                sf = tmp;
            }
        }
        // Special cases remain unchanged
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_double8 b_0 = x[0] + x[2];
            const simd_double8 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        else if (_nbr_bits == 1) {
            f[0] = x[0] + x[1];
            f[1] = x[0] - x[1];
        }
        else {
            f[0] = x[0];
        }
    }
    
    void do_ifft_neon_f2(const simd_float2 *_Nonnull f, simd_float2 *_Nonnull x, bool do_scale = false)
    {
        const float32x2_t c2 = vdup_n_f32(2.0f);
        float32x2_t mul = vdup_n_f32(1.0f);
        
        // Initialize scaling factor
        if (unlikely(do_scale)) {
            mul = vdiv_f32(mul, vdup_n_f32((float)_N));
        }

        if (likely(_nbr_bits > 2)) {
            simd_float2 *sf = (simd_float2*)f;
            simd_float2 *df;
            simd_float2 *df_temp;

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }

            // First pass optimization
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sfr = &sf[i];
                    auto sfi = &sfr[nbr_coef];
                    auto df1r = &df[i];
                    auto df2r = &df1r[nbr_coef];

                    // Prefetch next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients with proper casting
                    float32x2_t sfr_0 = vld1_f32((float*)&sfr[0]);
                    float32x2_t sfr_nbr_coef = vld1_f32((float*)&sfr[nbr_coef]);
                    
                    vst1_f32((float*)&df1r[0], vadd_f32(sfr_0, sfr_nbr_coef));
                    vst1_f32((float*)&df2r[0], vsub_f32(sfr_0, sfr_nbr_coef));
                    
                    // Process h_nbr_coef with c2 multiplication
                    float32x2_t sfr_h_nbr_coef = vld1_f32((float*)&sfr[h_nbr_coef]);
                    float32x2_t sfi_h_nbr_coef = vld1_f32((float*)&sfi[h_nbr_coef]);
                    
                    vst1_f32((float*)&df1r[h_nbr_coef], vmul_f32(sfr_h_nbr_coef, c2));
                    vst1_f32((float*)&df2r[h_nbr_coef], vmul_f32(sfi_h_nbr_coef, c2));

                    // Process conjugate complex numbers
                    auto df1i = &df1r[h_nbr_coef];
                    auto df2i = &df1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    float c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache->get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        float c = c_next;
                        float s = s_next;
                        
                        // Prefetch next iteration data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sfr[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[-(j + 1)], 0, 3);
                            _twiddle_cache->get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load data with proper casting for float2
                        float32x2_t sfr_j = vld1_f32((float*)&sfr[j]);
                        float32x2_t sfi_neg_j = vld1_f32((float*)&sfi[-j]);
                        float32x2_t sfi_j = vld1_f32((float*)&sfi[j]);
                        float32x2_t sfi_nbr_j = vld1_f32((float*)&sfi[nbr_coef - j]);

                        // Calculate df1r and df1i with float2 operations
                        vst1_f32((float*)&df1r[j], vadd_f32(sfr_j, sfi_neg_j));
                        vst1_f32((float*)&df1i[j], vsub_f32(sfi_j, sfi_nbr_j));

                        // Calculate vr and vi
                        float32x2_t vr = vsub_f32(sfr_j, sfi_neg_j);
                        float32x2_t vi = vadd_f32(sfi_j, sfi_nbr_j);

                        // Calculate df2r and df2i with optimized operations
                        float32x2_t df2r_j = vmla_n_f32(vmul_n_f32(vr, c), vi, s);
                        float32x2_t df2i_j = vmls_n_f32(vmul_n_f32(vi, c), vr, s);
                        
                        // Store results
                        vst1_f32((float*)&df2r[j], df2r_j);
                        vst1_f32((float*)&df2i[j], df2i_j);
                    }
                }

                // Prepare for next pass
                if (pass < _nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }

            // Antepenultimate pass with SQ2_2 optimization
            const float32x2_t sq2_2 = vdup_n_f32(SQ2_2);
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&sf[i + 8], 0, 3);
                }

                // Load data with proper casting
                float32x2_t f0 = vld1_f32((float*)&sf[i]);
                float32x2_t f4 = vld1_f32((float*)&sf[i+4]);
                
                float32x2_t sum04 = vadd_f32(f0, f4);
                float32x2_t diff04 = vsub_f32(f0, f4);
                
                vst1_f32((float*)&df[i], sum04);
                vst1_f32((float*)&df[i+4], diff04);
                
                // Load and process c2 multiplications
                float32x2_t f2 = vld1_f32((float*)&sf[i+2]);
                float32x2_t f6 = vld1_f32((float*)&sf[i+6]);
                
                vst1_f32((float*)&df[i+2], vmul_f32(f2, c2));
                vst1_f32((float*)&df[i+6], vmul_f32(f6, c2));
                
                // Load additional data
                float32x2_t f1 = vld1_f32((float*)&sf[i+1]);
                float32x2_t f3 = vld1_f32((float*)&sf[i+3]);
                float32x2_t f5 = vld1_f32((float*)&sf[i+5]);
                float32x2_t f7 = vld1_f32((float*)&sf[i+7]);
                
                // Process simple additions
                vst1_f32((float*)&df[i+1], vadd_f32(f1, f3));
                vst1_f32((float*)&df[i+3], vsub_f32(f5, f7));
                
                // Calculate vr and vi for SQ2_2 operations
                float32x2_t vr = vsub_f32(f1, f3);
                float32x2_t vi = vadd_f32(f5, f7);
                
                // Optimized SQ2_2 calculation
                vst1_f32((float*)&df[i+5], vmul_f32(vadd_f32(vr, vi), sq2_2));
                vst1_f32((float*)&df[i+7], vmul_f32(vsub_f32(vi, vr), sq2_2));
            }

            // Penultimate and last pass with bit-reversal
            auto lut_ptr = _bit_rev_lut->get_ptr();
            
            for (auto i = 0; i < _N; i += 8) {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                // Prefetch output locations
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&x[lut[8]], 1, 0);
                }

                // Process first 4 outputs with proper casting
                {
                    float32x2_t sf0 = vld1_f32((float*)&sf2[0]);
                    float32x2_t sf1 = vld1_f32((float*)&sf2[1]);
                    float32x2_t sf2val = vld1_f32((float*)&sf2[2]);
                    float32x2_t sf3 = vld1_f32((float*)&sf2[3]);

                    // Butterfly calculation
                    float32x2_t b_0 = vadd_f32(sf0, sf2val);
                    float32x2_t b_2 = vsub_f32(sf0, sf2val);
                    float32x2_t b_1 = vmul_f32(sf1, c2);
                    float32x2_t b_3 = vmul_f32(sf3, c2);

                    // Apply scaling and store with bit-reversal
                    vst1_f32((float*)&x[lut[0]], vmul_f32(vadd_f32(b_0, b_1), mul));
                    vst1_f32((float*)&x[lut[1]], vmul_f32(vsub_f32(b_0, b_1), mul));
                    vst1_f32((float*)&x[lut[2]], vmul_f32(vadd_f32(b_2, b_3), mul));
                    vst1_f32((float*)&x[lut[3]], vmul_f32(vsub_f32(b_2, b_3), mul));
                }

                // Process second 4 outputs with proper casting
                {
                    float32x2_t sf4 = vld1_f32((float*)&sf2[4]);
                    float32x2_t sf5 = vld1_f32((float*)&sf2[5]);
                    float32x2_t sf6 = vld1_f32((float*)&sf2[6]);
                    float32x2_t sf7 = vld1_f32((float*)&sf2[7]);

                    // Butterfly calculation
                    float32x2_t b_0 = vadd_f32(sf4, sf6);
                    float32x2_t b_2 = vsub_f32(sf4, sf6);
                    float32x2_t b_1 = vmul_f32(sf5, c2);
                    float32x2_t b_3 = vmul_f32(sf7, c2);

                    // Apply scaling and store with bit-reversal
                    vst1_f32((float*)&x[lut[4]], vmul_f32(vadd_f32(b_0, b_1), mul));
                    vst1_f32((float*)&x[lut[5]], vmul_f32(vsub_f32(b_0, b_1), mul));
                    vst1_f32((float*)&x[lut[6]], vmul_f32(vadd_f32(b_2, b_3), mul));
                    vst1_f32((float*)&x[lut[7]], vmul_f32(vsub_f32(b_2, b_3), mul));
                }
            }
        }
        // Special cases for small IFFTs
        else if (unlikely(_nbr_bits == 2)) {
            // 4-point IFFT optimized with vectorized operations
            const float32x2_t b_0 = vadd_f32(vld1_f32((float*)&f[0]), vld1_f32((float*)&f[2]));
            const float32x2_t b_2 = vsub_f32(vld1_f32((float*)&f[0]), vld1_f32((float*)&f[2]));
            const float32x2_t f1_c2 = vmul_f32(vld1_f32((float*)&f[1]), c2);
            const float32x2_t f3_c2 = vmul_f32(vld1_f32((float*)&f[3]), c2);
            
            vst1_f32((float*)&x[0], vmul_f32(vadd_f32(b_0, f1_c2), mul));
            vst1_f32((float*)&x[2], vmul_f32(vsub_f32(b_0, f1_c2), mul));
            vst1_f32((float*)&x[1], vmul_f32(vadd_f32(b_2, f3_c2), mul));
            vst1_f32((float*)&x[3], vmul_f32(vsub_f32(b_2, f3_c2), mul));
        }
        else if (unlikely(_nbr_bits == 1)) {
            // 2-point IFFT
            vst1_f32((float*)&x[0], vmul_f32(vadd_f32(vld1_f32((float*)&f[0]), vld1_f32((float*)&f[1])), mul));
            vst1_f32((float*)&x[1], vmul_f32(vsub_f32(vld1_f32((float*)&f[0]), vld1_f32((float*)&f[1])), mul));
        }
        else {
            // 1-point IFFT
            vst1_f32((float*)&x[0], vmul_f32(vld1_f32((float*)&f[0]), mul));
        }
    }
    
    void do_ifft_neon_f4(const simd_float4 *_Nonnull f, simd_float4 *_Nonnull x, bool do_scale = false)
    {
        const float32x4_t c2 = vdupq_n_f32(2.0f);
        
        // Initialize scaling factor
        float32x4_t mul = vdupq_n_f32(1.0f);
        if (unlikely(do_scale)) {
            mul = vdivq_f32(mul, vdupq_n_f32((float)_N));
        }

        if (likely(_nbr_bits > 2)) {
            simd_float4 *sf = (simd_float4*)f;
            simd_float4 *df;
            simd_float4 *df_temp;

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }

            // First pass with prefetching and optimization
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sfr = &sf[i];
                    auto sfi = &sfr[nbr_coef];
                    auto df1r = &df[i];
                    auto df2r = &df1r[nbr_coef];

                    // Prefetch next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients
                    float32x4_t sfr_0 = vld1q_f32((float*)&sfr[0]);
                    float32x4_t sfr_nbr_coef = vld1q_f32(&sfr[nbr_coef].f[0]);
                    
                    vst1q_f32((float*)&df1r[0], vaddq_f32(sfr_0, sfr_nbr_coef));
                    vst1q_f32(&df2r[0].f[0], vsubq_f32(sfr_0, sfr_nbr_coef));
                    
                    // Process h_nbr_coef coefficients with c2 multiply
                    float32x4_t sfr_h_nbr_coef = vld1q_f32(&sfr[h_nbr_coef].f[0]);
                    float32x4_t sfi_h_nbr_coef = vld1q_f32(&sfi[h_nbr_coef].f[0]);
                    
                    vst1q_f32(&df1r[h_nbr_coef].f[0], vmulq_f32(sfr_h_nbr_coef, c2));
                    vst1q_f32(&df2r[h_nbr_coef].f[0], vmulq_f32(sfi_h_nbr_coef, c2));

                    // Process conjugate complex numbers
                    auto df1i = &df1r[h_nbr_coef];
                    auto df2i = &df1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    float c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        float c = c_next;
                        float s = s_next;
                        
                        // Prefetch next iteration data and twiddle factors
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sfr[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[-(j + 1)], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load vectorized twiddle factors
                        float32x4_t c_vec = vdupq_n_f32(c);
                        float32x4_t s_vec = vdupq_n_f32(s);

                        // Load data with optimized memory access patterns
                        float32x4_t sfr_j = vld1q_f32((float*)&sfr[j]);
                        float32x4_t sfi_neg_j = vld1q_f32(&sfi[-j].f[0]);
                        float32x4_t sfi_j = vld1q_f32(&sfi[j].f[0]);
                        float32x4_t sfi_nbr_j = vld1q_f32(&sfi[nbr_coef - j].f[0]);

                        // Calculate df1r and df1i
                        vst1q_f32((float*)&df1r[j], vaddq_f32(sfr_j, sfi_neg_j));
                        vst1q_f32(&df1i[j].f[0], vsubq_f32(sfi_j, sfi_nbr_j));

                        // Calculate vr and vi
                        float32x4_t vr = vsubq_f32(sfr_j, sfi_neg_j);
                        float32x4_t vi = vaddq_f32(sfi_j, sfi_nbr_j);

                        // Calculate with optimized FMA
                        float32x4_t df2r_j = vfmaq_f32(vmulq_f32(vr, c_vec), vi, s_vec);
                        float32x4_t df2i_j = vfmsq_f32(vmulq_f32(vi, c_vec), vr, s_vec);
                        
                        // Store final results
                        vst1q_f32(&df2r[j].f[0], df2r_j);
                        vst1q_f32(&df2i[j].f[0], df2i_j);
                    }
                }

                // Prepare for next pass
                if (pass < _nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }

            // Antepenultimate pass with SQ2_2 optimization
            const float32x4_t sq2_2 = vdupq_n_f32(SQ2_2);
            
            for (auto i = 0; i < _N; i += 8) {
                auto df2 = &df[i];
                auto sf2 = &sf[i];
                
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&sf[i + 8], 0, 3);
                }

                // Load and process data in interleaved fashion
                float32x4_t f0 = vld1q_f32((float*)&sf2[0]);
                float32x4_t f4 = vld1q_f32((float*)&sf2[4]);
                
                float32x4_t sum04 = vaddq_f32(f0, f4);
                float32x4_t diff04 = vsubq_f32(f0, f4);
                
                vst1q_f32((float*)&df2[0], sum04);
                vst1q_f32((float*)&df2[4], diff04);
                
                // Load and process c2 multiplications
                float32x4_t f2 = vld1q_f32((float*)&sf2[2]);
                float32x4_t f6 = vld1q_f32((float*)&sf2[6]);
                
                vst1q_f32((float*)&df2[2], vmulq_f32(f2, c2));
                vst1q_f32((float*)&df2[6], vmulq_f32(f6, c2));
                
                // Load additional data
                float32x4_t f1 = vld1q_f32((float*)&sf2[1]);
                float32x4_t f3 = vld1q_f32((float*)&sf2[3]);
                float32x4_t f5 = vld1q_f32((float*)&sf2[5]);
                float32x4_t f7 = vld1q_f32((float*)&sf2[7]);
                
                // Store simple additions and subtractions
                vst1q_f32((float*)&df2[1], vaddq_f32(f1, f3));
                vst1q_f32((float*)&df2[3], vsubq_f32(f5, f7));
                
                // Calculate vr and vi for SQ2_2 operations
                float32x4_t vr = vsubq_f32(f1, f3);
                float32x4_t vi = vaddq_f32(f5, f7);
                
                // Optimized SQ2_2 calculation using FMA
                vst1q_f32((float*)&df2[5], vmulq_f32(vaddq_f32(vr, vi), sq2_2));
                vst1q_f32((float*)&df2[7], vmulq_f32(vsubq_f32(vi, vr), sq2_2));
            }

            // Penultimate and last pass with bit-reversal
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
            for (auto i = 0; i < _N; i += 8) {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                // Prefetch output locations
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&x[lut[8]], 1, 0);
                }

                // Process first 4 outputs
                {
                    float32x4_t sf0 = vld1q_f32((float*)&sf2[0]);
                    float32x4_t sf1 = vld1q_f32((float*)&sf2[1]);
                    float32x4_t sf2val = vld1q_f32((float*)&sf2[2]);
                    float32x4_t sf3 = vld1q_f32((float*)&sf2[3]);

                    // Butterfly calculation
                    float32x4_t b_0 = vaddq_f32(sf0, sf2val);
                    float32x4_t b_2 = vsubq_f32(sf0, sf2val);
                    float32x4_t b_1 = vmulq_f32(sf1, c2);
                    float32x4_t b_3 = vmulq_f32(sf3, c2);

                    // Apply scaling and store using bit-reversal
                    vst1q_f32(&x[lut[0]].f[0], vmulq_f32(vaddq_f32(b_0, b_1), mul));
                    vst1q_f32(&x[lut[1]].f[0], vmulq_f32(vsubq_f32(b_0, b_1), mul));
                    vst1q_f32(&x[lut[2]].f[0], vmulq_f32(vaddq_f32(b_2, b_3), mul));
                    vst1q_f32(&x[lut[3]].f[0], vmulq_f32(vsubq_f32(b_2, b_3), mul));
                }

                // Process second 4 outputs
                {
                    float32x4_t sf4 = vld1q_f32((float*)&sf2[4]);
                    float32x4_t sf5 = vld1q_f32((float*)&sf2[5]);
                    float32x4_t sf6 = vld1q_f32((float*)&sf2[6]);
                    float32x4_t sf7 = vld1q_f32((float*)&sf2[7]);

                    // Butterfly calculation
                    float32x4_t b_0 = vaddq_f32(sf4, sf6);
                    float32x4_t b_2 = vsubq_f32(sf4, sf6);
                    float32x4_t b_1 = vmulq_f32(sf5, c2);
                    float32x4_t b_3 = vmulq_f32(sf7, c2);

                    // Apply scaling and store using bit-reversal
                    vst1q_f32(&x[lut[4]].f[0], vmulq_f32(vaddq_f32(b_0, b_1), mul));
                    vst1q_f32(&x[lut[5]].f[0], vmulq_f32(vsubq_f32(b_0, b_1), mul));
                    vst1q_f32(&x[lut[6]].f[0], vmulq_f32(vaddq_f32(b_2, b_3), mul));
                    vst1q_f32(&x[lut[7]].f[0], vmulq_f32(vsubq_f32(b_2, b_3), mul));
                }
            }
        }
        // Special cases for small IFFTs
        else if (unlikely(_nbr_bits == 2)) {
            // 4-point IFFT optimized with vectorized operations
            const float32x4_t b_0 = vaddq_f32(vld1q_f32((float*)&f[0]), vld1q_f32((float*)&f[2]));
            const float32x4_t b_2 = vsubq_f32(vld1q_f32((float*)&f[0]), vld1q_f32((float*)&f[2]));
            const float32x4_t f1_c2 = vmulq_f32(vld1q_f32((float*)&f[1]), c2);
            const float32x4_t f3_c2 = vmulq_f32(vld1q_f32((float*)&f[3]), c2);
            
            vst1q_f32((float*)&x[0], vmulq_f32(vaddq_f32(b_0, f1_c2), mul));
            vst1q_f32((float*)&x[2], vmulq_f32(vsubq_f32(b_0, f1_c2), mul));
            vst1q_f32((float*)&x[1], vmulq_f32(vaddq_f32(b_2, f3_c2), mul));
            vst1q_f32((float*)&x[3], vmulq_f32(vsubq_f32(b_2, f3_c2), mul));
        }
        else if (unlikely(_nbr_bits == 1)) {
            // 2-point IFFT with vectorized operations
            vst1q_f32((float*)&x[0], vmulq_f32(vaddq_f32(vld1q_f32((float*)&f[0]), vld1q_f32((float*)&f[1])), mul));
            vst1q_f32((float*)&x[1], vmulq_f32(vsubq_f32(vld1q_f32((float*)&f[0]), vld1q_f32((float*)&f[1])), mul));
        }
        else {
            // 1-point IFFT
            vst1q_f32((float*)&x[0], vmulq_f32(vld1q_f32((float*)&f[0]), mul));
        }
    }
    
    void do_ifft_neon_d2(const simd_double2 *_Nonnull f, simd_double2 *_Nonnull x, bool do_scale = false)
    {
        // Initialize constants
        const float64x2_t c2 = vdupq_n_f64(2.0);
        
        // Initialize scaling factor
        float64x2_t mul = vdupq_n_f64(1.0);
        if (unlikely(do_scale)) {
            mul = vdivq_f64(mul, vdupq_n_f64((double)_N));
        }

        if (likely(_nbr_bits > 2)) {
            simd_double2 *sf = (simd_double2*)f;
            simd_double2 *df;
            simd_double2 *df_temp;

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }

            // First pass optimization
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sfr = &sf[i];
                    auto sfi = &sfr[nbr_coef];
                    auto df1r = &df[i];
                    auto df2r = &df1r[nbr_coef];

                    // Prefetch next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }

                    // Process extreme coefficients with proper casting
                    float64x2_t sfr_0 = vld1q_f64((double*)&sfr[0]);
                    float64x2_t sfr_nbr_coef = vld1q_f64((double*)&sfr[nbr_coef]);
                    
                    vst1q_f64((double*)&df1r[0], vaddq_f64(sfr_0, sfr_nbr_coef));
                    vst1q_f64((double*)&df2r[0], vsubq_f64(sfr_0, sfr_nbr_coef));
                    
                    // Process h_nbr_coef coefficients with c2 multiply
                    float64x2_t sfr_h_nbr_coef = vld1q_f64((double*)&sfr[h_nbr_coef]);
                    float64x2_t sfi_h_nbr_coef = vld1q_f64((double*)&sfi[h_nbr_coef]);
                    
                    vst1q_f64((double*)&df1r[h_nbr_coef], vmulq_f64(sfr_h_nbr_coef, c2));
                    vst1q_f64((double*)&df2r[h_nbr_coef], vmulq_f64(sfi_h_nbr_coef, c2));

                    // Process conjugate complex numbers
                    auto df1i = &df1r[h_nbr_coef];
                    auto df2i = &df1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    double c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache->get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        double c = c_next;
                        double s = s_next;
                        
                        // Prefetch next iteration data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sfr[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[-(j + 1)], 0, 3);
                            _twiddle_cache->get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Load data with proper casting
                        float64x2_t sfr_j = vld1q_f64((double*)&sfr[j]);
                        float64x2_t sfi_neg_j = vld1q_f64((double*)&sfi[-j]);
                        float64x2_t sfi_j = vld1q_f64((double*)&sfi[j]);
                        float64x2_t sfi_nbr_j = vld1q_f64((double*)&sfi[nbr_coef - j]);

                        // Calculate df1r and df1i values
                        vst1q_f64((double*)&df1r[j], vaddq_f64(sfr_j, sfi_neg_j));
                        vst1q_f64((double*)&df1i[j], vsubq_f64(sfi_j, sfi_nbr_j));

                        // Calculate vr and vi
                        float64x2_t vr = vsubq_f64(sfr_j, sfi_neg_j);
                        float64x2_t vi = vaddq_f64(sfi_j, sfi_nbr_j);

                        // Calculate with scalar multiplication (no vmlaq_lane_f64)
                        float64x2_t c_vec = vdupq_n_f64(c);
                        float64x2_t s_vec = vdupq_n_f64(s);
                        
                        float64x2_t vr_c = vmulq_f64(vr, c_vec);
                        float64x2_t vi_s = vmulq_f64(vi, s_vec);
                        float64x2_t df2r_j = vaddq_f64(vr_c, vi_s);
                        
                        float64x2_t vi_c = vmulq_f64(vi, c_vec);
                        float64x2_t vr_s = vmulq_f64(vr, s_vec);
                        float64x2_t df2i_j = vsubq_f64(vi_c, vr_s);
                        
                        // Store results
                        vst1q_f64((double*)&df2r[j], df2r_j);
                        vst1q_f64((double*)&df2i[j], df2i_j);
                    }
                }

                // Prepare for next pass
                if (pass < _nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }

            // Antepenultimate pass with SQ2_2 optimization
            const float64x2_t sq2_2 = vdupq_n_f64(SQ2_2);
            
            for (auto i = 0; i < _N; i += 8) {
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&sf[i + 8], 0, 3);
                }

                // Load with proper casting
                float64x2_t f0 = vld1q_f64((double*)&sf[i]);
                float64x2_t f4 = vld1q_f64((double*)&sf[i+4]);
                
                float64x2_t sum04 = vaddq_f64(f0, f4);
                float64x2_t diff04 = vsubq_f64(f0, f4);
                
                // Store sums and differences
                vst1q_f64((double*)&df[i], sum04);
                vst1q_f64((double*)&df[i+4], diff04);
                
                // Load and process c2 multiplications
                float64x2_t f2 = vld1q_f64((double*)&sf[i+2]);
                float64x2_t f6 = vld1q_f64((double*)&sf[i+6]);
                
                vst1q_f64((double*)&df[i+2], vmulq_f64(f2, c2));
                vst1q_f64((double*)&df[i+6], vmulq_f64(f6, c2));
                
                // Load additional data
                float64x2_t f1 = vld1q_f64((double*)&sf[i+1]);
                float64x2_t f3 = vld1q_f64((double*)&sf[i+3]);
                float64x2_t f5 = vld1q_f64((double*)&sf[i+5]);
                float64x2_t f7 = vld1q_f64((double*)&sf[i+7]);
                
                // Store simple additions and subtractions
                vst1q_f64((double*)&df[i+1], vaddq_f64(f1, f3));
                vst1q_f64((double*)&df[i+3], vsubq_f64(f5, f7));
                
                // Calculate vr and vi for SQ2_2 operations
                float64x2_t vr = vsubq_f64(f1, f3);
                float64x2_t vi = vaddq_f64(f5, f7);
                
                // Calculate SQ2_2 multiplications
                vst1q_f64((double*)&df[i+5], vmulq_f64(vaddq_f64(vr, vi), sq2_2));
                vst1q_f64((double*)&df[i+7], vmulq_f64(vsubq_f64(vi, vr), sq2_2));
            }

            // Penultimate and last pass with bit-reversal
            auto lut_ptr = _bit_rev_lut->get_ptr();
            
            for (auto i = 0; i < _N; i += 8) {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                // Prefetch output locations
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&x[lut[8]], 1, 0);
                }

                // Process first 4 outputs
                {
                    float64x2_t sf0 = vld1q_f64((double*)&sf2[0]);
                    float64x2_t sf1 = vld1q_f64((double*)&sf2[1]);
                    float64x2_t sf2val = vld1q_f64((double*)&sf2[2]);
                    float64x2_t sf3 = vld1q_f64((double*)&sf2[3]);

                    // Calculate butterfly values
                    float64x2_t b_0 = vaddq_f64(sf0, sf2val);
                    float64x2_t b_2 = vsubq_f64(sf0, sf2val);
                    float64x2_t b_1 = vmulq_f64(sf1, c2);
                    float64x2_t b_3 = vmulq_f64(sf3, c2);

                    // Apply scaling and store with bit-reversal
                    vst1q_f64((double*)&x[lut[0]], vmulq_f64(vaddq_f64(b_0, b_1), mul));
                    vst1q_f64((double*)&x[lut[1]], vmulq_f64(vsubq_f64(b_0, b_1), mul));
                    vst1q_f64((double*)&x[lut[2]], vmulq_f64(vaddq_f64(b_2, b_3), mul));
                    vst1q_f64((double*)&x[lut[3]], vmulq_f64(vsubq_f64(b_2, b_3), mul));
                }

                // Process second 4 outputs
                {
                    float64x2_t sf4 = vld1q_f64((double*)&sf2[4]);
                    float64x2_t sf5 = vld1q_f64((double*)&sf2[5]);
                    float64x2_t sf6 = vld1q_f64((double*)&sf2[6]);
                    float64x2_t sf7 = vld1q_f64((double*)&sf2[7]);

                    // Calculate butterfly values
                    float64x2_t b_0 = vaddq_f64(sf4, sf6);
                    float64x2_t b_2 = vsubq_f64(sf4, sf6);
                    float64x2_t b_1 = vmulq_f64(sf5, c2);
                    float64x2_t b_3 = vmulq_f64(sf7, c2);

                    // Apply scaling and store with bit-reversal
                    vst1q_f64((double*)&x[lut[4]], vmulq_f64(vaddq_f64(b_0, b_1), mul));
                    vst1q_f64((double*)&x[lut[5]], vmulq_f64(vsubq_f64(b_0, b_1), mul));
                    vst1q_f64((double*)&x[lut[6]], vmulq_f64(vaddq_f64(b_2, b_3), mul));
                    vst1q_f64((double*)&x[lut[7]], vmulq_f64(vsubq_f64(b_2, b_3), mul));
                }
            }
        }
        // Special cases for small IFFTs
        else if (unlikely(_nbr_bits == 2)) {
            // 4-point IFFT
            const float64x2_t b_0 = vaddq_f64(vld1q_f64((double*)&f[0]), vld1q_f64((double*)&f[2]));
            const float64x2_t b_2 = vsubq_f64(vld1q_f64((double*)&f[0]), vld1q_f64((double*)&f[2]));
            const float64x2_t f1_c2 = vmulq_f64(vld1q_f64((double*)&f[1]), c2);
            const float64x2_t f3_c2 = vmulq_f64(vld1q_f64((double*)&f[3]), c2);
            
            vst1q_f64((double*)&x[0], vmulq_f64(vaddq_f64(b_0, f1_c2), mul));
            vst1q_f64((double*)&x[2], vmulq_f64(vsubq_f64(b_0, f1_c2), mul));
            vst1q_f64((double*)&x[1], vmulq_f64(vaddq_f64(b_2, f3_c2), mul));
            vst1q_f64((double*)&x[3], vmulq_f64(vsubq_f64(b_2, f3_c2), mul));
        }
        else if (unlikely(_nbr_bits == 1)) {
            // 2-point IFFT
            vst1q_f64((double*)&x[0], vmulq_f64(vaddq_f64(vld1q_f64((double*)&f[0]), vld1q_f64((double*)&f[1])), mul));
            vst1q_f64((double*)&x[1], vmulq_f64(vsubq_f64(vld1q_f64((double*)&f[0]), vld1q_f64((double*)&f[1])), mul));
        }
        else {
            // 1-point IFFT
            vst1q_f64((double*)&x[0], vmulq_f64(vld1q_f64((double*)&f[0]), mul));
        }
    }
    
    void do_ifft_neon_d4(const simd_double4 *_Nonnull f, simd_double4 *_Nonnull x, bool do_scale = false)
    {
        // Initialize constants
        const float64x2_t c2 = vdupq_n_f64(2.0);
        
        // Initialize scaling factor
        float64x2_t mul = vdupq_n_f64(1.0);
        if (unlikely(do_scale)) {
            mul = vdivq_f64(mul, vdupq_n_f64((double)_N));
        }
        
        if (likely(_nbr_bits > 2)) {
            simd_double4 *sf = (simd_double4*)f;
            simd_double4 *df;
            simd_double4 *df_temp;
            
            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }
            
            // First pass optimization
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sfr = &sf[i];
                    auto sfi = &sfr[nbr_coef];
                    auto df1r = &df[i];
                    auto df2r = &df1r[nbr_coef];
                    
                    // Prefetch next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                    }
                    
                    // Process extreme coefficients for double4
                    float64x2_t sfr_0_low = vld1q_f64((double*)&sfr[0]);
                    float64x2_t sfr_0_high = vld1q_f64((double*)&sfr[0] + 2);
                    float64x2_t sfr_nbr_coef_low = vld1q_f64(&sfr[nbr_coef].f[0]);
                    float64x2_t sfr_nbr_coef_high = vld1q_f64(&sfr[nbr_coef].f[2]);
                    
                    vst1q_f64((double*)&df1r[0], vaddq_f64(sfr_0_low, sfr_nbr_coef_low));
                    vst1q_f64((double*)&df1r[0] + 2, vaddq_f64(sfr_0_high, sfr_nbr_coef_high));
                    vst1q_f64(&df2r[0].f[0], vsubq_f64(sfr_0_low, sfr_nbr_coef_low));
                    vst1q_f64(&df2r[0].f[2], vsubq_f64(sfr_0_high, sfr_nbr_coef_high));
                    
                    // Process h_nbr_coef with c2 multiply
                    float64x2_t sfr_h_nbr_coef_low = vld1q_f64(&sfr[h_nbr_coef].f[0]);
                    float64x2_t sfr_h_nbr_coef_high = vld1q_f64(&sfr[h_nbr_coef].f[2]);
                    float64x2_t sfi_h_nbr_coef_low = vld1q_f64(&sfi[h_nbr_coef].f[0]);
                    float64x2_t sfi_h_nbr_coef_high = vld1q_f64(&sfi[h_nbr_coef].f[2]);
                    
                    vst1q_f64(&df1r[h_nbr_coef].f[0], vmulq_f64(sfr_h_nbr_coef_low, c2));
                    vst1q_f64(&df1r[h_nbr_coef].f[2], vmulq_f64(sfr_h_nbr_coef_high, c2));
                    vst1q_f64(&df2r[h_nbr_coef].f[0], vmulq_f64(sfi_h_nbr_coef_low, c2));
                    vst1q_f64(&df2r[h_nbr_coef].f[2], vmulq_f64(sfi_h_nbr_coef_high, c2));
                    
                    // Process conjugate complex numbers
                    auto df1i = &df1r[h_nbr_coef];
                    auto df2i = &df1i[nbr_coef];
                    
                    // Preload first twiddle factors
                    double c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }
                    
                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        double c = c_next;
                        double s = s_next;
                        
                        // Prefetch next iteration data
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sfr[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[-(j + 1)], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }
                        
                        // Create scalar twiddle factors
                        float64x1_t cos_val = vdup_n_f64(c);
                        float64x1_t sin_val = vdup_n_f64(s);
                        
                        // Load data with optimized patterns for double4
                        float64x2_t sfr_j_low = vld1q_f64((double*)&sfr[j]);
                        float64x2_t sfr_j_high = vld1q_f64((double*)&sfr[j] + 2);
                        float64x2_t sfi_neg_j_low = vld1q_f64(&sfi[-j].f[0]);
                        float64x2_t sfi_neg_j_high = vld1q_f64(&sfi[-j].f[2]);
                        float64x2_t sfi_j_low = vld1q_f64(&sfi[j].f[0]);
                        float64x2_t sfi_j_high = vld1q_f64(&sfi[j].f[2]);
                        float64x2_t sfi_nbr_j_low = vld1q_f64(&sfi[nbr_coef-j].f[0]);
                        float64x2_t sfi_nbr_j_high = vld1q_f64(&sfi[nbr_coef-j].f[2]);
                        
                        // Calculate df1r and df1i
                        vst1q_f64((double*)&df1r[j], vaddq_f64(sfr_j_low, sfi_neg_j_low));
                        vst1q_f64((double*)&df1r[j] + 2, vaddq_f64(sfr_j_high, sfi_neg_j_high));
                        vst1q_f64(&df1i[j].f[0], vsubq_f64(sfi_j_low, sfi_nbr_j_low));
                        vst1q_f64(&df1i[j].f[2], vsubq_f64(sfi_j_high, sfi_nbr_j_high));
                        
                        // Calculate vr and vi
                        float64x2_t vr_low = vsubq_f64(sfr_j_low, sfi_neg_j_low);
                        float64x2_t vr_high = vsubq_f64(sfr_j_high, sfi_neg_j_high);
                        float64x2_t vi_low = vaddq_f64(sfi_j_low, sfi_nbr_j_low);
                        float64x2_t vi_high = vaddq_f64(sfi_j_high, sfi_nbr_j_high);
                        
                        // Calculate df2r and df2i values properly for doubles
                        float64x2_t vr_c_low = vmulq_lane_f64(vr_low, cos_val, 0);
                        float64x2_t vr_c_high = vmulq_lane_f64(vr_high, cos_val, 0);
                        float64x2_t vi_s_low = vmulq_lane_f64(vi_low, sin_val, 0);
                        float64x2_t vi_s_high = vmulq_lane_f64(vi_high, sin_val, 0);
                        
                        float64x2_t df2r_j_low = vaddq_f64(vr_c_low, vi_s_low);
                        float64x2_t df2r_j_high = vaddq_f64(vr_c_high, vi_s_high);
                        
                        // Second part for df2i
                        float64x2_t vi_c_low = vmulq_lane_f64(vi_low, cos_val, 0);
                        float64x2_t vi_c_high = vmulq_lane_f64(vi_high, cos_val, 0);
                        float64x2_t vr_s_low = vmulq_lane_f64(vr_low, sin_val, 0);
                        float64x2_t vr_s_high = vmulq_lane_f64(vr_high, sin_val, 0);
                        
                        float64x2_t df2i_j_low = vsubq_f64(vi_c_low, vr_s_low);
                        float64x2_t df2i_j_high = vsubq_f64(vi_c_high, vr_s_high);
                        
                        // Store results
                        vst1q_f64(&df2r[j].f[0], df2r_j_low);
                        vst1q_f64(&df2r[j].f[2], df2r_j_high);
                        vst1q_f64(&df2i[j].f[0], df2i_j_low);
                        vst1q_f64(&df2i[j].f[2], df2i_j_high);
                    }
                }
                
                // Prepare for next pass
                if (pass < _nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }
            
            // Antepenultimate pass with optimized SQ2_2 calculations
            const float64x1_t sq2_2 = vdup_n_f64(SQ2_2);
            
            for (auto i = 0; i < _N; i += 8) {
                auto df2 = &df[i];
                auto sf2 = &sf[i];
                
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&sf[i + 8], 0, 3);
                }
                
                // Load and process data in halves for double4
                float64x2_t f0_low = vld1q_f64((double*)&sf2[0]);
                float64x2_t f0_high = vld1q_f64((double*)&sf2[0] + 2);
                float64x2_t f4_low = vld1q_f64((double*)&sf2[4]);
                float64x2_t f4_high = vld1q_f64((double*)&sf2[4] + 2);
                
                // Compute sums and differences
                float64x2_t sum04_low = vaddq_f64(f0_low, f4_low);
                float64x2_t sum04_high = vaddq_f64(f0_high, f4_high);
                float64x2_t diff04_low = vsubq_f64(f0_low, f4_low);
                float64x2_t diff04_high = vsubq_f64(f0_high, f4_high);
                
                // Store first part of results
                vst1q_f64((double*)&df2[0], sum04_low);
                vst1q_f64((double*)&df2[0] + 2, sum04_high);
                vst1q_f64((double*)&df2[4], diff04_low);
                vst1q_f64((double*)&df2[4] + 2, diff04_high);
                
                // Load c2 multiplication data
                float64x2_t f2_low = vld1q_f64((double*)&sf2[2]);
                float64x2_t f2_high = vld1q_f64((double*)&sf2[2] + 2);
                float64x2_t f6_low = vld1q_f64((double*)&sf2[6]);
                float64x2_t f6_high = vld1q_f64((double*)&sf2[6] + 2);
                
                // Store c2 multiplications
                vst1q_f64((double*)&df2[2], vmulq_f64(f2_low, c2));
                vst1q_f64((double*)&df2[2] + 2, vmulq_f64(f2_high, c2));
                vst1q_f64((double*)&df2[6], vmulq_f64(f6_low, c2));
                vst1q_f64((double*)&df2[6] + 2, vmulq_f64(f6_high, c2));
                
                // Load additional data
                float64x2_t f1_low = vld1q_f64((double*)&sf2[1]);
                float64x2_t f1_high = vld1q_f64((double*)&sf2[1] + 2);
                float64x2_t f3_low = vld1q_f64((double*)&sf2[3]);
                float64x2_t f3_high = vld1q_f64((double*)&sf2[3] + 2);
                float64x2_t f5_low = vld1q_f64((double*)&sf2[5]);
                float64x2_t f5_high = vld1q_f64((double*)&sf2[5] + 2);
                float64x2_t f7_low = vld1q_f64((double*)&sf2[7]);
                float64x2_t f7_high = vld1q_f64((double*)&sf2[7] + 2);
                
                // Store simple additions
                vst1q_f64((double*)&df2[1], vaddq_f64(f1_low, f3_low));
                vst1q_f64((double*)&df2[1] + 2, vaddq_f64(f1_high, f3_high));
                vst1q_f64((double*)&df2[3], vsubq_f64(f5_low, f7_low));
                vst1q_f64((double*)&df2[3] + 2, vsubq_f64(f5_high, f7_high));
                
                // Calculate vr and vi
                float64x2_t vr_low = vsubq_f64(f1_low, f3_low);
                float64x2_t vr_high = vsubq_f64(f1_high, f3_high);
                float64x2_t vi_low = vaddq_f64(f5_low, f7_low);
                float64x2_t vi_high = vaddq_f64(f5_high, f7_high);
                
                // Calculate SQ2_2 multiplications properly for doubles
                float64x2_t df5_low = vmulq_lane_f64(vaddq_f64(vr_low, vi_low), sq2_2, 0);
                float64x2_t df5_high = vmulq_lane_f64(vaddq_f64(vr_high, vi_high), sq2_2, 0);
                float64x2_t df7_low = vmulq_lane_f64(vsubq_f64(vi_low, vr_low), sq2_2, 0);
                float64x2_t df7_high = vmulq_lane_f64(vsubq_f64(vi_high, vr_high), sq2_2, 0);
                
                // Store SQ2_2 results
                vst1q_f64((double*)&df2[5], df5_low);
                vst1q_f64((double*)&df2[5] + 2, df5_high);
                vst1q_f64((double*)&df2[7], df7_low);
                vst1q_f64((double*)&df2[7] + 2, df7_high);
            }
            
            // Penultimate and last pass with bit reversal
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
            for (auto i = 0; i < _N; i += 8) {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                // Prefetch output locations
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&x[lut[8]], 1, 0);
                }
                
                // Process first 4 outputs
                {
                    float64x2_t sf0_low = vld1q_f64((double*)&sf2[0]);
                    float64x2_t sf0_high = vld1q_f64((double*)&sf2[0] + 2);
                    float64x2_t sf1_low = vld1q_f64((double*)&sf2[1]);
                    float64x2_t sf1_high = vld1q_f64((double*)&sf2[1] + 2);
                    float64x2_t sf2val_low = vld1q_f64((double*)&sf2[2]);
                    float64x2_t sf2val_high = vld1q_f64((double*)&sf2[2] + 2);
                    float64x2_t sf3_low = vld1q_f64((double*)&sf2[3]);
                    float64x2_t sf3_high = vld1q_f64((double*)&sf2[3] + 2);
                    
                    // Calculate butterfly values
                    float64x2_t b_0_low = vaddq_f64(sf0_low, sf2val_low);
                    float64x2_t b_0_high = vaddq_f64(sf0_high, sf2val_high);
                    float64x2_t b_2_low = vsubq_f64(sf0_low, sf2val_low);
                    float64x2_t b_2_high = vsubq_f64(sf0_high, sf2val_high);
                    float64x2_t b_1_low = vmulq_f64(sf1_low, c2);
                    float64x2_t b_1_high = vmulq_f64(sf1_high, c2);
                    float64x2_t b_3_low = vmulq_f64(sf3_low, c2);
                    float64x2_t b_3_high = vmulq_f64(sf3_high, c2);
                    
                    // Store results with scaling and bit reversal
                    vst1q_f64(&x[lut[0]].f[0], vmulq_f64(vaddq_f64(b_0_low, b_1_low), mul));
                    vst1q_f64(&x[lut[0]].f[2], vmulq_f64(vaddq_f64(b_0_high, b_1_high), mul));
                    vst1q_f64(&x[lut[1]].f[0], vmulq_f64(vsubq_f64(b_0_low, b_1_low), mul));
                    vst1q_f64(&x[lut[1]].f[2], vmulq_f64(vsubq_f64(b_0_high, b_1_high), mul));
                    vst1q_f64(&x[lut[2]].f[0], vmulq_f64(vaddq_f64(b_2_low, b_3_low), mul));
                    vst1q_f64(&x[lut[2]].f[2], vmulq_f64(vaddq_f64(b_2_high, b_3_high), mul));
                    vst1q_f64(&x[lut[3]].f[0], vmulq_f64(vsubq_f64(b_2_low, b_3_low), mul));
                    vst1q_f64(&x[lut[3]].f[2], vmulq_f64(vsubq_f64(b_2_high, b_3_high), mul));
                }
                
                // Process second 4 outputs
                {
                    float64x2_t sf4_low = vld1q_f64((double*)&sf2[4]);
                    float64x2_t sf4_high = vld1q_f64((double*)&sf2[4] + 2);
                    float64x2_t sf5_low = vld1q_f64((double*)&sf2[5]);
                    float64x2_t sf5_high = vld1q_f64((double*)&sf2[5] + 2);
                    float64x2_t sf6_low = vld1q_f64((double*)&sf2[6]);
                    float64x2_t sf6_high = vld1q_f64((double*)&sf2[6] + 2);
                    float64x2_t sf7_low = vld1q_f64((double*)&sf2[7]);
                    float64x2_t sf7_high = vld1q_f64((double*)&sf2[7] + 2);
                    
                    // Calculate butterfly values
                    float64x2_t b_0_low = vaddq_f64(sf4_low, sf6_low);
                    float64x2_t b_0_high = vaddq_f64(sf4_high, sf6_high);
                    float64x2_t b_2_low = vsubq_f64(sf4_low, sf6_low);
                    float64x2_t b_2_high = vsubq_f64(sf4_high, sf6_high);
                    float64x2_t b_1_low = vmulq_f64(sf5_low, c2);
                    float64x2_t b_1_high = vmulq_f64(sf5_high, c2);
                    float64x2_t b_3_low = vmulq_f64(sf7_low, c2);
                    float64x2_t b_3_high = vmulq_f64(sf7_high, c2);
                    
                    // Store results with scaling and bit reversal
                    vst1q_f64(&x[lut[4]].f[0], vmulq_f64(vaddq_f64(b_0_low, b_1_low), mul));
                    vst1q_f64(&x[lut[4]].f[2], vmulq_f64(vaddq_f64(b_0_high, b_1_high), mul));
                    vst1q_f64(&x[lut[5]].f[0], vmulq_f64(vsubq_f64(b_0_low, b_1_low), mul));
                    vst1q_f64(&x[lut[5]].f[2], vmulq_f64(vsubq_f64(b_0_high, b_1_high), mul));
                    vst1q_f64(&x[lut[6]].f[0], vmulq_f64(vaddq_f64(b_2_low, b_3_low), mul));
                    vst1q_f64(&x[lut[6]].f[2], vmulq_f64(vaddq_f64(b_2_high, b_3_high), mul));
                    vst1q_f64(&x[lut[7]].f[0], vmulq_f64(vsubq_f64(b_2_low, b_3_low), mul));
                    vst1q_f64(&x[lut[7]].f[2], vmulq_f64(vsubq_f64(b_2_high, b_3_high), mul));
                }
            }
        }
        // Special cases for small IFFTs
        else if (unlikely(_nbr_bits == 2)) {
            // 4-point IFFT
            const float64x2_t b_0_low = vaddq_f64(vld1q_f64((double*)&f[0]), vld1q_f64((double*)&f[2]));
            const float64x2_t b_0_high = vaddq_f64(vld1q_f64((double*)&f[0] + 2), vld1q_f64((double*)&f[2] + 2));
            const float64x2_t b_2_low = vsubq_f64(vld1q_f64((double*)&f[0]), vld1q_f64((double*)&f[2]));
            const float64x2_t b_2_high = vsubq_f64(vld1q_f64((double*)&f[0] + 2), vld1q_f64((double*)&f[2] + 2));
            
            const float64x2_t f1_c2_low = vmulq_f64(vld1q_f64((double*)&f[1]), c2);
            const float64x2_t f1_c2_high = vmulq_f64(vld1q_f64((double*)&f[1] + 2), c2);
            const float64x2_t f3_c2_low = vmulq_f64(vld1q_f64((double*)&f[3]), c2);
            const float64x2_t f3_c2_high = vmulq_f64(vld1q_f64((double*)&f[3] + 2), c2);
            
            vst1q_f64((double*)&x[0], vmulq_f64(vaddq_f64(b_0_low, f1_c2_low), mul));
            vst1q_f64((double*)&x[0] + 2, vmulq_f64(vaddq_f64(b_0_high, f1_c2_high), mul));
            vst1q_f64((double*)&x[2], vmulq_f64(vsubq_f64(b_0_low, f1_c2_low), mul));
            vst1q_f64((double*)&x[2] + 2, vmulq_f64(vsubq_f64(b_0_high, f1_c2_high), mul));
            vst1q_f64((double*)&x[1], vmulq_f64(vaddq_f64(b_2_low, f3_c2_low), mul));
            vst1q_f64((double*)&x[1] + 2, vmulq_f64(vaddq_f64(b_2_high, f3_c2_high), mul));
            vst1q_f64((double*)&x[3], vmulq_f64(vsubq_f64(b_2_low, f3_c2_low), mul));
            vst1q_f64((double*)&x[3] + 2, vmulq_f64(vsubq_f64(b_2_high, f3_c2_high), mul));
        }
        else if (unlikely(_nbr_bits == 1)) {
            // 2-point IFFT optimized for double4
            vst1q_f64((double*)&x[0], vmulq_f64(vaddq_f64(vld1q_f64((double*)&f[0]), vld1q_f64((double*)&f[1])), mul));
            vst1q_f64((double*)&x[0] + 2, vmulq_f64(vaddq_f64(vld1q_f64((double*)&f[0] + 2), vld1q_f64((double*)&f[1] + 2)), mul));
            vst1q_f64((double*)&x[1], vmulq_f64(vsubq_f64(vld1q_f64((double*)&f[0]), vld1q_f64((double*)&f[1])), mul));
            vst1q_f64((double*)&x[1] + 2, vmulq_f64(vsubq_f64(vld1q_f64((double*)&f[0] + 2), vld1q_f64((double*)&f[1] + 2)), mul));
        }
        else {
            // 1-point IFFT - simple scaling
            vst1q_f64((double*)&x[0], vmulq_f64(vld1q_f64((double*)&f[0]), mul));
            vst1q_f64((double*)&x[0] + 2, vmulq_f64(vld1q_f64((double*)&f[0] + 2), mul));
        }
    }
            
    void do_ifft_neon_f8(const simd_float8 *_Nonnull f, simd_float8 *_Nonnull x, bool do_scale = false)
    {
        const float32x4_t c2 = vdupq_n_f32(2.0f);
        
        // Initialize scaling factor
        float32x4_t mul = vdupq_n_f32(1.0f);
        if (unlikely(do_scale)) {
            mul = vdivq_f32(mul, vdupq_n_f32((float)_N));
        }

        if (likely(_nbr_bits > 2)) {
            simd_float8 *sf = (simd_float8*)f;
            simd_float8 *df;
            simd_float8 *df_temp;

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }

            // Optimized first pass with prefetching and software pipelining
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sfr = &sf[i];
                    auto sfi = &sfr[nbr_coef];
                    auto df1r = &df[i];
                    auto df2r = &df1r[nbr_coef];

                    // Prefetch next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                        __builtin_prefetch(&sf[i + d_nbr_coef + nbr_coef], 0, 3);
                    }

                    // Optimized extreme coefficient processing
                    float32x4_t sfr_0_low = vld1q_f32(&sfr[0].f[0]);
                    float32x4_t sfr_0_high = vld1q_f32(&sfr[0].f[4]);
                    float32x4_t sfr_nbr_coef_low = vld1q_f32(&sfr[nbr_coef].f[0]);
                    float32x4_t sfr_nbr_coef_high = vld1q_f32(&sfr[nbr_coef].f[4]);
                    
                    // Process and store immediately
                    vst1q_f32(&df1r[0].f[0], vaddq_f32(sfr_0_low, sfr_nbr_coef_low));
                    vst1q_f32(&df1r[0].f[4], vaddq_f32(sfr_0_high, sfr_nbr_coef_high));
                    vst1q_f32(&df2r[0].f[0], vsubq_f32(sfr_0_low, sfr_nbr_coef_low));
                    vst1q_f32(&df2r[0].f[4], vsubq_f32(sfr_0_high, sfr_nbr_coef_high));
                    
                    // Load and process h_nbr_coef values with c2 multiplication
                    float32x4_t sfr_h_nbr_coef_low = vld1q_f32(&sfr[h_nbr_coef].f[0]);
                    float32x4_t sfr_h_nbr_coef_high = vld1q_f32(&sfr[h_nbr_coef].f[4]);
                    float32x4_t sfi_h_nbr_coef_low = vld1q_f32(&sfi[h_nbr_coef].f[0]);
                    float32x4_t sfi_h_nbr_coef_high = vld1q_f32(&sfi[h_nbr_coef].f[4]);
                    
                    vst1q_f32(&df1r[h_nbr_coef].f[0], vmulq_f32(sfr_h_nbr_coef_low, c2));
                    vst1q_f32(&df1r[h_nbr_coef].f[4], vmulq_f32(sfr_h_nbr_coef_high, c2));
                    vst1q_f32(&df2r[h_nbr_coef].f[0], vmulq_f32(sfi_h_nbr_coef_low, c2));
                    vst1q_f32(&df2r[h_nbr_coef].f[4], vmulq_f32(sfi_h_nbr_coef_high, c2));

                    // Process conjugate complex numbers
                    auto df1i = &df1r[h_nbr_coef];
                    auto df2i = &df1i[nbr_coef];
                    
                    // Twiddle factor preloading
                    float c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        float c = c_next;
                        float s = s_next;
                        
                        // Prefetch for next iteration
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sfr[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[-(j + 1)], 0, 3);
                            __builtin_prefetch(&sfi[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[nbr_coef - (j + 1)], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Create vectorized twiddle factors
                        float32x4_t c_vec = vdupq_n_f32(c);
                        float32x4_t s_vec = vdupq_n_f32(s);

                        // Load values with grouping for cache efficiency
                        float32x4_t sfr_j_low = vld1q_f32(&sfr[j].f[0]);
                        float32x4_t sfr_j_high = vld1q_f32(&sfr[j].f[4]);
                        float32x4_t sfi_neg_j_low = vld1q_f32(&sfi[-j].f[0]);
                        float32x4_t sfi_neg_j_high = vld1q_f32(&sfi[-j].f[4]);
                        float32x4_t sfi_j_low = vld1q_f32(&sfi[j].f[0]);
                        float32x4_t sfi_j_high = vld1q_f32(&sfi[j].f[4]);
                        float32x4_t sfi_nbr_j_low = vld1q_f32(&sfi[nbr_coef - j].f[0]);
                        float32x4_t sfi_nbr_j_high = vld1q_f32(&sfi[nbr_coef - j].f[4]);

                        // Compute df1r and df1i values
                        vst1q_f32(&df1r[j].f[0], vaddq_f32(sfr_j_low, sfi_neg_j_low));
                        vst1q_f32(&df1r[j].f[4], vaddq_f32(sfr_j_high, sfi_neg_j_high));
                        vst1q_f32(&df1i[j].f[0], vsubq_f32(sfi_j_low, sfi_nbr_j_low));
                        vst1q_f32(&df1i[j].f[4], vsubq_f32(sfi_j_high, sfi_nbr_j_high));

                        // Calculate vr and vi
                        float32x4_t vr_low = vsubq_f32(sfr_j_low, sfi_neg_j_low);
                        float32x4_t vr_high = vsubq_f32(sfr_j_high, sfi_neg_j_high);
                        float32x4_t vi_low = vaddq_f32(sfi_j_low, sfi_nbr_j_low);
                        float32x4_t vi_high = vaddq_f32(sfi_j_high, sfi_nbr_j_high);

                        // Compute with optimized FMA operations
                        float32x4_t df2r_j_low = vfmaq_f32(vmulq_f32(vr_low, c_vec), vi_low, s_vec);
                        float32x4_t df2r_j_high = vfmaq_f32(vmulq_f32(vr_high, c_vec), vi_high, s_vec);
                        
                        float32x4_t df2i_j_low = vfmsq_f32(vmulq_f32(vi_low, c_vec), vr_low, s_vec);
                        float32x4_t df2i_j_high = vfmsq_f32(vmulq_f32(vi_high, c_vec), vr_high, s_vec);
                        
                        // Store with optimized memory access
                        vst1q_f32(&df2r[j].f[0], df2r_j_low);
                        vst1q_f32(&df2r[j].f[4], df2r_j_high);
                        vst1q_f32(&df2i[j].f[0], df2i_j_low);
                        vst1q_f32(&df2i[j].f[4], df2i_j_high);
                    }
                }

                // Prepare for next pass
                if (pass < _nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }

            // Antepenultimate pass optimization
            const float32x4_t sq2_2 = vdupq_n_f32(SQ2_2);
            
            for (auto i = 0; i < _N; i += 8) {
                auto df2 = &df[i];
                auto sf2 = &sf[i];
                
                // Prefetch for next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&sf[i + 8], 0, 3);
                }

                // Load data with interleaved prefetching and computation
                float32x4_t f0_low = vld1q_f32(&sf2[0].f[0]);
                float32x4_t f0_high = vld1q_f32(&sf2[0].f[4]);
                float32x4_t f4_low = vld1q_f32(&sf2[4].f[0]);
                float32x4_t f4_high = vld1q_f32(&sf2[4].f[4]);
                
                // Sum and difference computation
                float32x4_t sum04_low = vaddq_f32(f0_low, f4_low);
                float32x4_t sum04_high = vaddq_f32(f0_high, f4_high);
                float32x4_t diff04_low = vsubq_f32(f0_low, f4_low);
                float32x4_t diff04_high = vsubq_f32(f0_high, f4_high);
                
                // Load c2 multiplications
                float32x4_t f2_low = vld1q_f32(&sf2[2].f[0]);
                float32x4_t f2_high = vld1q_f32(&sf2[2].f[4]);
                float32x4_t f6_low = vld1q_f32(&sf2[6].f[0]);
                float32x4_t f6_high = vld1q_f32(&sf2[6].f[4]);
                
                // Store first set of results
                vst1q_f32(&df2[0].f[0], sum04_low);
                vst1q_f32(&df2[0].f[4], sum04_high);
                vst1q_f32(&df2[4].f[0], diff04_low);
                vst1q_f32(&df2[4].f[4], diff04_high);
                
                // Load additional data
                float32x4_t f1_low = vld1q_f32(&sf2[1].f[0]);
                float32x4_t f1_high = vld1q_f32(&sf2[1].f[4]);
                float32x4_t f3_low = vld1q_f32(&sf2[3].f[0]);
                float32x4_t f3_high = vld1q_f32(&sf2[3].f[4]);
                float32x4_t f5_low = vld1q_f32(&sf2[5].f[0]);
                float32x4_t f5_high = vld1q_f32(&sf2[5].f[4]);
                float32x4_t f7_low = vld1q_f32(&sf2[7].f[0]);
                float32x4_t f7_high = vld1q_f32(&sf2[7].f[4]);
                
                // Store c2 multiplications
                vst1q_f32(&df2[2].f[0], vmulq_f32(f2_low, c2));
                vst1q_f32(&df2[2].f[4], vmulq_f32(f2_high, c2));
                vst1q_f32(&df2[6].f[0], vmulq_f32(f6_low, c2));
                vst1q_f32(&df2[6].f[4], vmulq_f32(f6_high, c2));
                
                // Process sum and difference with optimized memory patterns
                vst1q_f32(&df2[1].f[0], vaddq_f32(f1_low, f3_low));
                vst1q_f32(&df2[1].f[4], vaddq_f32(f1_high, f3_high));
                vst1q_f32(&df2[3].f[0], vsubq_f32(f5_low, f7_low));
                vst1q_f32(&df2[3].f[4], vsubq_f32(f5_high, f7_high));
                
                // Compute vr and vi for SQ2_2 calculations
                float32x4_t vr_low = vsubq_f32(f1_low, f3_low);
                float32x4_t vr_high = vsubq_f32(f1_high, f3_high);
                float32x4_t vi_low = vaddq_f32(f5_low, f7_low);
                float32x4_t vi_high = vaddq_f32(f5_high, f7_high);
                
                // Optimized SQ2_2 multiplication and store
                vst1q_f32(&df2[5].f[0], vmulq_f32(vaddq_f32(vr_low, vi_low), sq2_2));
                vst1q_f32(&df2[5].f[4], vmulq_f32(vaddq_f32(vr_high, vi_high), sq2_2));
                vst1q_f32(&df2[7].f[0], vmulq_f32(vsubq_f32(vi_low, vr_low), sq2_2));
                vst1q_f32(&df2[7].f[4], vmulq_f32(vsubq_f32(vi_high, vr_high), sq2_2));
            }

            // Penultimate and last pass optimized with bit reversal
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
            for (auto i = 0; i < _N; i += 8) {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                // Prefetch output locations for better store performance
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&x[lut[8]], 1, 0);
                }

                // Process first 4 outputs with FMA optimizations
                {
                    float32x4_t sf0_low = vld1q_f32(&sf2[0].f[0]);
                    float32x4_t sf0_high = vld1q_f32(&sf2[0].f[4]);
                    float32x4_t sf1_low = vld1q_f32(&sf2[1].f[0]);
                    float32x4_t sf1_high = vld1q_f32(&sf2[1].f[4]);
                    float32x4_t sf2val_low = vld1q_f32(&sf2[2].f[0]);
                    float32x4_t sf2val_high = vld1q_f32(&sf2[2].f[4]);
                    float32x4_t sf3_low = vld1q_f32(&sf2[3].f[0]);
                    float32x4_t sf3_high = vld1q_f32(&sf2[3].f[4]);

                    // Calculate butterfly values using FMA operations
                    float32x4_t b_0_low = vaddq_f32(sf0_low, sf2val_low);
                    float32x4_t b_0_high = vaddq_f32(sf0_high, sf2val_high);
                    float32x4_t b_2_low = vsubq_f32(sf0_low, sf2val_low);
                    float32x4_t b_2_high = vsubq_f32(sf0_high, sf2val_high);
                    float32x4_t b_1_low = vmulq_f32(sf1_low, c2);
                    float32x4_t b_1_high = vmulq_f32(sf1_high, c2);
                    float32x4_t b_3_low = vmulq_f32(sf3_low, c2);
                    float32x4_t b_3_high = vmulq_f32(sf3_high, c2);

                    // Apply scaling and store with bit-reversal
                    vst1q_f32(&x[lut[0]].f[0], vmulq_f32(vaddq_f32(b_0_low, b_1_low), mul));
                    vst1q_f32(&x[lut[0]].f[4], vmulq_f32(vaddq_f32(b_0_high, b_1_high), mul));
                    vst1q_f32(&x[lut[1]].f[0], vmulq_f32(vsubq_f32(b_0_low, b_1_low), mul));
                    vst1q_f32(&x[lut[1]].f[4], vmulq_f32(vsubq_f32(b_0_high, b_1_high), mul));
                    vst1q_f32(&x[lut[2]].f[0], vmulq_f32(vaddq_f32(b_2_low, b_3_low), mul));
                    vst1q_f32(&x[lut[2]].f[4], vmulq_f32(vaddq_f32(b_2_high, b_3_high), mul));
                    vst1q_f32(&x[lut[3]].f[0], vmulq_f32(vsubq_f32(b_2_low, b_3_low), mul));
                    vst1q_f32(&x[lut[3]].f[4], vmulq_f32(vsubq_f32(b_2_high, b_3_high), mul));
                }

                // Process second 4 outputs with similar optimizations
                {
                    float32x4_t sf4_low = vld1q_f32(&sf2[4].f[0]);
                    float32x4_t sf4_high = vld1q_f32(&sf2[4].f[4]);
                    float32x4_t sf5_low = vld1q_f32(&sf2[5].f[0]);
                    float32x4_t sf5_high = vld1q_f32(&sf2[5].f[4]);
                    float32x4_t sf6_low = vld1q_f32(&sf2[6].f[0]);
                    float32x4_t sf6_high = vld1q_f32(&sf2[6].f[4]);
                    float32x4_t sf7_low = vld1q_f32(&sf2[7].f[0]);
                    float32x4_t sf7_high = vld1q_f32(&sf2[7].f[4]);

                    float32x4_t b_0_low = vaddq_f32(sf4_low, sf6_low);
                    float32x4_t b_0_high = vaddq_f32(sf4_high, sf6_high);
                    float32x4_t b_2_low = vsubq_f32(sf4_low, sf6_low);
                    float32x4_t b_2_high = vsubq_f32(sf4_high, sf6_high);
                    float32x4_t b_1_low = vmulq_f32(sf5_low, c2);
                    float32x4_t b_1_high = vmulq_f32(sf5_high, c2);
                    float32x4_t b_3_low = vmulq_f32(sf7_low, c2);
                    float32x4_t b_3_high = vmulq_f32(sf7_high, c2);

                    vst1q_f32(&x[lut[4]].f[0], vmulq_f32(vaddq_f32(b_0_low, b_1_low), mul));
                    vst1q_f32(&x[lut[4]].f[4], vmulq_f32(vaddq_f32(b_0_high, b_1_high), mul));
                    vst1q_f32(&x[lut[5]].f[0], vmulq_f32(vsubq_f32(b_0_low, b_1_low), mul));
                    vst1q_f32(&x[lut[5]].f[4], vmulq_f32(vsubq_f32(b_0_high, b_1_high), mul));
                    vst1q_f32(&x[lut[6]].f[0], vmulq_f32(vaddq_f32(b_2_low, b_3_low), mul));
                    vst1q_f32(&x[lut[6]].f[4], vmulq_f32(vaddq_f32(b_2_high, b_3_high), mul));
                    vst1q_f32(&x[lut[7]].f[0], vmulq_f32(vsubq_f32(b_2_low, b_3_low), mul));
                    vst1q_f32(&x[lut[7]].f[4], vmulq_f32(vsubq_f32(b_2_high, b_3_high), mul));
                }
            }
        }
        // Special cases for small IFFTs with optimized FMA
        else if (unlikely(_nbr_bits == 2)) {
            // 4-point IFFT optimized with vectorized operations
            const float32x4_t b_0_low = vaddq_f32(vld1q_f32(&f[0].f[0]), vld1q_f32(&f[2].f[0]));
            const float32x4_t b_0_high = vaddq_f32(vld1q_f32(&f[0].f[4]), vld1q_f32(&f[2].f[4]));
            const float32x4_t b_2_low = vsubq_f32(vld1q_f32(&f[0].f[0]), vld1q_f32(&f[2].f[0]));
            const float32x4_t b_2_high = vsubq_f32(vld1q_f32(&f[0].f[4]), vld1q_f32(&f[2].f[4]));
            
            const float32x4_t f1_c2_low = vmulq_f32(vld1q_f32(&f[1].f[0]), c2);
            const float32x4_t f1_c2_high = vmulq_f32(vld1q_f32(&f[1].f[4]), c2);
            const float32x4_t f3_c2_low = vmulq_f32(vld1q_f32(&f[3].f[0]), c2);
            const float32x4_t f3_c2_high = vmulq_f32(vld1q_f32(&f[3].f[4]), c2);
            
            vst1q_f32(&x[0].f[0], vmulq_f32(vaddq_f32(b_0_low, f1_c2_low), mul));
            vst1q_f32(&x[0].f[4], vmulq_f32(vaddq_f32(b_0_high, f1_c2_high), mul));
            vst1q_f32(&x[2].f[0], vmulq_f32(vsubq_f32(b_0_low, f1_c2_low), mul));
            vst1q_f32(&x[2].f[4], vmulq_f32(vsubq_f32(b_0_high, f1_c2_high), mul));
            vst1q_f32(&x[1].f[0], vmulq_f32(vaddq_f32(b_2_low, f3_c2_low), mul));
            vst1q_f32(&x[1].f[4], vmulq_f32(vaddq_f32(b_2_high, f3_c2_high), mul));
            vst1q_f32(&x[3].f[0], vmulq_f32(vsubq_f32(b_2_low, f3_c2_low), mul));
            vst1q_f32(&x[3].f[4], vmulq_f32(vsubq_f32(b_2_high, f3_c2_high), mul));
        }
        else if (unlikely(_nbr_bits == 1)) {
            // 2-point IFFT
            vst1q_f32(&x[0].f[0], vmulq_f32(vaddq_f32(vld1q_f32(&f[0].f[0]), vld1q_f32(&f[1].f[0])), mul));
            vst1q_f32(&x[0].f[4], vmulq_f32(vaddq_f32(vld1q_f32(&f[0].f[4]), vld1q_f32(&f[1].f[4])), mul));
            vst1q_f32(&x[1].f[0], vmulq_f32(vsubq_f32(vld1q_f32(&f[0].f[0]), vld1q_f32(&f[1].f[0])), mul));
            vst1q_f32(&x[1].f[4], vmulq_f32(vsubq_f32(vld1q_f32(&f[0].f[4]), vld1q_f32(&f[1].f[4])), mul));
        }
        else {
            // 1-point IFFT
            vst1q_f32(&x[0].f[0], vmulq_f32(vld1q_f32(&f[0].f[0]), mul));
            vst1q_f32(&x[0].f[4], vmulq_f32(vld1q_f32(&f[0].f[4]), mul));
        }
    }
 
    void do_ifft_neon_d8(const simd_double8 *_Nonnull f, simd_double8 *_Nonnull x, bool do_scale = false)
    {
        // Initialize constants
        const float64x2_t c2 = vdupq_n_f64(2.0);
        
        // Initialize scaling factor
        float64x2_t mul = vdupq_n_f64(1.0);
        if (unlikely(do_scale)) {
            mul = vdivq_f64(mul, vdupq_n_f64((double)_N));
        }

        if (likely(_nbr_bits > 2)) {
            simd_double8 *sf = (simd_double8*)f;
            simd_double8 *df;
            simd_double8 *df_temp;

            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }

            // First pass optimization with prefetching
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;

                for (auto i = 0; i < _N; i += d_nbr_coef) {
                    auto sfr = &sf[i];
                    auto sfi = &sfr[nbr_coef];
                    auto df1r = &df[i];
                    auto df2r = &df1r[nbr_coef];

                    // Prefetch next block
                    if (likely(i + d_nbr_coef < _N)) {
                        __builtin_prefetch(&sf[i + d_nbr_coef], 0, 3);
                        __builtin_prefetch(&sf[i + d_nbr_coef + nbr_coef], 0, 3);
                    }

                    // Optimized extreme coefficient processing
                    float64x2_t sfr_0_low = vld1q_f64(&sfr[0].f[0]);
                    float64x2_t sfr_0_high = vld1q_f64(&sfr[0].f[2]);
                    float64x2_t sfr_nbr_coef_low = vld1q_f64(&sfr[nbr_coef].f[0]);
                    float64x2_t sfr_nbr_coef_high = vld1q_f64(&sfr[nbr_coef].f[2]);
                    
                    // Process and store immediately
                    vst1q_f64(&df1r[0].f[0], vaddq_f64(sfr_0_low, sfr_nbr_coef_low));
                    vst1q_f64(&df1r[0].f[2], vaddq_f64(sfr_0_high, sfr_nbr_coef_high));
                    vst1q_f64(&df2r[0].f[0], vsubq_f64(sfr_0_low, sfr_nbr_coef_low));
                    vst1q_f64(&df2r[0].f[2], vsubq_f64(sfr_0_high, sfr_nbr_coef_high));
                    
                    // Load and process h_nbr_coef values with c2 multiplication
                    float64x2_t sfr_h_nbr_coef_low = vld1q_f64(&sfr[h_nbr_coef].f[0]);
                    float64x2_t sfr_h_nbr_coef_high = vld1q_f64(&sfr[h_nbr_coef].f[2]);
                    float64x2_t sfi_h_nbr_coef_low = vld1q_f64(&sfi[h_nbr_coef].f[0]);
                    float64x2_t sfi_h_nbr_coef_high = vld1q_f64(&sfi[h_nbr_coef].f[2]);
                    
                    vst1q_f64(&df1r[h_nbr_coef].f[0], vmulq_f64(sfr_h_nbr_coef_low, c2));
                    vst1q_f64(&df1r[h_nbr_coef].f[2], vmulq_f64(sfr_h_nbr_coef_high, c2));
                    vst1q_f64(&df2r[h_nbr_coef].f[0], vmulq_f64(sfi_h_nbr_coef_low, c2));
                    vst1q_f64(&df2r[h_nbr_coef].f[2], vmulq_f64(sfi_h_nbr_coef_high, c2));

                    // Process conjugate complex numbers
                    auto df1i = &df1r[h_nbr_coef];
                    auto df2i = &df1i[nbr_coef];
                    
                    // Twiddle factor preloading
                    double c_next, s_next;
                    if (h_nbr_coef > 1) {
                        _twiddle_cache.get_twiddle(pass, 1, c_next, s_next);
                    }

                    for (int j = 1; j < h_nbr_coef; ++j) {
                        // Use preloaded twiddle factors
                        double c = c_next;
                        double s = s_next;
                        
                        // Prefetch for next iteration
                        if (likely(j + 1 < h_nbr_coef)) {
                            __builtin_prefetch(&sfr[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[-(j + 1)], 0, 3);
                            __builtin_prefetch(&sfi[j + 1], 0, 3);
                            __builtin_prefetch(&sfi[nbr_coef - (j + 1)], 0, 3);
                            _twiddle_cache.get_twiddle(pass, j + 1, c_next, s_next);
                        }

                        // Create vectorized twiddle factors for doubles
                        float64x1_t cos_val = vdup_n_f64(c);
                        float64x1_t sin_val = vdup_n_f64(s);

                        // Load values with efficient memory access
                        float64x2_t sfr_j_low = vld1q_f64(&sfr[j].f[0]);
                        float64x2_t sfr_j_high = vld1q_f64(&sfr[j].f[2]);
                        float64x2_t sfi_neg_j_low = vld1q_f64(&sfi[-j].f[0]);
                        float64x2_t sfi_neg_j_high = vld1q_f64(&sfi[-j].f[2]);
                        float64x2_t sfi_j_low = vld1q_f64(&sfi[j].f[0]);
                        float64x2_t sfi_j_high = vld1q_f64(&sfi[j].f[2]);
                        float64x2_t sfi_nbr_j_low = vld1q_f64(&sfi[nbr_coef - j].f[0]);
                        float64x2_t sfi_nbr_j_high = vld1q_f64(&sfi[nbr_coef - j].f[2]);

                        // Compute df1r and df1i values
                        vst1q_f64(&df1r[j].f[0], vaddq_f64(sfr_j_low, sfi_neg_j_low));
                        vst1q_f64(&df1r[j].f[2], vaddq_f64(sfr_j_high, sfi_neg_j_high));
                        vst1q_f64(&df1i[j].f[0], vsubq_f64(sfi_j_low, sfi_nbr_j_low));
                        vst1q_f64(&df1i[j].f[2], vsubq_f64(sfi_j_high, sfi_nbr_j_high));

                        // Calculate vr and vi
                        float64x2_t vr_low = vsubq_f64(sfr_j_low, sfi_neg_j_low);
                        float64x2_t vr_high = vsubq_f64(sfr_j_high, sfi_neg_j_high);
                        float64x2_t vi_low = vaddq_f64(sfi_j_low, sfi_nbr_j_low);
                        float64x2_t vi_high = vaddq_f64(sfi_j_high, sfi_nbr_j_high);

                        // Compute with optimized FMA operations for doubles
                        float64x2_t df2r_j_low = vfmaq_lane_f64(vmulq_lane_f64(vr_low, cos_val, 0),
                                                                vi_low, sin_val, 0);
                        float64x2_t df2r_j_high = vfmaq_lane_f64(vmulq_lane_f64(vr_high, cos_val, 0),
                                                                 vi_high, sin_val, 0);
                        
                        float64x2_t df2i_j_low = vfmsq_lane_f64(vmulq_lane_f64(vi_low, cos_val, 0),
                                                                vr_low, sin_val, 0);
                        float64x2_t df2i_j_high = vfmsq_lane_f64(vmulq_lane_f64(vi_high, cos_val, 0),
                                                                 vr_high, sin_val, 0);
                        
                        // Store with optimized memory access
                        vst1q_f64(&df2r[j].f[0], df2r_j_low);
                        vst1q_f64(&df2r[j].f[2], df2r_j_high);
                        vst1q_f64(&df2i[j].f[0], df2i_j_low);
                        vst1q_f64(&df2i[j].f[2], df2i_j_high);
                    }
                }

                // Prepare for next pass
                if (pass < _nbr_bits - 1) {
                    auto tmp = df;
                    df = sf;
                    sf = tmp;
                } else {
                    sf = df;
                    df = df_temp;
                }
            }

            // Antepenultimate pass with optimized SQ2_2 calculations
            const float64x1_t sq2_2 = vdup_n_f64(SQ2_2);
            
            for (auto i = 0; i < _N; i += 8) {
                auto df2 = &df[i];
                auto sf2 = &sf[i];
                
                // Prefetch next block
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&sf[i + 8], 0, 3);
                }

                // Load data with interleaved memory access
                float64x2_t f0_low = vld1q_f64(&sf2[0].f[0]);
                float64x2_t f0_high = vld1q_f64(&sf2[0].f[2]);
                float64x2_t f4_low = vld1q_f64(&sf2[4].f[0]);
                float64x2_t f4_high = vld1q_f64(&sf2[4].f[2]);
                
                // Sum and difference computation - store immediately to free registers
                float64x2_t sum04_low = vaddq_f64(f0_low, f4_low);
                float64x2_t sum04_high = vaddq_f64(f0_high, f4_high);
                float64x2_t diff04_low = vsubq_f64(f0_low, f4_low);
                float64x2_t diff04_high = vsubq_f64(f0_high, f4_high);
                
                vst1q_f64(&df2[0].f[0], sum04_low);
                vst1q_f64(&df2[0].f[2], sum04_high);
                vst1q_f64(&df2[4].f[0], diff04_low);
                vst1q_f64(&df2[4].f[2], diff04_high);
                
                // Load more data
                float64x2_t f2_low = vld1q_f64(&sf2[2].f[0]);
                float64x2_t f2_high = vld1q_f64(&sf2[2].f[2]);
                float64x2_t f6_low = vld1q_f64(&sf2[6].f[0]);
                float64x2_t f6_high = vld1q_f64(&sf2[6].f[2]);
                float64x2_t f1_low = vld1q_f64(&sf2[1].f[0]);
                float64x2_t f1_high = vld1q_f64(&sf2[1].f[2]);
                float64x2_t f3_low = vld1q_f64(&sf2[3].f[0]);
                float64x2_t f3_high = vld1q_f64(&sf2[3].f[2]);
                
                // Compute c2 multiplications and store
                vst1q_f64(&df2[2].f[0], vmulq_f64(f2_low, c2));
                vst1q_f64(&df2[2].f[2], vmulq_f64(f2_high, c2));
                vst1q_f64(&df2[6].f[0], vmulq_f64(f6_low, c2));
                vst1q_f64(&df2[6].f[2], vmulq_f64(f6_high, c2));
                
                // Load remaining data
                float64x2_t f5_low = vld1q_f64(&sf2[5].f[0]);
                float64x2_t f5_high = vld1q_f64(&sf2[5].f[2]);
                float64x2_t f7_low = vld1q_f64(&sf2[7].f[0]);
                float64x2_t f7_high = vld1q_f64(&sf2[7].f[2]);
                
                // Process simple additions and store
                vst1q_f64(&df2[1].f[0], vaddq_f64(f1_low, f3_low));
                vst1q_f64(&df2[1].f[2], vaddq_f64(f1_high, f3_high));
                vst1q_f64(&df2[3].f[0], vsubq_f64(f5_low, f7_low));
                vst1q_f64(&df2[3].f[2], vsubq_f64(f5_high, f7_high));
                
                // Compute vr and vi for SQ2_2 calculations
                float64x2_t vr_low = vsubq_f64(f1_low, f3_low);
                float64x2_t vr_high = vsubq_f64(f1_high, f3_high);
                float64x2_t vi_low = vaddq_f64(f5_low, f7_low);
                float64x2_t vi_high = vaddq_f64(f5_high, f7_high);
                
                // Optimized SQ2_2 multiplication with FMA
                vst1q_f64(&df2[5].f[0], vmulq_lane_f64(vaddq_f64(vr_low, vi_low), sq2_2, 0));
                vst1q_f64(&df2[5].f[2], vmulq_lane_f64(vaddq_f64(vr_high, vi_high), sq2_2, 0));
                vst1q_f64(&df2[7].f[0], vmulq_lane_f64(vsubq_f64(vi_low, vr_low), sq2_2, 0));
                vst1q_f64(&df2[7].f[2], vmulq_lane_f64(vsubq_f64(vi_high, vr_high), sq2_2, 0));
            }

            // Penultimate and last pass with optimized bit reversal
            auto lut_ptr = _bit_rev_lut.get_ptr();
            
            for (auto i = 0; i < _N; i += 8) {
                auto lut = lut_ptr + i;
                auto sf2 = &df[i];
                
                // Prefetch output locations for reduced store latency
                if (likely(i + 8 < _N)) {
                    __builtin_prefetch(&x[lut[8]], 1, 0);
                    __builtin_prefetch(&x[lut[9]], 1, 0);
                }

                // Process first 4 outputs with optimized butterfly
                {
                    float64x2_t sf0_low = vld1q_f64(&sf2[0].f[0]);
                    float64x2_t sf0_high = vld1q_f64(&sf2[0].f[2]);
                    float64x2_t sf1_low = vld1q_f64(&sf2[1].f[0]);
                    float64x2_t sf1_high = vld1q_f64(&sf2[1].f[2]);
                    float64x2_t sf2val_low = vld1q_f64(&sf2[2].f[0]);
                    float64x2_t sf2val_high = vld1q_f64(&sf2[2].f[2]);
                    float64x2_t sf3_low = vld1q_f64(&sf2[3].f[0]);
                    float64x2_t sf3_high = vld1q_f64(&sf2[3].f[2]);

                    // Calculate butterfly values with SIMD
                    float64x2_t b_0_low = vaddq_f64(sf0_low, sf2val_low);
                    float64x2_t b_0_high = vaddq_f64(sf0_high, sf2val_high);
                    float64x2_t b_2_low = vsubq_f64(sf0_low, sf2val_low);
                    float64x2_t b_2_high = vsubq_f64(sf0_high, sf2val_high);
                    float64x2_t b_1_low = vmulq_f64(sf1_low, c2);
                    float64x2_t b_1_high = vmulq_f64(sf1_high, c2);
                    float64x2_t b_3_low = vmulq_f64(sf3_low, c2);
                    float64x2_t b_3_high = vmulq_f64(sf3_high, c2);

                    // Apply scaling and store with bit-reversal
                    vst1q_f64(&x[lut[0]].f[0], vmulq_f64(vaddq_f64(b_0_low, b_1_low), mul));
                    vst1q_f64(&x[lut[0]].f[2], vmulq_f64(vaddq_f64(b_0_high, b_1_high), mul));
                    vst1q_f64(&x[lut[1]].f[0], vmulq_f64(vsubq_f64(b_0_low, b_1_low), mul));
                    vst1q_f64(&x[lut[1]].f[2], vmulq_f64(vsubq_f64(b_0_high, b_1_high), mul));
                    vst1q_f64(&x[lut[2]].f[0], vmulq_f64(vaddq_f64(b_2_low, b_3_low), mul));
                    vst1q_f64(&x[lut[2]].f[2], vmulq_f64(vaddq_f64(b_2_high, b_3_high), mul));
                    vst1q_f64(&x[lut[3]].f[0], vmulq_f64(vsubq_f64(b_2_low, b_3_low), mul));
                    vst1q_f64(&x[lut[3]].f[2], vmulq_f64(vsubq_f64(b_2_high, b_3_high), mul));
                }

                // Process second 4 outputs with similar optimization
                {
                    float64x2_t sf4_low = vld1q_f64(&sf2[4].f[0]);
                    float64x2_t sf4_high = vld1q_f64(&sf2[4].f[2]);
                    float64x2_t sf5_low = vld1q_f64(&sf2[5].f[0]);
                    float64x2_t sf5_high = vld1q_f64(&sf2[5].f[2]);
                    float64x2_t sf6_low = vld1q_f64(&sf2[6].f[0]);
                    float64x2_t sf6_high = vld1q_f64(&sf2[6].f[2]);
                    float64x2_t sf7_low = vld1q_f64(&sf2[7].f[0]);
                    float64x2_t sf7_high = vld1q_f64(&sf2[7].f[2]);

                    float64x2_t b_0_low = vaddq_f64(sf4_low, sf6_low);
                    float64x2_t b_0_high = vaddq_f64(sf4_high, sf6_high);
                    float64x2_t b_2_low = vsubq_f64(sf4_low, sf6_low);
                    float64x2_t b_2_high = vsubq_f64(sf4_high, sf6_high);
                    float64x2_t b_1_low = vmulq_f64(sf5_low, c2);
                    float64x2_t b_1_high = vmulq_f64(sf5_high, c2);
                    float64x2_t b_3_low = vmulq_f64(sf7_low, c2);
                    float64x2_t b_3_high = vmulq_f64(sf7_high, c2);

                    vst1q_f64(&x[lut[4]].f[0], vmulq_f64(vaddq_f64(b_0_low, b_1_low), mul));
                    vst1q_f64(&x[lut[4]].f[2], vmulq_f64(vaddq_f64(b_0_high, b_1_high), mul));
                    vst1q_f64(&x[lut[5]].f[0], vmulq_f64(vsubq_f64(b_0_low, b_1_low), mul));
                    vst1q_f64(&x[lut[5]].f[2], vmulq_f64(vsubq_f64(b_0_high, b_1_high), mul));
                    vst1q_f64(&x[lut[6]].f[0], vmulq_f64(vaddq_f64(b_2_low, b_3_low), mul));
                    vst1q_f64(&x[lut[6]].f[2], vmulq_f64(vaddq_f64(b_2_high, b_3_high), mul));
                    vst1q_f64(&x[lut[7]].f[0], vmulq_f64(vsubq_f64(b_2_low, b_3_low), mul));
                    vst1q_f64(&x[lut[7]].f[2], vmulq_f64(vsubq_f64(b_2_high, b_3_high), mul));
                }
            }
        }
        // Special cases for small IFFTs optimized with vectorized operations
        else if (unlikely(_nbr_bits == 2)) {
            // 4-point IFFT
            const float64x2_t b_0_low = vaddq_f64(vld1q_f64(&f[0].f[0]), vld1q_f64(&f[2].f[0]));
            const float64x2_t b_0_high = vaddq_f64(vld1q_f64(&f[0].f[2]), vld1q_f64(&f[2].f[2]));
            const float64x2_t b_2_low = vsubq_f64(vld1q_f64(&f[0].f[0]), vld1q_f64(&f[2].f[0]));
            const float64x2_t b_2_high = vsubq_f64(vld1q_f64(&f[0].f[2]), vld1q_f64(&f[2].f[2]));
            
            const float64x2_t f1_c2_low = vmulq_f64(vld1q_f64(&f[1].f[0]), c2);
            const float64x2_t f1_c2_high = vmulq_f64(vld1q_f64(&f[1].f[2]), c2);
            const float64x2_t f3_c2_low = vmulq_f64(vld1q_f64(&f[3].f[0]), c2);
            const float64x2_t f3_c2_high = vmulq_f64(vld1q_f64(&f[3].f[2]), c2);
            
            vst1q_f64(&x[0].f[0], vmulq_f64(vaddq_f64(b_0_low, f1_c2_low), mul));
            vst1q_f64(&x[0].f[2], vmulq_f64(vaddq_f64(b_0_high, f1_c2_high), mul));
            vst1q_f64(&x[2].f[0], vmulq_f64(vsubq_f64(b_0_low, f1_c2_low), mul));
            vst1q_f64(&x[2].f[2], vmulq_f64(vsubq_f64(b_0_high, f1_c2_high), mul));
            vst1q_f64(&x[1].f[0], vmulq_f64(vaddq_f64(b_2_low, f3_c2_low), mul));
            vst1q_f64(&x[1].f[2], vmulq_f64(vaddq_f64(b_2_high, f3_c2_high), mul));
            vst1q_f64(&x[3].f[0], vmulq_f64(vsubq_f64(b_2_low, f3_c2_low), mul));
            vst1q_f64(&x[3].f[2], vmulq_f64(vsubq_f64(b_2_high, f3_c2_high), mul));
        }
        else if (unlikely(_nbr_bits == 1)) {
            // 2-point IFFT optimized with vectorized add/sub
            vst1q_f64(&x[0].f[0], vmulq_f64(vaddq_f64(vld1q_f64(&f[0].f[0]), vld1q_f64(&f[1].f[0])), mul));
            vst1q_f64(&x[0].f[2], vmulq_f64(vaddq_f64(vld1q_f64(&f[0].f[2]), vld1q_f64(&f[1].f[2])), mul));
            vst1q_f64(&x[1].f[0], vmulq_f64(vsubq_f64(vld1q_f64(&f[0].f[0]), vld1q_f64(&f[1].f[0])), mul));
            vst1q_f64(&x[1].f[2], vmulq_f64(vsubq_f64(vld1q_f64(&f[0].f[2]), vld1q_f64(&f[1].f[2])), mul));
        }
        else {
            // 1-point IFFT - simple multiplication with scaling
            vst1q_f64(&x[0].f[0], vmulq_f64(vld1q_f64(&f[0].f[0]), mul));
            vst1q_f64(&x[0].f[2], vmulq_f64(vld1q_f64(&f[0].f[2]), mul));
        }
    }
#else // NEW_NEON_OPT
    
    void do_fft_neon_d8(const simd_double8 *x, simd_double8 *f)
    {
        if (_nbr_bits > 2) {
            simd_double8 *sf, *df;
            
            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }
            
            // First stage: bit-reversal and initial butterfly computation
            auto bit_rev_lut_ptr = _bit_rev_lut.get_ptr();
            
            for (int i = 0; i < _N; i += 4) {
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
            
            for (auto i = 0; i < _N; i += 8) {
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
            
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                const double *cos_ptr = _trigo_lut.get_ptr(pass);
                for (auto i = 0; i < _N; i += d_nbr_coef) {
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
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_double8 b_0 = x[0] + x[2];
            const simd_double8 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        // 2-point FFT
        else if (_nbr_bits == 1) {
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
        if (_nbr_bits > 2) {
            simd_float8 *sf, *df;
            
            if (_nbr_bits & 1) {
                df = buffer_ptr;
                sf = f;
            } else {
                df = f;
                sf = buffer_ptr;
            }
            
            // First stage: bit-reversal and initial butterfly computation
            auto bit_rev_lut_ptr = _bit_rev_lut.get_ptr();
            
            for (int i = 0; i < _N; i += 4) {
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
            
            for (auto i = 0; i < _N; i += 8) {
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
            
            for (auto pass = 3; pass < _nbr_bits; ++pass) {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                
                const auto cos_ptr = _trigo_lut.get_ptr(pass);
                for (auto i = 0; i < _N; i += d_nbr_coef) {
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
        else if (_nbr_bits == 2) {
            f[1] = x[0] - x[2];
            f[3] = x[1] - x[3];
            
            const simd_float8 b_0 = x[0] + x[2];
            const simd_float8 b_2 = x[1] + x[3];
            
            f[0] = b_0 + b_2;
            f[2] = b_0 - b_2;
        }
        // 2-point FFT
        else if (_nbr_bits == 1) {
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
        
        if (_nbr_bits > 2)
        {
            simd_double8 * sf = (T*) f;
            simd_double8 * df;
            simd_double8 * df_temp;
            
            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }
            
            // Do the transformation in several pass
            
            // First pass
            
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass)
            {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                auto cos_ptr = _trigo_lut.get_ptr (pass);
                
                for (auto i = 0; i < _N; i += d_nbr_coef)
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
                
                if (pass < _nbr_bits - 1) {
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
            
            for (auto i = 0; i < _N; i += 8)
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
            
            for (auto i = 0; i < _N; i += 8)
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
        else if (_nbr_bits == 2) {
            auto b_0 = f[0] + f[2];
            auto b_2 = f[0] - f[2];
            
            x[0] = b_0 + f[1] * c2;
            x[2] = b_0 - f[1] * c2;
            x[1] = b_2 + f[3] * c2;
            x[3] = b_2 - f[3] * c2;
        }
        // 2-point IFFT
        else if (_nbr_bits == 1) {
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
        
        if (_nbr_bits > 2)
        {
            simd_float8 * sf = (T*) f;
            simd_float8 * df;
            simd_float8 * df_temp;
            
            if (_nbr_bits & 1) {
                df = buffer_ptr;
                df_temp = x;
            } else {
                df = x;
                df_temp = buffer_ptr;
            }
            
            // Do the transformation in several pass
            
            // First pass
            
            for (auto pass = _nbr_bits - 1; pass >= 3; --pass)
            {
                auto nbr_coef = 1 << pass;
                auto h_nbr_coef = nbr_coef >> 1;
                auto d_nbr_coef = nbr_coef << 1;
                auto cos_ptr = _trigo_lut.get_ptr (pass);
                
                for (auto i = 0; i < _N; i += d_nbr_coef)
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
                
                if (pass < _nbr_bits - 1) {
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
            
            for (auto i = 0; i < _N; i += 8)
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
            
            
            for (auto i = 0; i < _N; i += 8)
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
        else if (_nbr_bits == 2) {
            auto b_0 = f[0] + f[2];
            auto b_2 = f[0] - f[2];
            
            x[0] = b_0 + f[1] * c2;
            x[2] = b_0 - f[1] * c2;
            x[1] = b_2 + f[3] * c2;
            x[3] = b_2 - f[3] * c2;
        }
        // 2-point IFFT
        else if (_nbr_bits == 1) {
            x[0] = f[0] + f[1];
            x[1] = f[0] - f[1];
        }
        // 1-point IFFT
        else {
            x[0] = f[0];
        }
    }
#endif
    
#endif // USE_NEON
};



template<typename T, int N>
class FFTRealHybrid {
public:
    // --- Type aliases ---
    using T1 = SimdBase<T>;        // scalar base type
    using cmplxTT = cmplxT<T>;     // complex type matching T

    // Select the largest backend SIMD type for performance
    static constexpr auto select_backend() {
        if constexpr (std::is_same_v<T,float>) {
            // Scalar float → 8-lane SIMD for max performance
            return simd_float8{};
        } else if constexpr (std::is_same_v<T,simd_float2> ||
                             std::is_same_v<T,simd_float4> ||
                             std::is_same_v<T,simd_float8>) {
            // Smaller float vectors → upcast to 8-lane SIMD
            return simd_float8{};
        } else if constexpr (std::is_same_v<T,double>) {
            // Scalar double → 4-lane SIMD (double is 2x float size)
            return simd_double4{};
        } else if constexpr (std::is_same_v<T,simd_double2> ||
                             std::is_same_v<T,simd_double4>) {
            // Smaller double vectors → upcast to 4-lane SIMD
            return simd_double4{};
        } else if constexpr (std::is_same_v<T,simd_double8>) {
            // Already maxed out for double
            return simd_double8{};
        } else {
            static_assert(sizeof(T)==0, "Unsupported type for FFTRealHybrid backend");
        }
    }

    using BT = decltype(select_backend());   // backend type
    using cmplxBT = cmplxT<BT>;             // complex type of backend

    // --- Constructor ---
    FFTRealHybrid() {
        simd_size = sizeof(BT)/sizeof(T1);
        simd_N = N / simd_size;
        X.resize(simd_N);
        Y.resize(simd_N);
    }

    // --- Public API ---
    void real_fft(const T* in, cmplxTT* out, bool do_scale = false) {
        pack_input(in);
        backend.real_fft(X.data(), Y.data(), do_scale);
        unpack_output(out);
    }

    void real_ifft(const cmplxTT* in, T* out, bool do_scale = false) {
        pack_spectrum(in);
        backend.inverse(Y.data(), X.data(), do_scale);
        unpack_output(out);
    }

    void reset() { backend.reset(); }

private:
    int simd_size;         // number of lanes in backend vector
    int simd_N;            // number of backend vectors to hold N elements
    std::vector<BT> X, Y;  // input/output buffers
    FFTReal<BT> backend;   // backend FFT

    // --- Packing / Unpacking ---
    void pack_input(const T* in) {
        for(int i=0; i<simd_N; ++i) {
            BT v{};
            for(int j=0; j<simd_size; ++j) {
                int idx = i*simd_size + j;
                if(idx < N) v_insert(v, j, in[idx]);
            }
            X[i] = v;
        }
    }

    void unpack_output(cmplxTT* out){
        for(int i=0;i<simd_N;i++){
            BT v = Y[i];
            for(int j=0;j<simd_size;j++){
                int idx = i*simd_size + j;
                if(idx < N) {
                    out[idx].re = v_extract(v,j);
                    out[idx].im = T1(0);
                }
            }
        }
    }

    void pack_spectrum(const cmplxTT* in){
        for(int i=0;i<simd_N;i++){
            BT v{};
            for(int j=0;j<simd_size;j++){
                int idx = i*simd_size + j;
                if(idx < N) v_insert(v,j,in[idx].re);
            }
            Y[i] = v;
        }
    }

    // --- Low-level SIMD helpers ---
    template<typename V>
    void v_insert(BT &v,int lane,const V &val){
        auto tmp = convertvector_safe<BT>(val);
        int lanes = std::min(SimdSize<BT>, simd_size - lane);
        for(int k=0;k<lanes;k++){
            v[k+lane] = tmp[k];
        }
    }

    T1 v_extract(const BT &v,int lane) const {
        return v[lane];
    }
};

