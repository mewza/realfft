//  realfft.h - A highly optimized C++ SIMD vector templated class
//  ---
//  FFTReal v1.2 (C) 2025 Dmitry Boldyrev <subband@gmail.com>
//  Pascal version (C) 2024 Laurent de Soras <ldesoras@club-internet.fr>
//  Object Pascal port (C) 2024 Frederic Vanmol <frederic@fruityloops.com>
//
//  NOTE: I have a highly hand optimizations for neon that I made that I offer
//        for sale $200/app or project, contact me on the email above directly if interested
//        the current optimizations are only give you surface level of what is possible

#pragma once

#include <memory>
#include <mss/const1.h>

#if TARGET_OS_MACCATALYST && TARGET_CPU_ARM64
#ifdef USE_NEON
#include <simd/simd.h>
#endif
#else
#undef USE_NEON
#endif

// #define NEW_NEON_OPT

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
    T* buffer_ptr = nullptr;
    T* yy = nullptr;
    T* xx = nullptr;
    
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
#ifdef USE_NEON
        if constexpr( std::is_same_v<T, simd_double8> ) {
            do_fft_neon_d8(xx, yy);
        } else if constexpr( std::is_same_v<T, simd_float8> )
            do_fft_neon_f8(xx, yy);
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
    
    void real_ifft(const cmplxT<T>* x, T* y, bool do_scale = false)
    {
        for (int i=1; i < _N2; i++) {
            yy[ i       ] = x[i].re;
            yy[ i + _N2 ] = x[i].im;
        }
        yy[   0 ] = x[0].re;
        yy[ _N2 ] = 0.0;
        
#ifdef USE_NEON
        if constexpr( std::is_same_v<T, simd_double8> ) {
            do_ifft_neon_d8(yy, y, do_scale);
        } else if constexpr( std::is_same_v<T, simd_float8> )
            do_ifft_neon_f8(yy, y, do_scale);
        else
            do_ifft(yy, y, do_scale);
#else
        do_ifft(yy, y, do_scale);
#endif
    }
    
    void do_fft(const T *x, T *f)
    {
        T c, s;
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
            
            auto lut_ptr = _bit_rev_lut->get_ptr();
            for (auto i = 0; i < _N; i += 4)
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
    
    void do_ifft(const T *f, T *x, bool do_scale = false)
    {
        T c, s;
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
    
    inline void do_rescale(T *x) const {
        const T1 mul = 1./(T1)_N;
        for (auto i = 0; i < _N; ++i)
            x[i] *= mul;
    }
    
    inline void do_rescale(cmplxT<T> *x) const {
        const T mul = 1./(T1)_N;
        x[0] = cmplxT<T>(x[0].re, 0.0) * mul;
        for (auto i = 1; i < _N2; ++i)
            x[i] *= mul;
    }
    
    template <typename U>
    static U* alignedAlloc(size_t size) {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(size * sizeof(U), 64);
#else
        posix_memalign(&ptr, 64, size * sizeof(U));
#endif
        return static_cast<U*>(ptr);
    }
    
    // Helper function for aligned deallocation
    static void alignedFree(void* ptr) {
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
        int* offsets = nullptr;
        T1* cos_data = nullptr;
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
        inline const T1* get_ptr(int pass) const {
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
        int* indices = nullptr;
        int N = 0;
        int nbr_bits = 0;
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
        inline const int* get_ptr() const {
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
        T1** cos_data = nullptr;
        T1** sin_data = nullptr;
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
        
        inline void get_twiddle(int pass, int j, T& cos_val, T& sin_val) const
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

#ifdef USE_NEON
    
#ifdef NEW_NEON_OPT
   // highly optimized routines go here
#else // NEW_NEON_OPT
    
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
#endif
    
#endif // USE_NEON
};
