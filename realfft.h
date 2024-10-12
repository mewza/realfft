//  realfft.h - A highly optimized C++ SIMD vector templated class
//  --- 
//  FFTReal v1.0 (C) 2024 Dmitry Boldyrev <subband@gmail.com>
//  (C) 2024 Laurent de Soras <ldesoras@club-internet.fr>
//  Object Pascal port (C) 2024 Frederic Vanmol <frederic@fruityloops.com>

#pragma once

template <typename T, unsigned N>
class FFTReal
{
    using T1 = SimdBase<T>;
    
    static constexpr int floorlog2(unsigned x) {
        return (x == 1) ? 0 : 1 + floorlog2(x >> 1);
    }
    
    static const int nbr_bits = floorlog2( N );
    static const int N2 = N / 2;
    
protected:
    
    T   buffer_ptr[ N ];
    T   pt[ N + 2 ];
    
public:
    
    inline void real_fft(const T* x, cmplxT<T>* y, bool do_scale = false)
    {
        T * ppt = pt;
        
        do_fft(x, ppt);
        
        for (int i=1; i < N2; i++)
            y[i] = cmplxT<T>(ppt[i], ppt[i+N2]);
        
        y[0] = cmplxT<T>(ppt[0], 0.0);
        
        if (do_scale)
            do_rescale(y, N2);
    }
    
    inline void real_ifft(const cmplxT<T>* x, T* y, bool do_scale = false)
    {
        T * ppt = pt;
        
        for (int i=1; i < N2; i++) {
            ppt[i   ] = x[i].re;
            ppt[i+N2] = x[i].im;
        }
        ppt[0 ] = x[0].re;
        ppt[N2] = 0.0;
        
        do_ifft(ppt, y);
        
        if (do_scale)
            do_rescale(y, N);
    }
    
private:
    
    // ========================================================================== //
    //      Description: Scale an array by divide each element by its N.          //
    //                   This function should be called after FFT + IFFT.         //
    //      Input/Output parameters:                                              //
    //        - x: pointer on array to rescale (time or frequency).               //
    // ========================================================================== //
    
    inline void do_rescale(T *x, int len) const
    {
        const T1 mul = 1./(T1)N;
        for (int i=0; i < len; ++i)
            x[i] *= mul;
    }
    
    inline void do_rescale(cmplxT<T> *x, int len) const
    {
        const T1 mul = 1./(T1)N;
        for (int i=0; i < len; ++i)
            x[i] *= mul;
    }
    
    // ========================================================================== //
    //      Description: Compute the FFT of the array.                            //
    //      Input parameters:                                                     //
    //        - x: pointer on the source array (time).                            //
    //      Output parameters:                                                    //
    //        - f: pointer on the destination array (frequencies).                //
    //             f [0...N(x)/2] = real values,                                  //
    //             f [N(x)/2+1...N(x)-1] = imaginary values of                    //
    //               coefficents 1...N(x)/2-1.                                    //
    // ========================================================================== //

    void do_fft(const T *x, T *f)
    {
        int pass, nbr_coef, h_nbr_coef, d_nbr_coef, coef_index;
        
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
            
            //  Do the transformation in several pass
            
            //  First and second pass at once
            
            coef_index = 0;
            auto bit_rev_lut_ptr = _bit_rev_lut.get_ptr();
            do {
                auto df2 = &df [coef_index];
                auto x0 = x[ bit_rev_lut_ptr[coef_index + 0] ];
                auto x1 = x[ bit_rev_lut_ptr[coef_index + 1] ];
                auto x2 = x[ bit_rev_lut_ptr[coef_index + 2] ];
                auto x3 = x[ bit_rev_lut_ptr[coef_index + 3] ];
                
                df2[0] = x0 + x1 + x2 + x3;
                df2[1] = x0 - x1;
                df2[2] = x0 + x1 - x2 - x3;
                df2[3] = x2 - x3;
                
                coef_index += 4;
            } while (coef_index < N);
            
            
            //  Third pass
            
            coef_index = 0;
            do {
                sf [coef_index] = df [coef_index] + df [coef_index + 4];
                sf [coef_index + 4] = df [coef_index] - df [coef_index + 4];
                sf [coef_index + 2] = df [coef_index + 2];
                sf [coef_index + 6] = df [coef_index + 6];
                
                T v = (df [coef_index + 5] - df [coef_index + 7]) * SQ2_2;
                sf [coef_index + 1] = df [coef_index + 1] + v;
                sf [coef_index + 3] = df [coef_index + 1] - v;
                
                v = (df [coef_index + 5] + df [coef_index + 7]) * SQ2_2;
                sf [coef_index + 5] = v + df [coef_index + 3];
                sf [coef_index + 7] = v - df [coef_index + 3];
                
                coef_index += 8;
            } while (coef_index < N);
            
            //  Next pass
            
            for (pass = 3; pass < nbr_bits; ++pass)
            {
                coef_index = 0;
                nbr_coef = 1 << pass;
                h_nbr_coef = nbr_coef >> 1;
                d_nbr_coef = nbr_coef << 1;
                
                auto cos_ptr = _trigo_lut.get_ptr(pass);
                do {
                    const T * const sf1r = sf + coef_index;
                    const T * const sf2r = sf1r + nbr_coef;
                    T * const dfr = df + coef_index;
                    T * const dfi = dfr + nbr_coef;
                    
                    //  Extreme coefficients are always real
                    
                    dfr [0] = sf1r [0] + sf2r [0];
                    dfi [0] = sf1r [0] - sf2r [0];              // dfr [nbr_coef] =
                    dfr [h_nbr_coef] = sf1r [h_nbr_coef];
                    dfi [h_nbr_coef] = sf2r [h_nbr_coef];
                    
                    //  Others are conjugate complex numbers
                    
                    const T * const sf1i = &sf1r [h_nbr_coef];
                    const T * const sf2i = &sf1i [nbr_coef];
                    for (int i = 1; i < h_nbr_coef; ++i)
                    {
                        const T c = cos_ptr [i];                // cos (i*PI/nbr_coef);
                        const T s = cos_ptr [h_nbr_coef - i];   // sin (i*PI/nbr_coef);
                        
                        T v = sf2r [i] * c - sf2i [i] * s;
                        dfr [ i] = sf1r [i] + v;
                        dfi [-i] = sf1r [i] - v;                // dfr [nbr_coef - i] =
                        
                        v = sf2r [i] * s + sf2i [i] * c;
                        dfi [i] = v + sf1i [i];
                        dfi [nbr_coef - i] = v - sf1i [i];
                    }
                    
                    coef_index += d_nbr_coef;
                } while (coef_index < N);
                
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

    // ========================================================================== //
    //      Description: Compute the inverse FFT of the array. Notice that        //
    //                   IFFT (FFT (x)) = x * N (x). Data must be                 //
    //                   post-scaled.                                             //
    //      Input parameters:                                                     //
    //        - f: pointer on the source array (frequencies).                     //
    //             f [0...N(x)/2] = real values,                                  //
    //             f [N(x)/2+1...N(x)] = imaginary values of                      //
    //               coefficents 1...N(x)-1.                                      //
    //      Output parameters:                                                    //
    //        - x: pointer on the destination array (time).                       //
    //      Throws: Nothing                                                       //
    // ========================================================================== //
    
    void do_ifft(const T *f, T *x)
    {
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
                
                auto coef_index = 0;
                do {
                    auto coef_im_index = coef_index + nbr_coef;
                    auto sfr = &sf [coef_index];
                    auto sfi = &sf [coef_im_index];
                    auto df1r = &df [coef_index];
                    auto df2r = &df [coef_im_index];
                    
                    // Extreme coefficients are always real
                    
                    df1r [0] = sfr [0] + sfr [nbr_coef];
                    df2r [0] = sfr [0] - sfr [nbr_coef];
                    df1r [h_nbr_coef] = sfr [h_nbr_coef] * 2;
                    df2r [h_nbr_coef] = sfi [h_nbr_coef] * 2;
                    
                    // Others are conjugate complex numbers
                    
                    auto df1i = &df1r [h_nbr_coef];
                    auto df2i = &df1i [nbr_coef ];
                    for (auto i = 1; i < h_nbr_coef; ++i)
                    {
                        df1r [i] = sfr [i] + sfi [-i]; // + sfr [nbr_coef - i]
                        df1i [i] = sfi [i] - sfi [nbr_coef - i];
                        
                        auto c = cos_ptr [i];
                        auto s = cos_ptr [h_nbr_coef - i];
                        auto vr = sfr [i] - sfi [-i]; // - sfr [nbr_coef - i];
                        auto vi = sfi [i] + sfi [nbr_coef - i];
                        
                        df2r [i] = vr * c + vi * s;
                        df2i [i] = vi * c - vr * s;
                    }
                    
                    coef_index += d_nbr_coef;
                } while (coef_index < N);
                
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
            
            auto coef_index = 0;
            do {
                df [coef_index] = sf [coef_index] + sf [coef_index + 4];
                df [coef_index + 4] = sf [coef_index] - sf [coef_index + 4];
                df [coef_index + 2] = sf [coef_index + 2] * 2;
                df [coef_index + 6] = sf [coef_index + 6] * 2;
                
                df [coef_index + 1] = sf [coef_index + 1] + sf [coef_index + 3];
                df [coef_index + 3] = sf [coef_index + 5] - sf [coef_index + 7];
                
                auto vr = sf [coef_index + 1] - sf [coef_index + 3];
                auto vi = sf [coef_index + 5] + sf [coef_index + 7];
                
                df [coef_index + 5] = (vr + vi) * SQ2_2;
                df [coef_index + 7] = (vi - vr) * SQ2_2;
                
                coef_index += 8;
            } while (coef_index < N);
            
            
            // Penultimate and last pass at once
            
            coef_index = 0;
            auto bit_rev_lut_ptr = _bit_rev_lut.get_ptr();
            const T * sf2 = df;
            do {
                {   auto b_0 = sf2[0] + sf2[2];
                    auto b_2 = sf2[0] - sf2[2];
                    auto b_1 = sf2[1] * 2;
                    auto b_3 = sf2[3] * 2;
                    
                    x[bit_rev_lut_ptr[0]] = b_0 + b_1;
                    x[bit_rev_lut_ptr[1]] = b_0 - b_1;
                    x[bit_rev_lut_ptr[2]] = b_2 + b_3;
                    x[bit_rev_lut_ptr[3]] = b_2 - b_3;
                }
                {   auto b_0 = sf2[4] + sf2[6];
                    auto b_2 = sf2[4] - sf2[6];
                    auto b_1 = sf2[5] * 2;
                    auto b_3 = sf2[7] * 2;
                    
                    x[bit_rev_lut_ptr[4]] = b_0 + b_1;
                    x[bit_rev_lut_ptr[5]] = b_0 - b_1;
                    x[bit_rev_lut_ptr[6]] = b_2 + b_3;
                    x[bit_rev_lut_ptr[7]] = b_2 - b_3;
                }
                sf2 += 8;
                coef_index += 8;
                bit_rev_lut_ptr += 8;
            } while (coef_index < N);
        }
        
        //   Special cases
        
        // 4-point IFFT
        else if (nbr_bits == 2) {
            auto b_0 = f[0] + f[2];
            auto b_2 = f[0] - f[2];
            
            x[0] = b_0 + f[1] * 2;
            x[2] = b_0 - f[1] * 2;
            x[1] = b_2 + f[3] * 2;
            x[3] = b_2 - f[3] * 2;
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
    
private:
    
    
    // Bit-reversed look-up table nested class
    class BitReversedLUT
    {
    public:
        
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
        const int *get_ptr () const {
            return _ptr;
        }
        
    private:
        int *_ptr = NULL;
    };
    
    // Trigonometric look-up table nested class
    class TrigoLUT
    {
    public:
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
        const T1 * get_ptr(const int level) const
        {
            return (_ptr + (1 << (level - 1)) - 4);
        }
        
    private:
        T1 *_ptr = NULL;
    };
    
protected:
    
    const BitReversedLUT _bit_rev_lut;
    const TrigoLUT       _trigo_lut;
    
};


