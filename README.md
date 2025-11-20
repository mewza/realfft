FFTReal v1.4
------------

<b>NEW (v1.4):</b> Added FFTRealHybrid auto-optimizer class so you can continue constructing FFTReal with scalars or simd type vectors, and it will auto-optimize it with best possible vector SIMD to get maximum performane. So if you were to pass FFTRealHybrid it would actually process FFTReal internally with simd_float8 but you will see data resulting from forward transform as cmplxT. So, this is essentially an optimizer class for float, simd_float2, simd_float4, double, simd_double2 types utilizing SIMD architecture. 

How to reach me: 
            Dmitry Boldyrev <subband@gmail.com> or 
                            <subband@protonmail.com>

This is the best (fastest, most precise and best sounding) real FFT/iFFT transform available 
in existence! I have gone through at least 10, and kept upgrading until I found this 
one, so I arranged it into a templated class that suports base types like float and double
as well as simd vectors like simd_float8, simd_double4, which can be specified as a template
parameter T. 

This is what I am using for now in all of my audio projects instead of AVFFT, PFFFT,
etc. It is by far the best sounding and precise FFT algorithm I found. Now, go make some awesome
audio apps! 
      // compute_fft_size() is an inline const function in your class
     
      const int fft_size = compute_fft_size(1024); 
      { 
            // Delcare instance of FFTReal
            FFTReal<simd_double2> fft(fft_size);

            fft.real_fft(data_in, data_out);
            
            // ... process data_out here ...
            
            // 3rd parameter optional, true = scale by 1/length, 
            // omitted or false = don't scale
            
            fft.real_ifft(data_out, data_in, true); 
      }
      
This style of initialization was not possible the way class was setup before (static),
which required maximum FFT size initialization. Well, enjoy the world's best FFT class
in the world! =)

Dmitry <subband@protonmail.com>
