<b>FFTReal 2.1   --  The most optimized and versitile SIMD vector or scalar forward/inverse FFT real/complex
                  transform developed by the Russian-American software engineer and inventor of MacAmp and Winamp<b>

<b>NEW (v1.67):</b> Updated to v2.1 (Fixed NEON compilation issue with Xcode v26.1 (17B54))
                    for some reason .f[2] and .f[4] suddenly stopped working and had to be replaced
                    with (float*)addr + 2, + 4 throughout entire NEON code. God knows?

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
