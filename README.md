FFTReal v1.3
------------

NEW (v1.3): I decided to give away the highly optimized NEON code!  This is my gift and contribution
            to betterment of humanity. Enjoy, humanity, and (stop killing each other, stealing, plagirizing,
            work together and share fairly the winning! not try rip each other off, create havoc, 
            GOD doesn't like that and will punish you for that) - get well! One more stipulation, if you
            somehow manage to create a faster version of this using neon that does not sacrifice quality,
            you must email me your version of optimization. That's it!

How to reach me: 
            Dmitry Boldyrev <subband@gmail.com> or 
                            <subband@protonmail.com>
                            
TODO: Implement hybrid vectorization for base types : float, double utilizing vector simd_float8
and simd_double8 neon and non-neon instructions.

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
