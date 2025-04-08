FFTReal v1.2
------------

This is the best (fastest, most precise and best sounding) real FFT/iFFT transform available 
in existence! I have gone through at least 10, and kept upgrading until I found this 
one, so I arranged it into a templated class that suports base types like float and double
as well as simd vectors like simd_float8, simd_double4, which can be specified as a template
parameter T. I also spent a great deal of time hand optimizing it in neon assembler, and
then spent another month refining that, which I am pleased to announce that it is available
for sale from me directly. The additional (for sale) optimization takes optimizations to a 
whole another level gaining about 30%-50% more performnace gain. I apologize for the extra
hassle w/ transactions and emails, I will make it a paypal check out soon. The pricing for 
my highly optimized neon asm routines is $200/project and $1000 for a company-wide use on
multiple projects - up to 10. This is an ideal package if you are doing multi-channel p
rocessing that would leverage instrinsic vector architecture. 

This is what I am using for now in all of my audio projects instead of AVFFT, PFFFT,
etc. It is by far the best sounding and precise FFT I found, and you have my word on it,
I don't just say things without testing.

v1.2 (NEW): I switched the internal dynamic arrays allocation which gives it better
flexibility and efficiency for memory usage. Here's an example usage of RealFFT: 

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
