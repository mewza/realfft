FFTReal v1.2
------------

This is the best (fastest, most precise and best sounding) real FFT/iFFT transform available 
in existence! I have gone through at least 10, and kept upgrading until I found this 
one, so I adapted it for SIMD capability and arranged it into a templated class that
works with SIMD vectors like simd_float8, simd_double4, etc. I highly optimized it 
in neon asm, and now offering the highly optimized neon version that performs at least
50% better than the one available for free here neon optimizations. Contact my email 
for more info, but I intend to make a simple checkout cart, so you can instantly buy it 
with an instant checkout. The pricing for my highly optimized neon asm routines is 
$200/project and I will offer a company or organization-wide license for $500. It is also
an ideal package if you are doing multi-channel processing like I am doing in my iOS app,
that individually processes 7.1 surround (8 channels) simultaneously with SIMD intrinsic
vectors representation.

This is what I am using now in all of my audio projects instead of AVFFT, PFFFT,
etc. It is by far the best sounding that I found that is also fast as lightning.

NEW v1.2: I optimized the neon assembly routines so that they gave another 30%-50%
boost in performance, which directly translates in audio apps how they feel - snappy 
or laggy especially in dsp-intensive apps like one I am developing. I added tweedle 
caching, which gave it an extra boost in the free verrsion. The code here is free to use, 
but if you want maximum performance, consider licensing highly optimized neon version. 
For the commercial part of it, I implemented custom routines for every simd type: 
simd_float8, simd_float4, simd_float2, simd_float, simd_double8, simd_double4, 
simd_double2, and simd_double, each routine is customly optimized in neon asm for 
maximum performance. I also switched the internal arrays allocation into dynamic 
for reasons that you can now statically declare like this and it is more efficient
on memory usage:

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
      
This was not possible before with FFTReal<simd_double2, 1024> fft; way of doing it.

Dmitry <subband@protonmail.com>
