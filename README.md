FFTReal v1.1

This is one of the best real FFT/iFFT transforms available in existence!
I have gone through at least 10, and kept upgrading until I found this one, 
so I adapted it for SIMD capability, optimized it in C++ and neon asm a bit.

This is what I am using now in all of my audio projects instead of AVFFT, PFFFT,
etc. It is by far the best sounding that I found that is also fast as lightning.

NEW v1.1: I optimized the algo a bit with neon inline asm but I am no wizard at neon, 
just trying to learn. If anyone wanting to optimize this further, I'd appreciate if 
you send me your optimizations, email below.

DMT <subband@protonmail.com>
