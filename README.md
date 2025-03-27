FFTReal v1.2
------------

This is the best (most precise and best sounding) real FFT/iFFT transforms available 
in existence! I have gone through at least 10, and kept upgrading until I found this 
one, so I adapted it for SIMD capability and arranged it into a templated class that
works with SIMD vectors like simd_float8, simd_double4, etc. I highly optimized it 
in neon asm, and now offering the highly optimized neon version that performs at least
50% better than the one available for free here neon optimizations. Contact my email 
for more info, but I intend to make a simple checkout cart, so you can instantly buy it 
with an instant checkout. The pricing for my highly optimized neon asm routines is 
$200/project but I will offer a company or organization-wide license also for $500.

This is what I am using now in all of my audio projects instead of AVFFT, PFFFT,
etc. It is by far the best sounding that I found that is also fast as lightning.

NEW v1.1: I optimized the algo a bit with neon inline asm but I am no wizard at neon, 
just trying to learn. If anyone wanting to optimize this further, I'd appreciate if 
you send me your optimizations, email below.

DMT <subband@protonmail.com>
