| Method | Attack | BER (mean ± std) | NC (mean ± std) |
|---|---|---|---|
| DCT-DWT | None | 0.0050 ± 0.0157 | 0.9949 ± 0.0161 |
| DCT-DWT | JPEG Q=90 | 0.0058 ± 0.0166 | 0.9941 ± 0.0170 |
| DCT-DWT | JPEG Q=70 | 0.0110 ± 0.0213 | 0.9888 ± 0.0217 |
| DCT-DWT | JPEG Q=50 | 0.0358 ± 0.0206 | 0.9639 ± 0.0214 |
| DCT-DWT | Gaussian noise s=5 | 0.0195 ± 0.0270 | 0.9803 ± 0.0275 |
| DCT-DWT | Gaussian noise s=10 | 0.1230 ± 0.0358 | 0.8765 ± 0.0370 |
| DCT-DWT | Gaussian blur 5x5 s=1 | 0.1265 ± 0.0732 | 0.8727 ± 0.0747 |
| DCT-DWT | Median 3x3 | 0.1133 ± 0.0805 | 0.8860 ± 0.0817 |
| DCT-DWT | Scaling 0.5 -> 1 | 0.0172 ± 0.0258 | 0.9826 ± 0.0265 |
| DCT-DWT | Cropping 25% | 0.1285 ± 0.0157 | 0.8617 ± 0.0179 |
| DCT-DWT | Rotation 5 (realigned) | 0.0598 ± 0.0335 | 0.9397 ± 0.0345 |
| DCT-DWT | Rotation 5 (no realign) | 0.4970 ± 0.0147 | 0.5015 ± 0.0208 |
| DCT-DWT | Combined JPEG70+noise5 | 0.0402 ± 0.0324 | 0.9596 ± 0.0330 |
| DCT-DWT-SVD | None | 0.0007 ± 0.0020 | 0.9993 ± 0.0020 |
| DCT-DWT-SVD | JPEG Q=90 | 0.0021 ± 0.0048 | 0.9979 ± 0.0047 |
| DCT-DWT-SVD | JPEG Q=70 | 0.0159 ± 0.0166 | 0.9842 ± 0.0162 |
| DCT-DWT-SVD | JPEG Q=50 | 0.0605 ± 0.0274 | 0.9400 ± 0.0263 |
| DCT-DWT-SVD | Gaussian noise s=5 | 0.0404 ± 0.0533 | 0.9596 ± 0.0531 |
| DCT-DWT-SVD | Gaussian noise s=10 | 0.1603 ± 0.0522 | 0.8398 ± 0.0509 |
| DCT-DWT-SVD | Gaussian blur 5x5 s=1 | 0.1232 ± 0.0755 | 0.8754 ± 0.0769 |
| DCT-DWT-SVD | Median 3x3 | 0.1522 ± 0.1006 | 0.8470 ± 0.1014 |
| DCT-DWT-SVD | Scaling 0.5 -> 1 | 0.0147 ± 0.0173 | 0.9852 ± 0.0174 |
| DCT-DWT-SVD | Cropping 25% | 0.1254 ± 0.0081 | 0.8654 ± 0.0083 |
| DCT-DWT-SVD | Rotation 5 (realigned) | 0.0585 ± 0.0314 | 0.9412 ± 0.0318 |
| DCT-DWT-SVD | Rotation 5 (no realign) | 0.4980 ± 0.0167 | 0.5020 ± 0.0196 |
| DCT-DWT-SVD | Combined JPEG70+noise5 | 0.0829 ± 0.0565 | 0.9171 ± 0.0571 |

| Method | PSNR (dB) | SSIM | Embed (ms) | Extract (ms) |
|---|---|---|---|---|
| DCT-DWT | 45.66 ± 0.43 | 0.9916 ± 0.0043 | 61.5 | 28.1 |
| DCT-DWT-SVD | 45.99 ± 0.22 | 0.9949 ± 0.0037 | 123.1 | 56.3 |
