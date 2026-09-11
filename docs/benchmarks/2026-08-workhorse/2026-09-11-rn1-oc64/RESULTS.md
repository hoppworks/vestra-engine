# rn1 F(2) OC64 product candidate — accepted

## Hypothesis

The accepted rn1 `128 -> 128` Winograd F(2) product used two adjacent 16-wide
output panels (OC32). `DA3_RN1_F2_OC64=1` retains four panels at once. That
shares each transformed activation across four output vectors and removes half
of the activation broadcasts, while each output lane still performs its 128
FMA operations in ascending input-channel order.

## Implementation and correctness

- Kernel commit: `b3047cf` in `vestra-kernels`.
- Engine pin: `e2e9d62`.
- The existing OC32 route remains the control; OC64 is opt-in only.
- The target AVX-512/FMA oracle compared generic, OC32 and OC64 products bit
  for bit for signed irregular inputs.
- The four-image C++ F32 gate passed with OC64:

| Image | Pearson r | MAE |
|---|---:|---:|
| canyon | 0.9999936282 | 0.0018124970 |
| desk | 0.9999782565 | 0.0017728067 |
| mountains | 0.9999855792 | 0.0036749614 |
| street | 0.9999721254 | 0.0008209983 |

## Target admission measurement

AMD Ryzen 9 9950X; same release binary, model, `desk` image, 504x336,
16-thread budget and accepted common flags in both arms. Every process ran one
warm-up plus ten timed iterations. Pair order alternated. Only
`DA3_RN1_F2_OC32=1` versus `DA3_RN1_F2_OC64=1` differed.

| Pair | OC32 median ms | OC64 median ms |
|---:|---:|---:|
| 1 | 160.138 | 155.670 |
| 2 | 164.496 | 160.826 |
| 3 | 165.358 | 162.081 |
| 4 | 163.126 | 158.253 |
| 5 | 161.771 | 160.573 |
| 6 | 163.004 | 162.422 |
| 7 | 162.133 | 158.669 |
| 8 | 164.036 | 152.910 |
| 9 | 166.946 | 163.674 |
| 10 | 169.030 | 162.686 |
| **Mean** | **164.004** | **159.776** |

OC64 reduced mean trial-median latency by 4.228 ms and won 10/10 pairs. It
therefore exceeds the predeclared admission threshold (at least 2 ms and at
least 8/10 wins). It is retained for the next randomized C++/Rust qualification.

