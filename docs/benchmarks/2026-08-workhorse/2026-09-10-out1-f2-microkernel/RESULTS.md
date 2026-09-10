# Experimental result: DA3 `out1` F(2) AVX-512 microkernel

## Hypothesis

The DA3-BASE DPT output convolution has one fixed high-work F(2) Winograd
shape:

```text
input tiles: 4
transformed positions: 16
channels: 128 input → 64 output
```

An AVX-512 microkernel with these dimensions as compile-time constants should
remove generic tile-count handling and dynamic slice indexing while retaining
the existing transformed-filter layout and FMA order.

## Change

`vestra-kernels` commit `f87a922d5df0dbf471df3269bde8a21d862cb239` adds an
opt-in kernel selected only by
`DA3_KERNELS_ENABLE_OUT1_F2_128X64=1`. All other shapes and normal builds use
the established generic F(2) kernel. The engine pins this candidate at
`d9ed47266aeb871fa338bc8d4c4376f4b9261554`.

The new kernel's unit test compares its complete 128→64/four-tile output
bit-for-bit with the generic F(2) route. The engine library suite passed 71/71
tests after the pin.

## Four-image C++ F32 parity

| Image | Pearson r | MAE |
| --- | ---: | ---: |
| canyon | 0.99999362795732 | 0.0018125212874871735 |
| desk | 0.9999782568224567 | 0.001772925931360581 |
| mountains | 0.9999855775226433 | 0.003675203580509249 |
| street | 0.9999721239891701 | 0.0008210032855921501 |

Every image passes the locked `r >= 0.9999`, `MAE <= 0.005` gate.

## Short diagnostic profiles

The Workhorse had unrelated activity, so these are not a qualified benchmark.
Consecutive `DA_HEAD_PROFILE=1` measurements nevertheless isolate the intended
operation:

| Arm | `out1` time |
| --- | ---: |
| existing generic F(2) | 7.701 ms |
| exact-shape candidate | 5.219 ms |
| existing generic F(2) | 6.400 ms |
| exact-shape candidate | 5.406 ms |

Paired whole-inference five-iteration smokes were not stable enough to decide
promotion (control medians: 189.397 and 202.750 ms; candidate medians:
197.035 and 200.154 ms). Therefore the candidate remains opt-in. It must win
an idle-host smoke before becoming the default and must then enter the locked
randomized trial protocol.

## Combined diagnostic with passive OpenMP workers

With the separately documented `KMP_BLOCKTIME=0` candidate enabled, alternating
five-iteration smokes gave 205.667 and 197.133 ms for blocktime-only versus
192.174 and 189.266 ms with the out1 microkernel as well. The direction is
consistent with an additional small gain, but the host remained busy; this is
not promotion evidence or an official benchmark result.
