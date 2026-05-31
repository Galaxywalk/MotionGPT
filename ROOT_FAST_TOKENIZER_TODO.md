# Root-FAST Tokenizer TODO

This is the cold-start TODO for the next compute window. Current state:

- Root/local factorization works. R3 is the quality reference, but its root
  latent is too large.
- Root-FAST continuous DCT coefficients are the strongest compact root
  representation so far.
- Coefficient discretization works with scalar quantization, but plain
  one-token-per-chunk vector VQ is too lossy.
- Product VQ is better than vector VQ, but still not good enough to be the
  final root tokenizer.
- Root-FAST RVQ has now been implemented and evaluated. It is the strongest
  token-like root representation so far.

## Current Artifacts

Code:

```text
src/motiongpt_m4human/factorized/root_fast_codec.py
src/motiongpt_m4human/factorized/root_fast_quantize.py
```

Experiment outputs:

```text
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_dct_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_quantized_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_vector_fixed_v2
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_scalar_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_product_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_rvq_v1
/cpfs01/liangbo/data/MotionGPT/factorized_experiments/root_fast_full_eval_v1
```

Main docs:

```text
ROOT_LATENT_COMPRESSION.md
FACTORIZED_TOKENIZER.md
M4HUMAN_EVAL_20260530.md
```

## Key Numbers

Root-only DCT continuous codec, using GT local pose:

| config | values/window | MPJPE | root xz mean | final xz |
| --- | ---: | ---: | ---: | ---: |
| chunk=16, K=4 | 208 | 1.20 mm | 1.12 mm | 0.42 mm |
| chunk=16, K=2 | 104 | 5.02 mm | 4.80 mm | 3.64 mm |
| chunk=32, K=4 | 112 | 4.16 mm | 3.96 mm | 2.94 mm |
| chunk=32, K=2 | 56 | 18.06 mm | 17.61 mm | 11.43 mm |

Product VQ, test196:

| root tokens | chunk | K | vocab/group | bits/window | MPJPE | root xz mean | final xz |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 64 | 2 | 256 | 128 | 82.61 mm | 80.09 mm | 106.65 mm |
| 28 | 32 | 2 | 256 | 224 | 53.67 mm | 51.57 mm | 80.14 mm |
| 52 | 16 | 2 | 256 | 416 | 41.20 mm | 39.68 mm | 63.61 mm |

Root-FAST RVQ, best test196 by token budget:

| root tokens | chunk | K | vocab | depth | bits/window | MPJPE | root xz mean | final xz |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 98 | 4 | 1024 | 4 | 80 | 89.97 mm | 87.11 mm | 128.46 mm |
| 16 | 98 | 4 | 512 | 8 | 144 | 54.57 mm | 53.09 mm | 67.76 mm |
| 28 | 32 | 2 | 1024 | 4 | 280 | 35.12 mm | 34.11 mm | 45.47 mm |
| 52 | 16 | 2 | 1024 | 4 | 520 | 23.23 mm | 22.52 mm | 34.07 mm |
| 56 | 32 | 2 | 512 | 8 | 504 | 20.15 mm | 19.64 mm | 16.55 mm |
| 104 | 16 | 2 | 1024 | 8 | 1040 | 7.03 mm | 6.76 mm | 7.47 mm |

Full local VQ + Root-FAST RVQ, M4Human test196:

| local ckpt | root setting | root tokens | total tokens | MPJPE | root-align | gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| local_v1 | balanced56 | 56 | 102.99 | 58.53 mm | 53.79 mm | 4.74 mm |
| local_v1 | high104_vocab1024 | 104 | 150.99 | 52.79 mm | 52.14 mm | 0.65 mm |
| scratch_full | balanced56 | 56 | 102.99 | 60.69 mm | 56.05 mm | 4.64 mm |
| scratch_full | high104_vocab1024 | 104 | 150.99 | 55.13 mm | 54.50 mm | 0.63 mm |

Scalar quantization, test196:

| config | scalar codes | bits/window | MPJPE | root xz mean | final xz |
| --- | ---: | ---: | ---: | ---: | ---: |
| chunk=16, K=2, 8-bit | 104 | 832 | 9.94 mm | 9.48 mm | 11.98 mm |
| chunk=32, K=4, 8-bit | 112 | 896 | 9.36 mm | 8.91 mm | 12.32 mm |
| chunk=64, K=4, 8-bit | 64 | 512 | 19.08 mm | 18.53 mm | 18.64 mm |

## Interpretation

Token count and vocabulary size are separate controls:

```text
token count:     controlled mainly by chunk size and quantizer layout
vocabulary size: controlled by codebook size per token
capacity:        approximately token_count * log2(vocab_size)
```

Plain vector VQ uses very few tokens but asks one token to represent a full DCT
chunk. That is too hard at vocab sizes up to 1024. Product VQ uses more tokens
and smaller per-token targets, so it improves substantially. Scalar quantization
shows that the coefficients themselves are easy to discretize if enough scalar
codes are allowed.

The updated target is a middle ground:

```text
28-56 root tokens for balanced compression
104 root tokens for high-quality reconstruction
shared vocab size around 256-1024
```

## Completed Compute Items

1. Fixed full vector VQ sweep.

   The fixed sweep confirms the earlier conclusion: full-chunk vector VQ is too
   lossy. Best test196 results are:

   ```text
   13 tokens, vocab 1024: 128.28 mm
    7 tokens, vocab 1024: 139.29 mm
    4 tokens, vocab 1024: 164.43 mm
   ```

2. Root-FAST RVQ implementation and sweep.

   RVQ is implemented as:

   ```text
   coeff vector -> code_1 + residual -> code_2 + ... -> code_R
   ```

   Completed sweep:

   ```text
   chunk sizes: 16, 32, 64, 98, 196
   K:           2, 4
   vocab:       256, 512, 1024
   RVQ depth:   2, 4, 8
   ```

3. Full local VQ + Root-FAST RVQ eval.

   The full eval confirms that high-quality Root-FAST RVQ makes root drift
   negligible in the full tokenizer. For `local_v1 + high104_vocab1024`, test196
   is:

   ```text
   full MPJPE / root-align / gap: 52.79 / 52.14 / 0.65 mm
   ```

   The local-only upper bound for the same checkpoint is `51.17 mm`, so the next
   reconstruction bottleneck is local body tokenization.

## Next Compute TODO

1. Decide the root-token operating point for downstream prediction.

   Current full-tokenizer interpretation:

   ```text
   28 root tokens:  too lossy for high-quality reconstruction
   56 root tokens:  compact balanced setting, 58.53 mm test196
   104 root tokens: high-quality setting, 52.79 mm test196
   ```

2. Improve local body tokenization.

   Root drift is no longer the dominant error at high104. The next
   reconstruction work should improve local VQ quality or replace the current
   local VQ with a stronger local/part-wise tokenizer.

3. Train token predictors.

   First verify proxy tasks:

   ```text
   local VQ tokens -> Root-FAST tokens
   local continuous features -> Root-FAST tokens
   ```

   Then move to:

   ```text
   M4Human/mmWave features -> local VQ tokens + Root-FAST tokens
   ```

## Do Not Prioritize

- More R3 training. R3 is already a quality reference, not a compact tokenizer.
- Plain one-token-per-chunk vector VQ. The fixed sweep confirms it is too lossy.
- Larger product VQ vocab. RVQ is now a better token-like path.
- New root trajectory neural losses before the token representation is settled.
