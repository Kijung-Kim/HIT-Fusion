# Unit-peak voting revision (v2)

The submitted manuscript is retained unchanged. This code revision is a
separate experiment and must not be described as an exact implementation of
its normalized Gaussian equation or as reproducing its archived 97.30%.

## Method change

Each candidate now contributes `exp(-r^2/(2*sigma^2))`, with peak exactly 1.
`sigma = 40 / diameter` remains unchanged. Votes still accumulate across
candidates and frames. Removing `1/(2*pi*sigma^2)` eliminates the quadratic
diameter-dependent peak multiplier. Equal peaks do not imply equal integrated
vote mass: candidate size still changes bandwidth and spatial overlap.

Retinex, candidate extraction, person ROI, motion scores, stable-neighbor
verification, refinement, and GT sets are unchanged by this method experiment.

## Preserved versions

- v0: original public code, commit `15619148902a025e0338aab317ea1d335f681da4`.
- v1: normalized Gaussian manuscript reconstruction, tag
  `baseline-normalized-gaussian-v1`, commit `b1d5115`.
- v2: unit-peak Gaussian revision; measured results are appended after the
  complete run. The default branch is a local development branch.

Ground-truth interpretation has a separate history. The original Excel has
74 annotated clips. Following the user's clarification, p33 and p47 are
unscored, leaving 72; P16 and P16_short retain their Excel labels pending
confirmation. Comparisons must use the same GT definition on both versions.

## Validation

The rule tests check identical unit peaks across candidate sizes, the expected
one-sigma value, invalid diameters, symmetry with floating-point tolerance,
and the existing temporal decision rules. Full evaluation reuses the same
cached person segmentation masks to isolate the voting change.

## Completed evaluation

| GT definition | v1 correct | v2 correct | v2 localization success |
|---|---:|---:|---:|
| excel74 | 52/74 | 60/74 | 81.08% |
| user72_P16_pending | 52/72 | 60/72 | 83.33% |

On the user-corrected 72-clip set, v2 has 5 wrong-frame
predictions and 7 missing outputs. Corrected failures:
p18, p19, p20, p21, p23, p42_green, M20191014160641664, M20191014160740950. Regressions from correct to incorrect:
none.

This is an improvement in correct local predictions.
These clips informed development and are not an independent generalization test.
No claim of reproducing the submitted archived result is made. The manuscript
was not edited and nothing was published remotely. Historical v1 documents
remain as the record of that version. Full private evidence is outside the repo.
