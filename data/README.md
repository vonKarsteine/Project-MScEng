# `data/` — empty by design

The Osteoarthritis Initiative (OAI) subset this project trains on is **not
redistributable**, so this directory ships empty and every command that touches real
data stops at index load. That is the expected state: the verification run book in
the top-level `README.md` is built entirely from contract tests, synthetic forward
passes and dry runs, none of which need this tree.

## Expected layout

```
data/
└── training/
    ├── xray/
    │   ├── index/
    │   │   └── pairs_00m.csv        the single source of truth
    │   └── <subject>/<pair_key>.png
    └── mri/
        └── <subject>/
            ├── <pair_key>_t2.nii.gz
            └── <pair_key>_r2.nii.gz
```

`configs/default.toml` points at the index through `[data] xray_index`, and paths
inside the CSV are relative to `[data] training_root`.

## The index contract

`pairs_00m.csv` has exactly eight columns, one row per (subject, knee):

| column | meaning |
|---|---|
| `subject_id` | **The grouping key.** Every leakage control, every out-of-fold split and every clustered bootstrap groups on this. A patient contributes two knees, and splitting them across folds would leak. |
| `laterality` | `left` or `right`. Pure passthrough — nothing branches on it. |
| `split` | `train`, `val` or `test`. Patient-level, so a subject appears in exactly one. |
| `grade` | Kellgren–Lawrence grade, 0–4, read from the radiograph. |
| `xray_relpath` | PNG, relative to `training_root`. |
| `t2_relpath` | Quantitative T2 volume, NIfTI. |
| `r2_relpath` | R2 volume, NIfTI. |
| `pair_key` | **The global primary key.** Joins OOF artifacts, C-MODES alignment and deployment records. Must be unique. |

`koa check data --root data` audits all of this — header, grade range, unique pair
keys, per-split counts, and subject-level leakage across splits — and reports the
tree as absent rather than crashing when it is.

## Expected distribution

From the dissertation Tables 4.1–4.2. The original X-ray corpus holds 8,147 radiographs;
the standalone MRI repository holds 4,324 volumes; pairing on `subject_id` yields
**3,536** pairs, split roughly 7:1:2:

| split | total | KL0 | KL1 | KL2 | KL3 | KL4 |
|---|---|---|---|---|---|---|
| train | 2,506 | 988 | 443 | 680 | 314 | 81 |
| val | 357 | 138 | 63 | 95 | 50 | 11 |
| test | 673 | 259 | 119 | 188 | 85 | 22 |

Two consequences worth keeping in view. Pairing discards more than half the
radiographs, which is why the ensemble pools are capped at four members and why the
missing-modality path is a first-class case rather than an error branch. And KL1 —
the early boundary the whole method targets — is the smallest of the three central
classes in both the original and the paired cohort, which is why KL1 recall is
reported beside accuracy and QWK everywhere.

## Preprocessing applied on load

X-ray: grayscale → knee-centred square crop → resize to 384×384 → CLAHE
(clip 2.0, 8×8 tiles) → `(1, 384, 384)` in `[0, 1]`. ImageNet-gray normalisation
`(x − 0.449) / 0.226` runs **after** augmentation, because the augmenter clips to
`[0, 1]`.

MRI: T2 clipped to `[0, 110]` then divided by 110, R2 clipped to `[0, 1]`, both
resized to 32×384×384 and stacked as `(2, 32, 384, 384)` with **channel order
(T2, R2)**. MRI is never ImageNet-normalised.
