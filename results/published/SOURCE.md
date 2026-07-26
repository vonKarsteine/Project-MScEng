# Published reference figures — provenance

**These numbers were not produced by this checkout.**

Every value under `tables/` is **transcribed by hand from MScEng.pdf, Chapter 4**.
They are a copy of what the dissertation reports, kept in machine-readable form.
They are not the output of any code in this repository, and nothing in this
repository has reproduced them.

## Where they came from

The figures were produced by an earlier checkout, on the hardware described in
MScEng.pdf section 4.1.4:

| | |
|---|---|
| Operating system | Windows 11 |
| CPU | Intel Core i9-14900KF |
| GPU | NVIDIA RTX 4090, 24 GB |
| Arithmetic | FP32 |

The GPU-hour figures of Table 4.4 are stated against that machine and mean
nothing on different hardware.

## What this directory is for

Exactly one thing: so that tests can assert **the code's constants agree with the
text**. The dissertation states class counts, pool sizes, route counts,
resampling parameters and thesis constants; the package hard-codes the same
values in `koa_multimodal.records.schema`, `koa_multimodal.config.schema` and
`koa_multimodal.models.catalog`. A test that reads a table here and compares it
against a constant there catches the case where the code and the write-up drift
apart.

That is a check of *agreement with the text*. It is not evidence that this
checkout can reproduce the results, and it must not be described as such.

## What this directory is not for

**Never copy a value from here into a run record under `results/<run_id>/`.**

A run record answers "what did this checkout produce?". Moving a transcribed
number into one changes the answer to a false statement, and it launders a figure
from the write-up into something that looks like machine output with a checkpoint
behind it.

This is enforced, not merely requested. `koa_multimodal.records.validate` refuses
any record whose manifest declares `status: "pending"` while a metric field
anywhere in the tree carries a value. Every record in this checkout is pending —
no training will be run here — so every one of these numbers is rejected on
sight if it is pasted in. Do not work around the check; the check is the point.

## Contents

`tables/index.json` lists every file with its caption.

| File | Table | Contents |
|---|---|---|
| `table_4_1.json` | 4.1 | KL distribution of the original X-ray corpus (8,147) |
| `table_4_2.json` | 4.2 | KL distribution of the paired cohort by split (3,536) |
| `table_4_3.json` | 4.3 | Training hyperparameters |
| `table_4_4.json` | 4.4 | Computational cost by stage |
| `table_4_5.json` | 4.5 | X-ray unimodal candidates |
| `table_4_6.json` | 4.6 | MRI unimodal candidates |
| `table_4_7.json` | 4.7 | Traditional fusion operators |
| `table_4_8.json` | 4.8 | Contrastive fusion |
| `table_4_9.json` | 4.9 | RCKF fusion |
| `table_4_10.json` | 4.10 | Ensemble and C-MODES routing |
| `table_4_11.json` | 4.11 | Oracle selection bounds |
| `table_4_12.json` | 4.12 | Bootstrap 95% confidence intervals |
| `table_4_13.json` | 4.13 | Paired bootstrap differences |
| `table_4_15.json` | 4.15 | QAT student against the FP32 teacher |

## Fidelity of the transcription

Numeric rows are verbatim. Two things are not:

- **Captions** are descriptive summaries written for this index, not quotations
  of the thesis captions.
- **Table 4.3** is transcribed *selectively*. Each row carries a `sourceLocus`
  naming where in MScEng.pdf the value is stated, and parameters the text does
  not state are omitted rather than filled in from the package defaults. A
  hyperparameter table that silently mixed reported values with code defaults
  would be useless as a cross-check, since the code would then be compared
  against itself.

Where MScEng.pdf names a model pool by family rather than member by member — the
bimodal pool of Tables 4.10 and 4.11 — the wording is transcribed as written and
not resolved into a list of candidate ids.

## Internal consistency

The transcription is self-consistent across tables, which is a useful check on
the data entry:

- Table 4.1 grades sum to 8,147; Table 4.2 sums to 3,536 by split, by grade, and
  in total.
- Table 4.9 row 1 equals the "ConvNeXt V2 + SimSiam SSL (RCKF)" point estimates
  of Table 4.12.
- Table 4.10 rows 2 and 3 equal the "X-ray C-MODES" and "Fusion OOF Stack" point
  estimates of Table 4.12.
- Table 4.10 row 4, the Table 4.12 "Fusion C-MODES" row, and the FP32 teacher of
  Table 4.15 are the same three numbers.
- Table 4.13's differences equal Fusion C-MODES minus each baseline as tabulated
  in Table 4.12.
- Table 4.15's delta row equals student minus teacher.
- Table 4.4 costs 11 fusion models, matching the 3 + 4 + 4 routes of Tables 4.7,
  4.8 and 4.9.

These identities are asserted in the verification described above. They confirm
the numbers were copied correctly; they say nothing about whether the numbers are
reproducible here.
