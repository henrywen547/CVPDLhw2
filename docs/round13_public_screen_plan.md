# Round 13: public-score screening of two principled BN candidates

Selected method: `lame8k20_ema03`, validation 2019/2112, public **0.89143**. Rounds 9–12 found no validation gain. Validation now sits at .956 while public is .891: 93 validation errors remain, too few to detect small real improvements, and the 6.5-point gap shows the test stream contains harder corruptions than validation. With the user's agreement, this round therefore runs one full test pass for each of two **predeclared, mechanism-motivated** candidates and lets the public score decide. Neither candidate was tuned on public. Public and private cover different corruption types, so this screening is limited to these two candidates.

| Candidate | Validation | Rationale |
| --- | ---: | --- |
| A `lame8k20_ema03_src01`: per-view EMA BN 0.3 + 10% source statistics | 2020 (+1) | On harder corruptions, statistics from 64 correlated images are less reliable; a small source prior anchors them at no validation cost. |
| B `lame8k20_mix07`: 0.7 current batch + 0.3 source statistics, no EMA | 2011 (−8) | No lag after domain changes; tests whether EMA's lag or its smoothing matters more on the longer test stream. |

Both run from the same code commit; later code additions (no-op hooks, optional sample weights) are inactive for these methods. Each CSV stays separate. The root `submission.csv` changes only if a candidate's public score exceeds 0.89143.

## Results

Both full test passes ran at commit `b654e43` on the local GTX 1080 Ti:

| Candidate | Local test s | Peak VRAM MiB | Test predictions differing from selected `lame8k20_ema03` | Upload file (SHA-256) |
| --- | ---: | ---: | ---: | --- |
| A `lame8k20_ema03_src01` | 92.3 | 848 | 341 / 12,288 (2.8%) | `runs/final_lame8k20_ema03_src01/submission.csv` (`d5e6f344…6aebb`) |
| B `lame8k20_mix07` | 92.0 | 848 | 916 / 12,288 (7.5%) | `runs/final_lame8k20_mix07/submission.csv` (`aee0e616…87080`) |

A and B differ from each other on 833 test predictions. Both CSVs match their logs, and reproduction snapshots are in `runs/<method>_candidate/`. No test labels were read. Public scores are pending; the root `submission.csv` stays `lame8k20_ema03`.

The user reported **A: public 0.89241** and **B: public 0.88688**, versus 0.89143 for the selected method.

- A's gain of +0.00098 is about six images of the 6,144-image public half, which is small and not established as meaningful. A changes only 2.8% of test predictions and is neutral on validation, so per the switching rule `lame8k20_ema03_src01` becomes the selected method for the root `submission.csv`, staged COOL files, and Q4 figure.
- B is lower by 0.00455. Dropping the EMA in favor of instant source-mixed statistics hurts on the test stream, so cross-batch smoothing matters more there than reacting immediately to domain changes.

## Follow-up: candidate C

At the user's request, after A's small public gain, the predeclared follow-up C, `lame8k20_ema03_src02` (EMA + 20% source statistics, validation 2018), received one full test pass at commit `a11ff47`. It took **92.3 s / 848 MiB** locally and differs from A on 343 and from `lame8k20_ema03` on 617 of the 12,288 test predictions. The upload file is `runs/final_lame8k20_ema03_src02/submission.csv` (SHA-256 `56924a47…a2fc57`); the snapshot is in `runs/lame8k20_ema03_src02_candidate/`. Public score pending; the root submission stays A. This ends the source-share direction: no larger share is planned whatever C scores, to avoid tuning on public.
