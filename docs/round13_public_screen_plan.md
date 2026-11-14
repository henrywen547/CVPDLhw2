# Round 13: public-score screening of two principled BN candidates

Selected method: `lame8k20_ema03`, validation 2019/2112, public **0.89143**. Rounds 9–12 found no validation gain. Validation now sits at .956 while public is .891: 93 validation errors remain, too few to detect small real improvements, and the 6.5-point gap shows the test stream contains harder corruptions than validation. With the user's agreement, this round therefore runs one full test pass for each of two **predeclared, mechanism-motivated** candidates and lets the public score decide. Neither candidate was tuned on public. Public and private cover different corruption types, so this screening is limited to these two candidates.

| Candidate | Validation | Rationale |
| --- | ---: | --- |
| A `lame8k20_ema03_src01`: per-view EMA BN 0.3 + 10% source statistics | 2020 (+1) | On harder corruptions, statistics from 64 correlated images are less reliable; a small source prior anchors them at no validation cost. |
| B `lame8k20_mix07`: 0.7 current batch + 0.3 source statistics, no EMA | 2011 (−8) | No lag after domain changes; tests whether EMA's lag or its smoothing matters more on the longer test stream. |

Both run from the same code commit; later code additions (no-op hooks, optional sample weights) are inactive for these methods. Each CSV stays separate. The root `submission.csv` changes only if a candidate's public score exceeds 0.89143.

## Results

Pending.
