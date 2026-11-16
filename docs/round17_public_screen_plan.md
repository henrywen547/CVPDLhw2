# Round 17: source share 40% and balanced BN + 30% source

Round 16 changed the picture. `lame8k20_ema03_src03` (public 0.89420) is now selected. Across 0/10/20/30% source shares, public rose monotonically while validation stayed flat. Class-balanced statistics without any source prior (`lame_bbn05`) also gained +0.00293 over `lame8k20_ema03`.

At the user's request, two follow-ups were checked with one test pass each at `356cea2`:

1. **`lame8k20_ema03_src04`**: EMA 0.3 + 40% source statistics. Does the public trend continue past 30%?
2. **`lame_bbn05_src03`**: class-balanced statistics (γ = 0.5) + EMA 0.3 + 30% source statistics. Do the two mechanisms, which each gain on public, combine?

| Run | Validation | Val s | Test s (local) | Peak VRAM MiB | Changed vs. src03 | Changed vs. bbn05 | submission.csv SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lame8k20_ema03_src04` | 2012 (.952652) | 16.2 | 92.3 | 848 | 305 | 950 | `fc83315c…a43d2ff` |
| `lame_bbn05_src03` | 2017 (.955019) | 16.9 | **96.1** | 850 | 254 | 769 | `654b6a42…53460e` |

Validation is within noise for both, as in rounds 9–16. `lame_bbn05_src03` takes 96.1 s locally, a 6.1 s overrun. If its public gain is clear, it should first get a speed optimization of the weighted statistics, or the user should decide explicitly on the time risk.

Selection rule: unchanged. Switch only on a public score above 0.89420. A gain of one or two images is noise.
