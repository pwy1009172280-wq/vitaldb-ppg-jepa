# VitalDB PPG preprocessing

The v0 baseline processes the VitalDB `SNUADC/PLETH` track at its source sampling
rate of 500 Hz. It first creates non-overlapping 10-second windows (5000 samples
per window), dropping an incomplete trailing window without padding. Each window
is screened independently. Blank waveform fields in the raw CSV are parsed as
missing/non-finite values; any NaN or Inf rejects the window, as does a clearly
degenerate near-constant signal protected by the numerical epsilon `1e-8`.

Each valid window is then filtered independently with `scipy.signal.cheby1`
(Chebyshev Type I, order 4, `rp=0.5` dB, cutoffs 0.5–12 Hz, `fs=500`) using
second-order sections and `scipy.signal.sosfiltfilt` for zero-phase filtering.
SciPy's default `sosfiltfilt` padding behavior is used; no padding parameter is
overridden. Filtered output and normalized output are checked for finiteness,
then the per-window Z-score is returned as float32.

Window-first processing prevents missing values in one window from propagating
into neighboring windows. Independent per-window zero-phase filtering can
introduce window-edge effects; this is an accepted limitation of this minimal
baseline. More advanced valid-segment filtering may be evaluated later if
needed. No resampling or interpolation is used, and raw files under `data/raw/`
remain unchanged. Advanced SQI, artifact, morphology, and physiological rules
are outside this baseline.
