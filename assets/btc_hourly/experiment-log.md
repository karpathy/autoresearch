# Experiment Log — autotrader/mar20b

**Goal**: Recalibrate pipeline for expanding windows with sample weight decay.
**Starting recipe**: 36 features, 9 monotonic constraints, 2-model HGB ensemble, power transform x^0.7, 0.3 dampening, EMA span 45, 0.7x partial demeaning. Prior best (sliding windows): 0.6031.
**Infrastructure change**: Expanding windows (train on all data from 2018 to eval year) + exponential sample weight decay on older samples.

---

