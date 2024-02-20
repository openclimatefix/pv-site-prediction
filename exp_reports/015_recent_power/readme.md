# Recent Power

The idea is to add some recent power values.
The model current uses the last 30 minutes but the average power. It would be good to give the actual values as well. 

We used `uk_pv.py` configuration, but removed the satellite input. We used 6 recent power values. 

We want to A/B test and see the difference. The total MAE for all horizons

|  | Model | Model with recent power |
|---------|-------|-------------------------|
| test   | 0.140 ± 0.020      | 0.122 ± 0.017                     |
| train   |0.192 ± 0.027       |  0.182 ± 0.026                    |

And just for the test set (The test set is 2020-01-01 to 2021-11-00):

| Horizon | MAE  | MAE with recent power |
|---------|------|--------|
| 0 -15   | 0.14 | 0.12   |
| 15-30   | 0.17 | 0.17   |
| 30-45   | 0.19 | 0.19   |
| 45-60   | 0.21 | 0.21   |

So this just makes a difference for the first 0-15 forecast of data