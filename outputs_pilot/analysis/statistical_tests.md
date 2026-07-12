# Statistical tests (eval_split=val)

## Wilcoxon signed-rank on paired per-class AP (seed-averaged)

| variant_a             | variant_b           |   n_classes |   mean_diff |   median_diff |   statistic |   p_value | scope   | eval_split   |
|:----------------------|:--------------------|------------:|------------:|--------------:|------------:|----------:|:--------|:-------------|
| aug_selective_inpaint | aug_uniform_inpaint |          16 |      0.0056 |       -0.0053 |     60.0000 |    0.7057 | all     | val          |
| aug_selective_inpaint | aug_uniform_inpaint |           5 |      0.0005 |       -0.0019 |      7.0000 |    1.0000 | tail    | val          |
| aug_selective_inpaint | basic_aug           |          16 |      0.0197 |        0.0015 |     55.0000 |    0.5282 | all     | val          |
| aug_selective_inpaint | basic_aug           |           5 |      0.0224 |        0.0478 |      6.0000 |    0.8125 | tail    | val          |

## Macro AP over seeds: mean ± std with t-distribution CI

| experiment            | scope   | eval_split   |   n_seeds |   macro_ap_mean |   macro_ap_std |   ci_low |   ci_high |   confidence |
|:----------------------|:--------|:-------------|----------:|----------------:|---------------:|---------:|----------:|-------------:|
| aug_copy_paste        | all     | val          |         1 |          0.1529 |         0.0000 |      nan |       nan |       0.9500 |
| aug_oversample        | all     | val          |         1 |          0.1541 |         0.0000 |      nan |       nan |       0.9500 |
| aug_rfs               | all     | val          |         1 |          0.1208 |         0.0000 |      nan |       nan |       0.9500 |
| aug_selective_inpaint | all     | val          |         1 |          0.1405 |         0.0000 |      nan |       nan |       0.9500 |
| aug_uniform_inpaint   | all     | val          |         1 |          0.1349 |         0.0000 |      nan |       nan |       0.9500 |
| basic_aug             | all     | val          |         1 |          0.1208 |         0.0000 |      nan |       nan |       0.9500 |
| real_only             | all     | val          |         1 |          0.1491 |         0.0000 |      nan |       nan |       0.9500 |
| aug_copy_paste        | tail    | val          |         1 |          0.1637 |         0.0000 |      nan |       nan |       0.9500 |
| aug_oversample        | tail    | val          |         1 |          0.1957 |         0.0000 |      nan |       nan |       0.9500 |
| aug_rfs               | tail    | val          |         1 |          0.1105 |         0.0000 |      nan |       nan |       0.9500 |
| aug_selective_inpaint | tail    | val          |         1 |          0.1329 |         0.0000 |      nan |       nan |       0.9500 |
| aug_uniform_inpaint   | tail    | val          |         1 |          0.1324 |         0.0000 |      nan |       nan |       0.9500 |
| basic_aug             | tail    | val          |         1 |          0.1105 |         0.0000 |      nan |       nan |       0.9500 |
| real_only             | tail    | val          |         1 |          0.1679 |         0.0000 |      nan |       nan |       0.9500 |
