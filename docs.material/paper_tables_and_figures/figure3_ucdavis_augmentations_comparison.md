# Figure 3: Critical distance plot of the accuracy obtained with each augmentation for the 32x32 and 64x64 cases.

[:simple-jupyter: :material-download:](../../paper_tables_and_figures/figure3_ucdavis_augmentations_comparison/figure3_ucdavis_augmentations_comparison.ipynb)


```python
import pandas as pd
```


```python
import matplotlib as mpl
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format='retina'
```


```python
import pathlib

import autorank
```


```python
df = pd.read_parquet(
    "./campaigns/ucdavis-icdm19/augmentation-at-loading-with-dropout/campaign_summary/1684447037/merged_runsinfo.parquet"
)
```


```python
def prepare_data(dat, test_split):
    res = dat.query("test_split_name == @test_split & flowpic_dim != 1500")
    res = res[["aug_name", "split_index", "flowpic_dim", "seed", "acc"]]
    res["id"] = (
        "split_index"
        + res["split_index"].astype(str)
        + "_seed"
        + res["seed"].astype(str)
        + "_flowpicdim"
        + res["flowpic_dim"].astype(str)
    )
    res = res[["aug_name", "id", "acc"]]
    return res.sort_values(["aug_name", "id"])


def get_ranks(df, test_split, force_ranks=True):
    df1 = prepare_data(df, test_split)
    df1 = df1.pivot(columns="aug_name", index="id").reset_index(drop=True)
    df1.columns = df1.columns.get_level_values(1)
    new_df = pd.DataFrame(
        {
            "changertt": df1["changertt"].values,
            "colorjitter": df1["colorjitter"].values,
            "horizontalflip": df1["horizontalflip"].values,
            "noaug": df1["noaug"].values,
            "packetloss": df1["packetloss"].values,
            "rotate": df1["rotate"].values,
            "timeshift": df1["timeshift"].values,
        }
    )
    replacement = {
        "noaug": "No augmentation",
        "horizontalflip": "Horizontal flip",
        "rotate": "Rotate",
        "timeshift": "Time shift",
        "colorjitter": "Color jitter",
        "changertt": "Change RTT",
        "packetloss": "Packet Loss",
    }
    new_df = new_df.rename(columns=replacement)
    if force_ranks:
        return autorank.autorank(new_df, force_mode="nonparametric")
    else:
        return autorank.autorank(new_df)
```


```python
res_script = get_ranks(df, "test-script", force_ranks=True)
res_human = get_ranks(df, "test-human", force_ranks=True)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 4))
ax = axes[0]
autorank.plot_stats(res_script, ax=ax)
ax.set_title("Test-script")

ax = axes[1]
autorank.plot_stats(res_human, ax=ax)
ax.set_title("Test-human")
plt.tight_layout()

# plt.savefig("figures/augmentations_rank_comparison.pdf", bbox_inches="tight")
```

    Tests for normality and homoscedacity are ignored for test selection, forcing nonparametric tests
    Tests for normality and homoscedacity are ignored for test selection, forcing nonparametric tests



    
![png](figure3_ucdavis_augmentations_comparison_files/figure3_ucdavis_augmentations_comparison_7_1.png)
    



```python
autorank.create_report(res_human)
```

    The statistical analysis was conducted for 7 populations with 30 paired samples.
    The family-wise significance level of the tests is alpha=0.050.
    We failed to reject the null hypothesis that the population is normal for all populations (minimal observed p-value=0.010). Therefore, we assume that all populations are normal.
    We applied Bartlett's test for homogeneity and failed to reject the null hypothesis (p=0.064) that the data is homoscedastic. Thus, we assume that our data is homoscedastic.
    Because we have more than two populations and all populations are normal and homoscedastic, we should use repeated measures ANOVA as omnibus test to determine if there are any significant differences between the mean values of the populations. However, the user decided to force the use of the less powerful Friedman test as omnibus test to determine if there are any significant differences between the mean values of the populations. We report the mean value (M), the standard deviation (SD) and the mean rank (MR) among all populations over the samples. Differences between populations are significant, if the difference of the mean rank is greater than the critical distance CD=1.644 of the Nemenyi test.
    We reject the null hypothesis (p=0.000) of the Friedman test that there is no difference in the central tendency of the populations No augmentation (MD=69.880+-1.033, MAD=1.205, MR=5.433), Color jitter (MD=69.880+-1.756, MAD=3.012, MR=4.883), Horizontal flip (MD=71.687+-1.281, MAD=1.807, MR=3.883), Change RTT (MD=71.084+-1.490, MAD=1.205, MR=3.867), Packet Loss (MD=72.289+-1.113, MAD=2.410, MR=3.633), Rotate (MD=72.289+-1.406, MAD=2.410, MR=3.467), and Time shift (MD=72.289+-1.179, MAD=1.205, MR=2.833). Therefore, we assume that there is a statistically significant difference between the median values of the populations.
    Based on the post-hoc Nemenyi test, we assume that there are no significant differences within the following groups: No augmentation, Color jitter, Horizontal flip, and Change RTT; Color jitter, Horizontal flip, Change RTT, Packet Loss, and Rotate; Horizontal flip, Change RTT, Packet Loss, Rotate, and Time shift. All other differences are significant.



```python
autorank.create_report(res_script)
```

    The statistical analysis was conducted for 7 populations with 30 paired samples.
    The family-wise significance level of the tests is alpha=0.050.
    We rejected the null hypothesis that the population is normal for the populations Horizontal flip (p=0.007), Rotate (p=0.006), Packet Loss (p=0.005), Change RTT (p=0.004), and Color jitter (p=0.000). Therefore, we assume that not all populations are normal.
    Because we have more than two populations and the populations and some of them are not normal, we use the non-parametric Friedman test as omnibus test to determine if there are any significant differences between the median values of the populations. We use the post-hoc Nemenyi test to infer which differences are significant. We report the median (MD), the median absolute deviation (MAD) and the mean rank (MR) among all populations over the samples. Differences between populations are significant, if the difference of the mean rank is greater than the critical distance CD=1.644 of the Nemenyi test.
    We reject the null hypothesis (p=0.000) of the Friedman test that there is no difference in the central tendency of the populations Horizontal flip (MD=95.333+-0.667, MAD=0.667, MR=6.250), No augmentation (MD=96.000+-0.333, MAD=0.000, MR=5.800), Rotate (MD=96.667+-0.667, MAD=0.667, MR=4.217), Time shift (MD=97.000+-1.000, MAD=0.333, MR=3.400), Packet Loss (MD=97.333+-1.000, MAD=0.667, MR=3.200), Change RTT (MD=97.333+-0.667, MAD=0.667, MR=3.033), and Color jitter (MD=97.333+-1.000, MAD=0.667, MR=2.100). Therefore, we assume that there is a statistically significant difference between the median values of the populations.
    Based on the post-hoc Nemenyi test, we assume that there are no significant differences within the following groups: Horizontal flip and No augmentation; No augmentation and Rotate; Rotate, Time shift, Packet Loss, and Change RTT; Time shift, Packet Loss, Change RTT, and Color jitter. All other differences are significant.

