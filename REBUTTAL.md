# Author responses to the LoG 2026 reviews (submission #127)

This document accompanies the OpenReview author rebuttal, whose length limit (2,500 characters) cannot hold point-by-point responses to four reviews. Every number here comes from the paper or from the rebuttal runs in this repository (scripts/03_rebuttal_runs.py, results/rebuttal.csv, results/rebuttal_importance.csv, logs/rebuttal_runs.log); the r=0.40 rerun reproduces all 12 published YelpChi/Amazon cells to 5 decimals.

## Reply to Reviewer YiMm (rating 7)

We thank the reviewer for the careful reading and the constructive pointers, especially on related work.

**Point 1 (strength of node features).** We agree this bounds the scope of the conclusion. The design is deliberate: we ask a question one step cheaper than model or representation choice, holding the input at the benchmark-provided features that practitioners actually start from. Section 5 already reserves message-passing rungs for the archival version, and we will consider learned-encoder rungs there as well. We would add one clarification: the two leakage mechanisms are representation-independent. Any pipeline that consumes walk-count label features inherits the self-walk term, whether the surrounding representation is raw features or a pretrained encoder, and on Elliptic no representation can recover label signal from test nodes that have zero train-labeled reachability. We verified the latter by execution: 0 of 234,355 edges cross timesteps, so every test node is unreachable from train labels under any train window ending before the test period. Stronger representations could change the size of the static-feature gains, but not these two findings.

**Point 2 (scope of the conclusion).** We agree and will comply exactly as suggested. The limitations paragraph in Section 5 already states the scope (three benchmarks, one split family per dataset), but the abstract and the Section 3 heading outrun it. We will rephrase both to claim the result only for the evaluated benchmark settings, i.e. "on these three benchmarks at the tested label rates." These are sentence-level edits and fit the 4-page format.

**Point 3 (missing related work on label leakage).** This is a fair and valuable catch, and we will cite all three works. To be fully candid about the relationship: SeHGNN already prevents self-label leakage inside its HGNN pre-computation by removing the diagonal of the label-propagation matrix, and Echoless-LP names exactly the returning-walk phenomenon we describe ("echo") and contributes a memory-efficient partition-based alternative to exact diagonal removal; UniMP addresses the same risk in an end-to-end model via stochastic masked label prediction. Our contribution is therefore not the discovery that self-inclusion exists, and we will position it accordingly. What our paper adds is: (i) showing the same diag(A²) self-inclusion silently inflates label-derived walk-count features in feature-engineered tabular pipelines, where exact removal is cheap, and deriving the exact degree-weighted correction there; (ii) quantifying what the leak was worth under honest evaluation, including the observation that the leaky variant passes standard train-test methodology and distribution checks, which is what makes it dangerous in practice; and (iii) the separate temporal-unreachability mechanism, which none of the three works consider. We will add a short related-work passage drawing these distinctions, which strengthens rather than diminishes the framing.

We hope this addresses the concerns and are happy to clarify further at any point, in this rebuttal round or the discussion phase.

---

## Reply to Reviewer od4H (rating 7)

We thank the reviewer for the careful reading and the constructive suggestions. We respond point by point.

**1. Narrowing generality.** We agree. Section 5 already limits the study to three benchmarks and one split family per dataset, but the abstract and Section 3 outrun that paragraph. We will rephrase both to scope the claims to the evaluated benchmarks, and replace the abstract's "moderate label rates" with the explicit tested range (r in {0.05, 0.1, 0.2, 0.4}, given the new sweep below).

**2. Lower label rates.** We ran the suggested sweep: r in {0.05, 0.1, 0.2, 0.4} on YelpChi and Amazon, both models, 5 seeds, same pipeline, loaders, and hyperparameters as the paper. The null result persists at every rate: across all rates, both datasets, and both models, the label-feature increment b minus b0 lies within [-0.0083, +0.0033]. LightGBM (AUPRC, mean over 5 seeds; seed stds 0.003 to 0.025):

| r | YelpChi b0 | YelpChi b | Amazon b0 | Amazon b |
|---|---|---|---|---|
| 0.05 | 0.7060 | 0.7020 | 0.8805 | 0.8785 |
| 0.10 | 0.7849 | 0.7812 | 0.9045 | 0.9035 |
| 0.20 | 0.8495 | 0.8451 | 0.9157 | 0.9165 |
| 0.40 | 0.8939 | 0.8920 | 0.9302 | 0.9335 |

XGBoost gives the same picture. We should be candid: this refutes our own Section 5 conjecture that label propagation "should finally earn its keep" at low rates. In the tested range it does not; on YelpChi b minus b0 is, if anything, slightly negative at low rates. We also added a non-nested rung aL (raw plus label features only): on Amazon label features alone do beat raw (positive in all 5 seeds for both models at r ≥ 0.10, up to +0.021 at r = 0.40) but are redundant once static aggregates are present, while on YelpChi they add nothing over raw at any rate (all deltas within seed noise). The script and raw CSVs will be added to the released anonymous repository, and the revision will report the sweep compactly.

**3. Elliptic generalization.** The generalizable artifact is the reachability diagnostic (Table 2), not the Elliptic zero itself. Elliptic is the extreme case: 0 of its 234,355 edges cross timesteps, so test nodes have exactly zero train-labeled reachability. We will add a sentence stating the graded version: deployment-style splits on temporally localized graphs shrink train-labeled reachability, the diagnostic measures the degree, and Elliptic's total unreachability is the worst case.

**4. Statistical analysis.** Per-seed paired contrasts make the conclusions unambiguous, and we will report them in the revision. The static gain b0 minus a is positive in every seed at every rate on both datasets and is many seed-stds wide (e.g. YelpChi r=0.05, LightGBM: +0.091). The label increment b minus b0 sits within seed noise; e.g. YelpChi XGBoost at r=0.40 it is negative in 5 of 5 seeds. On Elliptic the temporal split is fixed and the GBDT fits are deterministic, so all cells are bit-identical across 5 model seeds; Table 1's ±0.000 is literal determinism, not a measured spread. We will replace it with a clear note and drop the zero-length error bars in Figure 2.

**5. Learned graph representations.** This coincides with the message-passing rungs Section 5 reserves for the archival version, and we commit to adding at least one there. We scoped this abstract to GBDTs because GADBench finds tree ensembles competitive with or superior to GNNs on these benchmarks, and Vandervorst et al. outperform HinSAGE, HAN, and HGT with gradient boosting over neighborhood features. Both leakage mechanisms apply to any model consuming label-derived features, learned or not.

---

## Reply to Reviewer uDZK (rating 4)

We thank the reviewer for a careful and accurate review. Nearly every factual observation is correct, and we respond to each with a concrete revision or with new data: all new numbers below come from runs we completed during the rebuttal window with the paper's exact pipeline, loaders, and hyperparameters (5 seeds unless noted), and the script and raw CSVs will be added to the released anonymous repository. The audit we ran for this response also surfaced errata in our own text, self-reported below.

**W1.** We agree on framing and will present Section 3 as a controlled replication on three fraud benchmarks, citing [1] and [2]. The precise relationship: [1] aggregates node attributes for MLPs on standard node-classification benchmarks, with no fraud datasets, no label-derived features, and no leakage analysis; [2] evaluates one-hop attribute aggregation (NFA) for GBDTs and treats temporal splits as distribution shift, again with no leakage mechanisms. The contribution we claim is not the static rung but the marginal decomposition built on it and the two mechanisms in Section 4, which neither work studies. While rechecking this section we found an erratum of our own, which we will correct: 72 of Elliptic's 165 raw features are aggregates (93 local plus 72 aggregated after dropping txId and timestep), not 71.

**W2.** Correct, and we will add the full specification to Section 2. All feature aggregates use the binarized, symmetrized, self-loop-free homogeneous adjacency with row normalization: the one-hop mean D⁻¹Ax; the one-hop squared mean D⁻¹Ax², a cheap dispersion proxy chosen in place of a slow sparse per-feature max (with the mean it lets trees recover variance); and the two-hop mean (D⁻¹A)²x, a walk-weighted mean with backtracking, that is, a walk statistic rather than a 2-hop-ball set statistic. For raw attributes the walk semantics is benign precisely because no label enters, which is the distinction Section 4.1 draws. Only the log-degrees are per-relation; feature aggregates are homogeneous-only. Isolated nodes (13 in YelpChi) receive zero aggregates. The raw features are dense numeric columns aggregated uniformly: 4 of 32 are binary in YelpChi, 1 of 25 in Amazon, 0 in Elliptic; the mean of a binary column is a neighborhood proportion, and no arbitrarily coded multi-category columns exist in these benchmarks.

**W3.** Fully conceded. Our answer to how much graph features help is conditional on the CIKM 2020 graph construction for YelpChi and Amazon (Elliptic's edges are native payment flows), and the paper never says so. We will add an explicit sentence to Section 5 stating this conditionality and naming construction as an uncontrolled variable, plausibly larger than the aggregation choices of W2. The pointers are well taken: AutoG [5] (an ICLR 2025 paper, we note) and the task-aware construction line [6] study construction as the object of design, the construction-side counterpart of our feature-side question, and we will cite both.

**W4.** The reviewer is right: a strictly nested ladder cannot separate "label features carry no signal" from "label features carry signal the static aggregates already capture." We ran the missing rung, aL = raw plus the four label-propagation features only, over the full grid (both models, r in {0.05, 0.1, 0.2, 0.4}, 5 seeds); the rerun first reproduced all 12 published YelpChi and Amazon rung cells to 5 decimals. LightGBM deltas, means over 5 seeds:

| r | YelpChi aL-a | YelpChi b-b0 | Amazon aL-a | Amazon b-b0 |
|---|---|---|---|---|
| 0.05 | -0.004 | -0.004 | +0.012 | -0.002 |
| 0.10 | -0.001 | -0.004 | +0.012 | -0.001 |
| 0.20 | +0.001 | -0.004 | +0.019 | +0.001 |
| 0.40 | -0.001 | -0.002 | +0.021 | +0.003 |

The refined conclusion is dataset-dependent. On Amazon, label features alone do carry signal: aL beats (a) in every one of the 5 seeds for both models at r ≥ 0.10 (at r = 0.40, 0.925±0.006 vs 0.904±0.005 with LightGBM), yet b-b0 never exceeds +0.0033, so there label features are redundant given the static aggregates. On YelpChi, aL adds nothing over (a) at any rate (mean deltas between -0.0141 and +0.0020 across both models): uninformative outright. On Elliptic, aL sits below (a) (0.778 vs 0.791 LightGBM; 0.779 vs 0.787 XGBoost): the dead constant features actively hurt. The operative conclusion, that label features add nothing on top of static aggregates, holds in every case and extends down to r = 0.05, but the reviewer's distinction is real, and the revision will say "redundant on Amazon, uninformative on YelpChi, harmful on Elliptic" instead of the current blanket phrasing. An honest note: the low-rate results also contradict our own Section 5 conjecture that label propagation "should finally earn its keep" at low rates; we will revise that sentence too.

**W5.** Conceded. The Elliptic split is fixed and temporal, and the near-default GBDT configurations are deterministic (no row or column subsampling), so the fits do not vary with the model seed: in the new runs we verified every Elliptic cell is bit-identical across all 5 seeds. Table 1's "±0.000" entries and Figure 2's apparent error bars, which are zero-length matplotlib bars whose caps render as ticks, are misleading presentation of a deterministic quantity. We will replace those entries with a dash, drop the bars on that panel, and say so in both captions. For genuine uncertainty on the fixed split we will add test-set bootstrap confidence intervals to the released results, with a one-line summary in the revision; we note the effects carrying the temporal mechanism (7.5 LightGBM and 14.9 XGBoost points, leaky (b) vs (b0)) dwarf the seed-level noise everywhere else in Table 1 (stds 0.002 to 0.008).


**Q1.** Answered in full under W2 (neighbor set, one- and two-hop semantics, isolated nodes, column types); the specification will be added to Section 2.

**Q2.** The validation window is timesteps 30-34 (3,513 labeled nodes), held out by the harness but never consulted: there is no early stopping, and hyperparameters are fixed across rungs. Self-reported erratum: the paper's "train t ≤ 34" is imprecise; the fit uses t ≤ 29 (26,381 labeled nodes), with test t ≥ 35 (16,670). We will correct Section 2. The temporal-unreachability mechanism is unaffected: 0 of 234,355 edges cross timesteps, so test nodes have zero train-labeled neighbors under any train window ending before t = 35.

**Q3.** Yes. Gain-importance shares of the label-feature group (share of total split gain, seed 0, LightGBM / XGBoost):

| Dataset | rung (b) | (b) leaky |
|---|---|---|
| YelpChi | 0.050 / 0.053 | 0.120 / 0.130 |
| Amazon | 0.044 / 0.043 | 0.087 / 0.053 |
| Elliptic | 0.252 / 0.505 | 0.943 / 0.940 |

The leak inflates reliance on label features everywhere, and on Elliptic the leaky models stake about 94 percent of their split gain on features that are constants at test time (Table 2). That is exactly why the leak's damage explodes there (the 7.5 and 14.9 points above) while staying moderate on YelpChi (3.6 points) and Amazon (0.5). We will add these shares to Section 4.1, turning its "severity tracks reliance" sentence from a plausibility argument into a measurement.

**Q4.** Rung (b0) adds 1+R log-degrees (R relations) plus 3F neighborhood aggregates; rung (b) adds 4 label features. Totals: YelpChi 32 → 132 → 136; Amazon 25 → 104 → 108; Elliptic 165 → 661 → 665. The new aL rung has 36, 29, and 169 features respectively. We will add the counts to Section 2.

**Minor comments.** All accepted. We will rephrase Section 3 to "label features earn nothing on YelpChi and Amazon and slightly hurt on Elliptic," consistent with Section 4.2; replace the abstract's "at moderate label rates" with the explicit tested range, now r in {0.05, 0.1, 0.2, 0.4} given the sweep above; scope the Figure 1 caption's r = 0.40 to YelpChi and Amazon, since Elliptic uses its temporal split; define M1/M2 at first use and renumber them to match the abstract's order (our audit found the section tags currently invert it); relabel the rungs a/b/c with a mapping note in the released code; and replace "transaction graph" with "graph" in the abstract's first sentence. We will cite Grables [7], which poses our question at the general tabular-learning level, and add one introduction sentence mapping the title's "when" onto the two conditions the paper isolates, split family and label rate. FDB [3] is a good suggestion for breadth: its datasets ship without native graphs, so evaluating the ladder there makes graph construction itself the variable under test, exactly the W3 concern, and we flag it as the natural next experiment for the archival version.

In sum, the two experimental gaps the review identified are now closed with data that sharpen the findings rather than overturn them. The remaining points are specification, scoping, and presentation fixes: the sentence-level edits fit the extended-abstract format, and the full sweep tables, importance shares, and bootstrap intervals will live in the released repository with summary sentences in the revision. We would be glad to run further checks or answer follow-up questions at any point, in this rebuttal round or the discussion phase; we are monitoring the forum and will respond promptly.

---

## Reply to Reviewer FMA3 (rating 6)

Thank you for the careful reading and for finding the diagnostic observations useful. We respond to the three concerns in order.

**C1 (methodology is simple).** We agree, and the simplicity is the design. Because the ladder is strictly nested and every rung feeds the same two GBDTs with identical near-default hyperparameters, differences between rungs reflect information rather than tuning, so the attribution of gains to a feature family is clean. Where the nested design alone could not settle a question, namely how to read the null on label features, we added a non-nested rung during the rebuttal window (detailed in our response to Reviewer uDZK). Both mechanisms were surfaced precisely because the pipeline is this auditable: the self-walk term is visible in a closed-form feature definition, and the temporal-unreachability zero is a one-line diagnostic. We believe this is the appropriate scale for the extended-abstract track, with richer machinery reserved for the archival version (Section 5).

**C2 (no GNN baseline).** Correct as a fact, and we will keep the claims scoped accordingly. One clarification: the paper never claims richer graph models are unnecessary. Its claims are about feature families, which is orthogonal to model choice. For why a GBDT baseline is strong on these benchmarks we rely on the external evidence already cited: GADBench finds tree ensembles competitive with and often superior to GNNs on these datasets, and Vandervorst et al. outperform HinSAGE, HAN, and HGT with gradient boosting over neighborhood features. Section 5 pre-registers message-passing rungs for the archival version. We also note that both mechanisms apply to any model consuming label-derived features, GNN or not.

**C3 (only a 40% label rate).** We agree this was the main evidential gap, so we ran the sweep: r in {0.05, 0.1, 0.2, 0.4} on YelpChi and Amazon, same pipeline, loaders, and hyperparameters as the paper, 5 seeds. The null result persists at every rate: the label-rung increment (b) minus (b0) lies within [-0.0083, +0.0033] across both datasets, all four rates, and both models, while static graph features still deliver the gains everywhere, e.g. YelpChi at r=0.05: 0.615 to 0.706 AUPRC for LightGBM. YelpChi, LightGBM, mean ± std over 5 seeds:

| r | (a) raw | (b0) +static | (b) +label |
|---|---------|--------------|------------|
| 0.05 | 0.615 ± 0.016 | 0.706 ± 0.023 | 0.702 ± 0.025 |
| 0.10 | 0.704 ± 0.010 | 0.785 ± 0.010 | 0.781 ± 0.011 |
| 0.20 | 0.773 ± 0.008 | 0.850 ± 0.006 | 0.845 ± 0.007 |
| 0.40 | 0.828 ± 0.003 | 0.894 ± 0.004 | 0.892 ± 0.004 |

An honest note: Section 5 conjectured label propagation "should finally earn its keep" at low rates; the sweep does not bear this out, and we will revise that sentence. Elliptic is excluded by design: it uses a fixed temporal split under which no label rate can help, since test nodes have zero train-labeled reachability (Table 2). The script and raw CSVs will be added to the released anonymous repository.

We would be glad to clarify anything further at any point, in this rebuttal round or the discussion phase.
