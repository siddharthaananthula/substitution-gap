# Decisions

What was chosen, what was rejected, why, and what failed. Newest at the bottom. Each entry has a status: **Active**, **Superseded** (links to the replacement) or **Killed**.

---

## D001: Candidate substitutes come from behaviour, not attribute similarity
**Date:** 25 Sep 2026 · **Status:** Active

**Decision:** Products A and B are candidate substitutes when the same shopper buys each of them on different orders but rarely buys both in the same order. This is aggregated across shoppers, with a minimum evidence threshold.

**Rejected:** Similarity from name embeddings or shared aisle/brand.

**Why:** Similarity is the objective this project argues against, so it can't also be the candidate generator.

**Hypothesis to test in EDA:** the same brand in a different variety behaves more like a complement (bought together) than a substitute.

**Open:** Setting the evidence threshold, and how to handle thin-evidence pairs.

---

## D002: Acceptance labels from revealed behaviour, not a rule-based simulator
**Date:** 25 Sep 2026 · **Status:** Superseded by D013

**Decision:** A positive label is recorded when a shopper's regular product is absent from an order and a product from its substitute set is present. A negative label is recorded when the regular product is absent and no substitute is present, but the shopper buys the regular product again on a later order.

**Rejected:** Simulating accept/reject with hand-written rules, such as "same type, different brand, 80% accept".

**Why:** A model trained on rule-generated labels learns the rules back. The metrics would look strong and prove nothing.

**Known weakness:** This is a proxy. An absence may reflect choice rather than stock, and the negative labels are noisy. This is stated in the case study.

---

## D003: Out-of-stock events are simulated, and labels are not
**Date:** 25 Sep 2026 · **Status:** Active

**Decision:** Out-of-stock events are sampled for the evaluation harness, and their rate is varied by department. Acceptance labels still come from D002.

**Why:** The simulation only decides *when* a substitution is needed. It never decides whether the substitution was accepted.

---

## D004: Attribute blocking uses a hand-checked lexicon on product names
**Date:** 25 Sep 2026 · **Status:** Active

**Decision:** Dietary and flavour attributes (gluten free, lactose free, organic, vegan, sugar free, and flavour words) are extracted from product names using a small lexicon, and a sample is checked by hand.

**Rejected:** Full attribute extraction with an LLM or NER, which is out of scope for a four-week build.

**Why:** Instacart has no structured attributes. Coverage will be incomplete, and the precision of the lexicon is reported.

---

## D005: Dollar figures rest on stated assumptions
**Date:** 25 Sep 2026 · **Status:** Active

**Decision:** Unit prices, margin, and the cost of a refund versus an accepted substitute are set as explicit parameters, and a sensitivity range is reported.

**Why:** The dataset has no prices. Presenting assumed figures as measured would be wrong.

---

## D006: Killed: customers pre-approve substitutes at checkout
**Date:** Early to mid Sep 2026 · **Status:** Killed

**Why:** At fulfilment centre volume, nothing can wait for a customer response.

**Replaced by:** inferring each customer's tolerance from their own accept and reject history.

---

## D007: Killed: never charge more than the original item
**Date:** Early to mid Sep 2026 · **Status:** Killed

**Why:** This is already standard practice, so it adds nothing.

**Replaced by:** treating picker overrides as a training signal (to be designed).

---

## D008: Public data only
**Date:** 25 Sep 2026 · **Status:** Active

**Decision:** The project uses Instacart Market Basket data only, and all assumptions are documented. No employer or proprietary data is used, ever.

---

## D009: Data source is a Kaggle mirror, not the original competition page
**Date:** 26 Sep 2026 · **Status:** Active

**Decision:** The data comes from the psparks mirror on Kaggle.

**Why:** Easier access than the competition page, and it has all six original files at the full size. The "Simplified" Instacart dataset was rejected because it had already been modified by someone else. Starting from the complete raw data means every cleaning and transformation step is my own and is documented.

**Check done:** All six expected files are present (aisles, departments, orders, order_products__prior, order_products__train, products). Column names are to be confirmed in Mission 1.1.

---

## D010: Python 3.11 in a conda environment, outside OneDrive
**Date:** 26 Sep 2026 · **Status:** Active

**Decision:** The project uses a conda environment `subgap` on Python 3.11, located in `C:\Users\siddhu\projects`.

**Rejected:** A venv built on Anaconda's base Python 3.8.

**Why:** On Python 3.8, pip install failed because some packages no longer ship pre-built versions for 3.8 (it reached end of life in 2024), and building from source failed. 3.11 is supported by every tool planned for this project, and it will match the Docker image used for deployment.

## D011: Time-based split: build on prior orders, evaluate on each customer's final order
**Date:** 2 Oct 2026 · **Status:** Active

**Decision:** Regulars, substitute pairs and labels are built from `prior` orders only. Rankers are evaluated on each customer's final order (`train`, 131,209 customers). `test` orders (75,000) are ignored because their products are hidden.

**Rejected:** A random split of orders.

**Why:** A random split puts some of a customer's later orders into training and earlier ones into testing, so the model learns from the future and looks better than it is. In production, predictions only ever go forward in time.

---

## D012: A "regular" product needs 12+ prior orders and a 50% purchase share
**Date:** 2 Oct 2026 · **Status:** Active

**Decision:** A customer qualifies with at least 12 prior orders. A product is a regular for them if it appears in at least 50% of those orders.

**Evidence:**

| Min orders | Customers kept | Purchases kept |
|---|---|---|
| 6 | 71% | 93% |
| 8 | 59% | 88% |
| 10 | 49% | 83% |
| **12** | **42%** | **78%** |

At a 25% share, a typical customer had 8 regulars (median), the size of a whole basket (median basket is also 8). At 50%, they had 2.

**Why:** Dropped customers are light shoppers, so 12 orders loses a fifth of purchases in exchange for stable share estimates. At 25%, most "missing" regulars would just be normal shopping (tissues missing from 15 of 20 orders), which floods the data with false signals.

**Possible improvement:** judge "missing" against each product's own buying cycle (is it overdue?), which would handle monthly items like tissues without lowering the threshold.

---

## D013: Per-order acceptance labels dropped; replaced by pair-level lift
**Date:** 2 Oct 2026 · **Status:** Active. **Supersedes D002.**

**Test (2,000 sampled customers, same customers for every comparison):**

| Regular is... | Same-aisle product in order | First-time same-aisle product |
|---|---|---|
| Missing | 36% | 13.1% |
| Bought (control) | 45% | 10.6% |

**Finding:** A missing regular usually means the customer skipped that aisle, not that they swapped. First-time purchases show a real but weak signal: about 2.5 extra swaps per 100 missing events, so roughly 80% of per-order positive labels would be noise.

**Decision:** Score each product pair (A, B) across all customers instead: how much more often B is bought for the first time when A is missing than when A is bought (lift). A weak signal per event becomes strong evidence in aggregate.

**Caveat:** Events from the same customer are not independent. A proper significance test would bootstrap by customer.

---

## D014: The sample is too sparse for pair-level lift; run on full data in batches
**Date:** 5 Oct 2026 · **Status:** Active

**Finding:** On the 2,000-customer sample, 182,632 candidate pairs were found, but 94% were seen only once and only about 0.6% reached 5 sightings.

**Decision:** Develop on the sample, run on the full data. Customers are split into 10 batches, counts are computed per batch and summed, and lift is calculated once from the totals.

**Why batches:** The laptop has 8 GB. Counts add up exactly across batches; rates do not, so rates are never averaged.

---

## D015: Rank by an  Adjusted lift_low above  Pairs  Regular products covered (of 2,280 with any candidate pair; 15,178 regulars in total)
**Date:** 9 Oct 2026 · **Status:** Active
**Method:**
- Minimum evidence: B seen at least 5 times and A missing at least 30 times.
- +1 smoothing on the bought side, so pairs never seen when A was bought don't get infinite lift.
- Rank by the lower end of a 95% confidence interval on log-lift (`lift_low`), so a pair needs both a strong effect and enough evidence.
- Divide by an exploration ratio of 1.23: customers buy 23% more first-time products whenever a regular is missing (4.17 vs 3.39 per event), which pushed every pair's lift up. After adjusting, the median candidate lift fell from 1.35 to 1.10. The remaining gap is likely the winner's curse from the evidence filter.

**Results (full data, 32.4M lines):** 209,725 candidate pairs passed the evidence filter.

| Adjusted lift_low above | Pairs | Regular products covered (of 2,280) |
|---|---|---|
| 1 | 5,928 | 973 (43%) |
| **2** | **423** | **231 (10%)** |

**Decision:** Use adjusted lift_low > 2 as the confident substitute set.

**Why:** A wrong substitute leads to a refund, which is worse than no suggestion. In a sample of 10 pairs with unadjusted lift_low between 1 and 1.5, only 1 or 2 looked like real substitutes.

**Observations:**
- Top pairs are organic vs conventional or loose vs bagged versions of the same product (e.g. Banana to Bag of Organic Bananas: 2,404 sightings, lift 61.6), and most hold in both directions.
- Lift well below 1 marks complements (Banana Bread Mix vs Banana, lift 0.33), supporting the D001 hypothesis. Complements should never be offered as substitutes.
- Some raw-lift top pairs looked like catalogue duplicates (Water Mineral vs Mineral Water). They fell away once ranked by lift_low.

**Known limitation:** Only 10% of regular products have a confident substitute, almost all fresh produce and milk.

## D016: Evaluation design and results: behaviour first, popularity as fallback
**Date:** 12 Nov 2026 · **Status:** Active

**Evaluation set:** each customer's final order (`train`), never used to build regulars or pairs. An event is a regular product A (from prior orders) missing from the final order: 65,476 events from 29,701 customers. Customers whose final order is `test` are ignored (products hidden).

**Truth sets (first-time buys in the final order):**

| Truth | Definition | Ceiling (all events) |
|---|---|---|
| loose | any first-time buy | 78.6% |
| strict | first-time buy in A's department | 29.5% |
| aisle | first-time buy in A's aisle | 11.3% |

"Aisle" is the closest proxy for "bought instead of A". None of them is an observed accept/reject decision.

**Rankers:**
- similarity: same aisle, TF-IDF cosine on product names, character 3–4-grams (word matching missed "Banana" vs "Bananas" and would have made the baseline unfairly weak)
- popularity: same aisle, ranked by first-time buys in prior orders
- behaviour_dept: pairs with lift_low_adj > 2 (D015), restricted to A's department
- hybrid_sim / hybrid_pop: behaviour_dept candidates first, remaining slots filled from similarity / popularity

**Department filter:** without it, Banana's behavioural candidates included Coconut Cream, Ice and Belgian Lager. These are shopping-occasion effects (a party shop skips bananas), not substitutes. Removing all cross-department pairs cost zero hits and 0.4 points of coverage.

**Results (hit@k, all 65,476 events, aisle truth):**

| k | similarity | popularity | behaviour_dept | hybrid_sim | hybrid_pop |
|---|---|---|---|---|---|
| 1 | 0.2% | 0.6% | 0.6% | 0.7% | 0.7% |
| 5 | 1.1% | 2.2% | 0.9% | 1.4% | 2.3% |

- On the 25,582 events both cover, behaviour_dept beat similarity at k = 1: 1.5% vs 0.4%, difference +1.1 points [95% CI +1.0, +1.3].
- Putting behavioural candidates first improved both baselines on all events: hybrid_sim − similarity +0.4 points at every k (CI excludes zero); hybrid_pop − popularity +0.1 to +0.2 points (aisle truth at k = 3 and 5: lower bounds +0.01 and +0.02, small but real).
- hybrid_pop is the best ranker under every truth set and k.
- CIs: 200 bootstrap resamples of customers (not events), seed 42, paired on the same events.

**Decision:** Use hybrid_pop as the current best ranker and the baseline for the acceptance model. Similarity ranks worst of all five, which supports the project's core claim that "looks similar" is the wrong objective.

**Limitations:** absolute hit rates are low (under 2.5%) because truth is a proxy built from first-time buys, not observed substitution outcomes. Behaviour covers only 39% of events and usually has one candidate per product. Catalogue duplicates (e.g. "Organic  Whole Milk" with a double space) appear as separate products.

---

## D017: Warehouse per project: DuckDB + dbt here, Snowflake in Promo Effectiveness
**Date:** 12 Nov 2026 · **Status:** Active

**Decision:** The Substitution Gap's data layer uses DuckDB with dbt (dbt-duckdb). Snowflake + dbt is planned for Project 2 (Promo Effectiveness). The retail analytics project already uses Postgres.

**Rejected:** Snowflake for this project.

**Why:**
- The heavy computation here is in Python. The API only reads one pre-built table, so a database server adds cost and moving parts for no benefit.
- DuckDB runs inside the app container: no second container, no trial expiry, and the deployed demo keeps working.
- Snowflake's free trial is time-limited. It suits a SQL-first project (promo lift, cannibalisation, margin ledger) built in one focused window.
- Three projects, three data platforms (Postgres, DuckDB, Snowflake), each chosen for a reason.

**Limit:** DuckDB is not built for many concurrent writers. That doesn't matter for a read-only serving table.

## D018: Acceptance model: LightGBM ranker beats hybrid_pop, and the gain comes from behaviour features
**Date:** 12 Nov 2026 · **Status:** Active

**Leakage design:**
- Training label = each customer's LAST PRIOR order. Training features are rebuilt from the prior orders before it (regulars, pairs, lift, popularity).
- Evaluation is unchanged (D016): features from all prior orders, truth from the final order.
- Enforced in code: `assert_prior_only` fails the run on any final-order ID, and `no_final_orders()` blocks reads of the final-order file, the evaluation folder and the all-prior pairs file during training. The all-prior pairs file is blocked because it was built using the training label orders.

**Training data:**
- Training events: 96,448, but only 5,958 have a positive. Trained on those only, because lambdarank learns nothing from groups with all-zero labels.
- Candidate pool per event: top 20 similarity + top 20 popularity in A's aisle + behaviour pairs with lift_low_adj > 1. Aisle-truth pool recall: 6.5% of evaluation events (57% of the 11.3% ceiling). This is the model's hard upper limit.
- Label: B is a first-time buy in A's aisle.

**Model:**
- LightGBM LGBMRanker (lambdarank), 31 leaves, learning rate 0.05.
- Number of trees chosen by 5-fold GroupKFold by customer (median best iteration), then refit on all positive training events.
- A single early-stopping split gave 8 trees. Across folds the best iteration ranged from 8 to 51, so the single split was replaced.

**Results (all 65,476 evaluation events, aisle truth, 95% customer-bootstrap CIs):**

| k | hybrid_pop | model | model without behaviour features |
|---|---|---|---|
| 1 | 0.75% | 0.88% | 0.80% |
| 5 | 2.29% | 2.66% | 2.33% |

- Model − hybrid_pop: +0.13 pts [+0.07, +0.19] at k = 1; +0.37 pts [+0.29, +0.47] at k = 5 (16% relative).
- Model − model without behaviour: +0.07 [+0.01, +0.13] at k = 1; +0.33 [+0.24, +0.42] at k = 5. Significant at every k.
- Model without behaviour − hybrid_pop: not significant at any k. A single-split run had suggested otherwise, but it did not survive cross-validation.
- Seed stability (seeds 42, 7, 123): model hit@5 2.63–2.66%, a spread of about 0.02 pts, far smaller than the gaps.

**Interpretation:** popularity, similarity and structure alone only match a simple behaviour-first rule. Adding behavioural lift as model features is what beats it. This is the project's core claim, shown in a learned model.

**Feature importance (gain):** same_aisle 46% (partly by construction: the label requires same aisle), lift_low_adj about 16%, aisle popularity rank about 17%. Name similarity about 3%.

**Findings to keep:**
- Organic features added almost nothing (0.2–0.3% of gain): no sign in this data that customers avoid organic/conventional switches. The dietary hard-block lever (D004) is not supported by this proxy and would need real accept/reject data.
- Truth labels are noisy: one Organic Hass Avocado customer's actual first-time buys were Asian Pears and Granny Smith Apples.

**Limitations:**
- Absolute hit rates are low (under 3%) because truth is a proxy, not observed acceptance, and the pool caps reachable events at 6.5%.
- The intervals resample customers, not training randomness; seed spread was checked separately.
- Pool size was kept at N = 20 per source.

**Rejected:** hyperparameter search, which would risk tuning until the model wins.