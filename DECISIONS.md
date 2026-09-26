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
**Date:** 25 Sep 2026 · **Status:** Active

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
