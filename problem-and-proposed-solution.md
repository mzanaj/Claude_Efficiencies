# Filtering a Sea of Text: Problem Statement and Proposed Pipeline

*A design document for inducing a taxonomy over a broad, high-variance corpus and distilling it into a production classifier — with junk filtering as the first concrete payoff.*

---

## 1. The problem

### What we have
We can obtain a large corpus — documents in the **millions** — that is broad and high-variance: a sea of mixed-domain text with no single subject, no consistent structure, and no pre-existing labels. We do not know, up front, what categories live inside it. The taxonomy is unknown and has to be *discovered*, not assumed.

### What we want
The immediate practical motivation is a **junk filter**: reliably separate obvious low-value content (news, spam, SEO sludge, boilerplate) from whatever signal we actually care about.

Rather than build a narrow binary filter directly, we chose the more flexible path: **induce a full taxonomy of the corpus, with "junk" as one branch (or a small set of branches) inside it.** Once we have labeled data from that taxonomy, we can either ship the full multi-class classifier or collapse it into a dedicated junk-vs-keep binary model trained on the same labels. Building the broad model first and specializing later is strictly more flexible than committing to a filter at the outset, and it makes the expensive labeling pass do double duty.

### Constraints
Three constraints shape every design decision:

| Resource | Limit | Implication |
|---|---|---|
| Raw corpus | Millions of documents available | Plenty of real data to learn from — synthetic data is largely unnecessary |
| LLM processing | Only **hundreds of thousands** of docs are economically LLM-touchable | All expensive LLM work must happen on a representative *sample*, not the whole corpus |
| Production encoder | Must sustain **~1M documents/day** | The deployed classifier has to be a cheap, fast, LLM-free model |

### Why it's hard
Three things make this non-trivial. First, the **taxonomy is unknown** — we can't classify into categories we haven't discovered yet, so discovery and classification are intertwined. Second, the corpus is **broad and high-variance**, which is exactly the regime where short-text or short-descriptor clustering tends to produce vague, overlapping "blobby" clusters. Third — and most important — there is a **scale mismatch**: we must induce the taxonomy and label using LLMs on a small fraction of the data, then generalize to the full millions through a trained encoder. Making that small fraction genuinely representative, including the rare-but-important tail, is the crux of the whole effort.

---

## 2. The proposed solution

### The shape: an hourglass
The core mental model is an **hourglass**. The corpus starts wide (millions), is squeezed down to a sample small enough for expensive LLM reasoning (~200K), and then fans back out to the full millions once a cheap encoder has absorbed what the LLM produced. The expensive, intelligent work happens once, on the narrow waist; the cheap, high-throughput work happens forever, on the wide ends.

This is not a bespoke design. It mirrors the **TnT-LLM** framework (Microsoft + University of Washington, KDD 2024), a peer-reviewed pattern built for almost exactly this situation: an LLM induces and refines a taxonomy, an LLM pseudo-labels a sample, and a lightweight classifier is distilled from those labels and deployed at scale. In that work the distilled student matched or slightly beat using the LLM directly as a classifier, at a fraction of the serving cost — which is the entire reason the architecture is worth the trouble.

### The pipeline, stage by stage

**Stage 0 — Data hygiene.** Before anything expensive runs, deduplicate the full corpus: exact duplicates by normalized hashing, near-duplicates by MinHash/LSH or embedding similarity. This matters for two reasons — near-duplicates straddling a train/test boundary leak information and inflate measured accuracy, and duplicates otherwise waste LLM budget and distort cluster densities. Do this at the millions scale *before* drawing the LLM sample.

**Stage 1 — Representative sampling.** Draw the hundreds-of-thousands LLM-touchable subset deliberately, not randomly. Run a cheap embedding + clustering pass over the full corpus first, then sample with **stratification** (over source, language, length, coarse cluster) combined with **diversity / coreset selection** to cover the representation space, boosting rare clusters so the long tail is represented. Reserve part of the budget for later active-learning rounds. This is the stage that directly addresses the scale mismatch; get it wrong and everything downstream inherits a skewed view of the corpus.

**Stage 2 — Taxonomy induction.** Combine bottom-up discovery with top-down control. Use embeddings → dimensionality reduction (UMAP) → density clustering (HDBSCAN, with K-Means as a coverage-complete cross-check) to let the data reveal its structure cheaply. Then use a TnT-LLM-style iterative consolidation loop — process summaries in minibatches and have the LLM *update and merge* the taxonomy rather than append to it — to keep the label set from exploding. A complementary technique (from *Latent Topic Synthesis*, 2025) walks cluster by cluster and asks the LLM, with a forced yes/no, whether an existing label already fits before allowing a new one. Include an explicit "junk" branch and an "Other / Undefined" escape hatch.

**Stage 3 — Pseudo-labeling.** Run the LLM as an annotator over the sample using the final taxonomy, with constrained decoding to the label set plus "Other," and a confidence signal (e.g., from repeated sampling). Critically, the training signal is the **real, LLM-labeled documents** — not synthetic generated text (see §3).

**Stage 4 — Distillation.** Train a lightweight encoder on the pseudo-labeled sample. The default choice is `ModernBERT`; a cheap embedding + logistic-regression baseline is worth keeping as a sanity check, and `SetFit` is a good option for branches with few labels.

**Stage 5 — Production + junk filtering.** Serve the encoder over the full ~1M docs/day with no LLM calls. Read the "junk" branch off the multi-class head, or train a dedicated binary junk-vs-keep head from the same labels for tighter precision/recall control. Weak-supervision labeling functions (Snorkel-style spam/SEO heuristics) make a cheap, auditable backstop for the junk branch specifically.

**Stage 6 — Evaluation.** With no gold labels initially, evaluate taxonomy quality by **coverage** (the share of data falling into "Other" — lower is better) and human / LLM-as-judge spot checks for label accuracy and distinctness. Evaluate the downstream classifier on a small human-labeled gold set, tracking precision and recall separately, since distillation tends to trade recall for precision.

**Stage 7 — Human-in-the-loop retraining.** Route low-confidence and "Other" predictions to human review, fold the corrections back into the training set, and re-distill on a cadence. This closes the gap left by the LLM teacher over time and catches drift and newly emerging categories in an evolving corpus.

---

## 3. Key design decisions (what the research changed)

The research didn't just confirm the plan — it changed *which parts to trust*, and that is the real value.

**Use real pseudo-labeled documents, not synthetic data.** Popular AI-generated blueprints propose generating thousands of synthetic examples per category. The evidence is mixed-to-negative: an EMNLP 2023 study found that subjectivity in a task is associated with *worse* performance for models trained on synthetic data, and shortcut-learning research shows that encoders like BERT readily latch onto generation-template artifacts instead of real signal. Since we have a real corpus in the millions, we should pseudo-label real text and reserve synthetic generation only for targeted augmentation of rare branches.

**Embedding compressed "descriptors" is a tunable knob, not a law.** Another popular claim is that you must embed a short LLM-extracted "intent descriptor" rather than the raw document to avoid "vector drift." Peer-reviewed clustering work finds summarization does *not* consistently improve clustering. It helps mainly when the target labels are about *intent or function* (not surface topic) and when documents are long and noisy — and it costs an extra LLM pass over the whole sample. Treat it as an ablation to test on a validation slice, especially for the junk-vs-keep dimension.

**Prefer single-label / primary assignment over multi-label.** In the source research, LLM-vs-human agreement was strong for primary single-label assignment but collapsed for multi-label "all that apply" tasks, because the LLM over-applied labels. A junk filter is a single-label decision, which puts us squarely in the reliable regime.

**Encoder choice: ModernBERT by default.** `ModernBERT` leads on throughput and long-context handling; `DeBERTa-v3` tends to lead on raw accuracy and sample efficiency. For ~1M docs/day the throughput argument dominates, so ModernBERT is the default, with DeBERTa-v3 held in reserve if a validation gap demands it. At 1M/day the *average* sustained rate is only ~12 docs/sec, which is trivial for a batched GPU deployment; "sub-10ms single-doc on edge" is a separate, stricter target that needs quantization and a small model and should be benchmarked rather than assumed.

---

## 4. What is well-established vs still speculative

**Well-established — build on these.** The LLM-as-taxonomizer + LLM-as-labeler + lightweight-student architecture works and scales (TnT-LLM, KDD 2024), and the distilled student can match the LLM-as-classifier at far lower cost. Iterative "does an existing label fit?" consolidation prevents label explosion. UMAP-before-clustering improves embedding clustering. Deduplication is mandatory hygiene. LLM annotation rivals or beats crowd workers but produces *structured* (non-random) label noise that bounds the student.

**Speculative or contested — test, don't assume.** That embedding short descriptors "eliminates drift"; that synthetic-data-per-category should be the primary training signal; that "sub-10ms on edge" is achievable without caveats; and the reliability of multi-label distillation. Each of these should be treated as an experiment with a measured outcome, not a settled choice.

---

## 5. Primary risks

The two risks that will actually determine success are **not** the ones the popular blueprints fixate on (latency, architecture boxes). They are:

1. **The distillation ceiling.** The student's accuracy is bounded by the teacher's labeling quality, and the LLM's systematic biases get baked into the encoder as clean, confident-looking patterns. Mitigations: confidence-based QA on pseudo-labels, human spot-checks against a gold set, agreement analysis, and noise-aware training.

2. **Sampling representativeness.** If the LLM-touchable sample doesn't faithfully represent the millions — especially the rare tail — the taxonomy will have blind spots the encoder then propagates at scale. Mitigations: stratified + diversity sampling, coverage tracking, and the "Other" bucket as an early-warning signal for missing categories.

A third, softer risk: **"junk" is partly subjective.** Expect the junk branch to need the most human-in-the-loop refinement and the most weak-supervision backstopping — and recall that subjectivity is exactly what degrades synthetic-trained models, which is another reason to favor real, human-spot-checked junk labels.

---

## 6. Recommended next step

We are deliberately *not* jumping to a full build. The highest-value next move is a **cheap end-to-end trial on a small slice** — a few thousand documents run through every stage: dedup → cluster → LLM-induce a draft taxonomy → LLM-label → train a quick student → eyeball the result. One inexpensive full loop will teach us more than any additional reading, and it directly de-risks the two things that actually matter (the distillation ceiling and sampling), before we commit to the hundreds-of-thousands LLM spend.

The worked example we sketched — a single clickbait listicle traveling from raw text, through deduplication and sampling, into a "Junk / SEO" cluster, getting labeled, training the encoder, and then causing a *new* unseen junk page to be filtered in production — is essentially the shape of that trial run, just scaled down.

---

## References

- **TnT-LLM: Text Mining at Scale with Large Language Models** — Wan, Safavi, Jauhar et al. (Microsoft + University of Washington), KDD 2024. arXiv:2403.12173 · https://arxiv.org/abs/2403.12173
- **Latent Topic Synthesis: Leveraging LLMs for Electoral Ad Analysis** — Brady et al., 2025. arXiv:2510.15125 · https://arxiv.org/abs/2510.15125
- **From chaos to clarity: Building taxonomies from unstructured text using Large Language Models** — Vickie Liu, Data Science at Microsoft (Medium). https://medium.com/data-science-at-microsoft/from-chaos-to-clarity-building-taxonomies-from-unstructured-text-using-large-language-models-c1303db3adb1
- **ModernBERT: Smarter, Better, Faster, Longer** — Warner et al., 2024. arXiv:2412.13663 · https://arxiv.org/abs/2412.13663
- **ModernBERT or DeBERTaV3?** — Antoun, Sagot & Seddah, 2025. arXiv:2504.08716 · https://arxiv.org/abs/2504.08716
- **Synthetic Data Generation with Large Language Models for Text Classification: Potential and Limitations** — Li, Zhu, Lu & Yin, EMNLP 2023.
- **ChatGPT outperforms crowd workers for text-annotation tasks** — Gilardi, Alizadeh & Kubli, PNAS 2023.

*Note: arXiv identifiers are the reliable anchor for the academic sources; verify any specific quantitative figure against the primary paper before citing it externally.*
