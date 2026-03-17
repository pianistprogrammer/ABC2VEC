# ABC2Vec: Representation Learning for Symbolic Folk Music
## Research Brief & Publication Roadmap

---

## 1. Overview

This document covers the research landscape, novelty assessment, recommended approach, dataset options, and key papers for **ABC2Vec** — a project to build dedicated embedding models for ABC-notation folk tunes, analogous to word embeddings in NLP, trained purely on symbolic music.

Target applications:
- Tune similarity search
- Folk tune recommendation
- Clustering by style, mode, region
- Plagiarism / melodic borrowing detection
- Folk tune evolution analysis (computational musicology)

---

## 2. State of the Field (What Already Exists)

### 2.1 General Symbolic Music Embeddings

The space has matured significantly in 2023–2025. Key prior work:

**MidiBERT-Piano (2021)** — BERT-style pre-training on MIDI (OctupleMIDI tokens). Strong on piano tasks. Not folk-specific, not ABC-native.

**MusicBERT (2022)** — Large-scale MIDI pre-training from Microsoft. Bar-level masking. The go-to MIDI encoder for multimodal work (used in MIDI-LLaMA). Not text-based.

**CLaMP (ISMIR 2023, Best Student Paper)** — The most important prior work to understand. Contrastive Language-Music Pre-training over 1.4M music-text pairs. Uses ABC notation as the music representation, bar patching to compress sequences, and a Masked Music Model (M3) as the music encoder. It supports semantic search and zero-shot classification. ABC is used, but the goal is **cross-modal alignment with text**, not music-only embeddings. CLaMP 2 (2024) extended this to 101 languages and adds MIDI support alongside ABC.

**CLaMP 3 (2025)** — Adds audio as a third modality (ABC + MIDI + audio + text). State-of-the-art multilingual MIR.

**MelodyT5 (ISMIR 2024)** — Encoder-decoder Transformer trained on 261K ABC melodies (MelodyHub) across 7 tasks (harmonization, completion, transposition, etc.). Bar patching. Multi-task score-to-score learning.

**TunesFormer / IrishMAN** — GPT-2-style folk tune generation trained on 285K Irish/folk tunes from The Session + ABCnotation.com. Uses control codes (section count, edit-distance similarity) for structured generation.

**NotaGen (IJCAI 2025)** — Hierarchical GPT-2 pre-trained on 1.6M ABC sheets for sheet music generation with musicality optimization via CLaMP-DPO.

**PiRhDy (ACM MM 2020)** — Pitch, Rhythm, Dynamics-aware embeddings for symbolic music. An earlier but directly relevant reference for disentangled music embeddings.

### 2.2 Folk-Specific Prior Work

**Folk-RNN (Sturm et al., 2016)** — The original char-RNN trained on ~40K Irish tunes from The Session in ABC. Established the baseline for neural folk music generation. Not embedding-focused.

**GPT-2 Folk Music (Gwern/Presser, 2019–2020)** — Scaled folk-RNN to GPT-2 on 453K ABC pieces. Not embedding-focused.

**Tunepal** — Production tune identification app (22K+ Irish/Scots/Welsh tunes). Query-by-playing using DTW-based similarity. Traditional, not embedding-based.

### 2.3 What Does NOT Exist Yet

Despite the field's growth, there is a **clear gap**:

> No dedicated embedding model has been trained **purely on folk/traditional music in ABC notation** with the explicit goal of producing tune-level semantic embeddings for similarity, clustering, and evolution analysis.

CLaMP is the closest work, but:
- It is cross-modal (music + text), not music-only
- It is trained on general/classical music (WikiMT = 1010 lead sheets, mostly Western art music)
- It is not evaluated on folk-specific tasks (same-family tune retrieval, variant detection, regional clustering)
- Folk tunes have unique structures (AABB form, modes, ornaments, regional variation) that a general model may not encode well

This gap is your opening.

---

## 3. Novelty Assessment

### The ABC2Vec Claim

**ABC2Vec** is publishable if it makes at least one of the following novel contributions:

| Contribution | Novelty Level | Notes |
|---|---|---|
| First music-only (no text) encoder trained on folk ABC at scale | Medium-High | CLaMP uses text; this is self-supervised folk-only |
| Folk-specific pre-training objectives (e.g., section-pair, variant-aware masking) | High | No prior work uses folk structure as a training signal |
| Quantitative eval on folk similarity tasks (same-tune family retrieval) | High | No standard benchmark exists; you'd create it |
| Tune evolution / genealogy analysis via embedding geometry | High | Novel application, novel methodology |
| Cross-tradition transfer (Irish → Scottish → Appalachian) | Medium-High | Untested in the embedding literature |

The strongest paper angles are:
1. **Music-only self-supervised folk embedding** + new evaluation benchmark
2. **Folk-aware pre-training objectives** exploiting the structure of ABC (modes, parts, variants)
3. **Computational musicology application** — using the embedding space to study tune lineage/evolution

### Publication Venue Assessment

| Venue | Fit | Notes |
|---|---|---|
| **ISMIR** (International Society for Music Information Retrieval) | ★★★★★ | The top venue for this work. CLaMP won Best Student Paper here. Directly relevant. |
| **ACM Multimedia (ACM MM)** | ★★★★ | Strong for multimodal extensions |
| **ICASSP** | ★★★ | Good for signal-processing angle |
| **ACL / EMNLP (Findings)** | ★★★ | If you lean into the NLP analogy (music as language) |
| **Transactions of ISMIR** | ★★★★ | Journal version, less competitive |
| **Journal of New Music Research** | ★★★ | Good for musicology angle |

**Recommended primary target: ISMIR.** Submission typically opens in early spring, conference in autumn. ISMIR 2026 will likely be announced mid-2025.

---

## 4. Recommended Methodology

### 4.1 Pre-Training Architecture

Base the architecture on the M3 encoder from CLaMP, but adapted for folk:

```
ABC Notation → Bar Patchifier → Transformer Encoder → Tune Embedding
```

**Bar Patching:** Group characters within a single bar into a patch (following CLaMP/MelodyT5). Each bar ≈ one token. Reduces sequence length by ~10x. Critical for efficiency.

**Encoder:** 6–12 layer Transformer. Hidden size 256–512. Start modest — folk tunes are short (8–32 bars typically).

### 4.2 Pre-Training Objectives

This is where you can differentiate from CLaMP:

**Objective 1 — Masked Music Modeling (MMM)**
Standard BERT-style: randomly mask bar patches, predict them. Used in CLaMP. Include as baseline.

**Objective 2 — Section Contrastive Loss (SCL)** *(Novel)*
Folk tunes have an AABB structure. A-sections and B-sections are musically related but distinct. Train the model to produce embeddings where A-section and B-section of the *same tune* are closer than sections from *different tunes*. This bakes structural folk awareness in.

**Objective 3 — Transposition Invariance** *(Semi-novel)*
The same reel in D major and G major is the same tune. Apply contrastive loss between transposed versions of a tune as positive pairs. Teaches pitch-invariant representation.

**Objective 4 — Variant-Aware Contrastive** *(Novel, harder)*
The Session tags tunes with "settings" (variants). Use same-tune-different-setting as hard positives. Use different-tune as negatives. This directly trains similarity for variant detection.

### 4.3 Tokenization

Two options:

**Option A — Character-level with bar patching** (CLaMP/MelodyT5 approach)
- Treat each bar's characters as a single "patch"
- Linear projection to embedding dimension
- Simpler, proven to work

**Option B — Music-specific tokens**
- Tokenize pitch, duration, octave separately (OctupleMIDI style but for ABC)
- Richer structured input
- Higher complexity

Start with Option A. Move to Option B in an ablation study.

### 4.4 Training Data Strategy

```
Phase 1: Pre-train on large general folk corpus (IrishMAN / abc_cc, ~285K tunes)
Phase 2: Fine-tune on evaluation-specific subsets (with variant labels from The Session)
```

### 4.5 Evaluation Protocol

Since no standard folk similarity benchmark exists, you'll need to create one — which becomes a contribution:

**Benchmark 1 — Same-Tune-Family Retrieval**
The Session organizes tunes by "settings" (same tune, different notation). Use these as ground-truth pairs. Given tune A, retrieve setting B from a corpus. Metric: MRR, Recall@1, Recall@5, MAP.

**Benchmark 2 — Mode/Style Clustering**
Tunes are labeled (jig, reel, hornpipe, polka, slip jig, slide). Evaluate whether embeddings cluster by type. Metric: Silhouette score, NMI, clustering accuracy.

**Benchmark 3 — Cross-Tradition Transfer**
Train on Irish tunes. Evaluate retrieval on Scottish / Appalachian tunes (BFDB). Tests generalization.

**Benchmark 4 — Plagiarism / Borrowing Detection** (optional extension)
Use known melodically related tune pairs from musicological literature. Evaluate ranking.

---

## 5. Datasets

| Dataset | Size | Description | Access |
|---|---|---|---|
| **IrishMAN** | 216,284 tunes | Irish ABC from The Session + ABCnotation.com, public domain, normalized | HuggingFace: `sander-wood/irishman` |
| **abc_cc (TunesFormer)** | 285,449 tunes | Irish + general folk ABC, with control code annotations | HuggingFace: `sander-wood/abc_cc` |
| **The Session** | ~40,000+ | Irish trad, community-annotated, variant "settings" per tune — **critical for similarity labels** | thesession.org (scrapeable) |
| **ABCnotation.com index** | ~800,000 tunes | Broad index across world folk traditions | abcnotation.com/search |
| **BFDB (British Folk DB)** | 13,835 tunes | Dated British folk tunes with year metadata — ideal for **evolution analysis** | Zenodo |
| **Nottingham Music Database** | ~1,000 tunes | Classic benchmark, British folk | abc.sourceforge.net |
| **WikiMT (CLaMP)** | 1,010 lead sheets | Wikifonia-sourced, genre-labeled, ABC notation, text descriptions | CLaMP GitHub |

**Recommended corpus for pre-training:** IrishMAN (216K) or abc_cc (285K). Both are normalized, deduplicated, public domain.

**Recommended corpus for evaluation / fine-tuning:** The Session with scraped "settings" metadata. This is the only source with ground-truth tune variant labels at scale.

---

## 6. Papers to Read (Prioritized)

### Tier 1 — Must Read

1. **CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval**
   Wu et al., ISMIR 2023 (Best Student Paper)
   arXiv: 2304.11029
   *The most directly relevant prior work. Understand every design decision here.*

2. **MelodyT5: A Unified Score-to-Score Transformer for Symbolic Music Processing**
   ISMIR 2024
   arXiv: 2407.02277
   *Best example of ABC-native Transformer with bar patching for melody tasks.*

3. **TunesFormer / IrishMAN**
   Sander Wood et al.
   HuggingFace: `sander-wood/irishman`
   *The largest folk-specific ABC dataset + generation model. Your pre-training corpus comes from here.*

4. **PiRhDy: Learning Pitch-, Rhythm-, and Dynamics-Aware Embeddings for Symbolic Music**
   Liang et al., ACM MM 2020
   *Direct precedent for disentangled symbolic music embeddings. Your embedding design can build on this.*

5. **CLaMP 2 / CLaMP 3**
   Wu et al., 2024/2025
   arXiv (search "CLaMP 2 multilingual")
   *Know the current state of the art you're differentiating from.*

### Tier 2 — Important Context

6. **MidiBERT-Piano: Large-Scale Pre-Training for Symbolic Music Understanding**
   Chou et al., 2021
   *The BERT-for-MIDI baseline that kicked off this genre of work.*

7. **Folk-RNN: Music Transcription Modelling and Composition Using Deep Learning**
   Sturm et al., 2016 (arXiv)
   *The foundational folk-ABC deep learning paper. Establishes the tradition you're building on.*

8. **NotaGen: Advancing Musicality in Symbolic Music Generation**
   IJCAI 2025
   arXiv: 2502.18008
   *Shows CLaMP-DPO for musicality; relevant for understanding the M3 encoder ecosystem.*

9. **Symbolic Melodic Similarity: State of the Art and Future Challenges**
   Velardo, Vallati, Jan. Computer Music Journal, 2016
   *Survey of all pre-deep-learning melody similarity methods. Important for framing your baseline comparisons.*

10. **MelodySim: Measuring Melody-aware Music Similarity for Plagiarism Detection**
    arXiv: 2505.20979
    *Recent audio-side plagiarism detection using triplet networks. Compare your symbolic approach against this.*

### Tier 3 — Useful Background

11. **A Survey on Deep Learning for Symbolic Music Generation: Representations, Algorithms, Evaluations, and Challenges**
    ACM Computing Surveys (2023)
    *Comprehensive survey. Read the Representation and Evaluation sections.*

12. **ABC-Eval: Benchmarking Large Language Models on ABC Notation**
    arXiv: 2509.23350
    *Recent benchmark for LLM understanding of ABC. Useful for understanding what ABC encodes and where models fail.*

13. **Classification of Origin with Feature Selection and Network Construction for Folk Tunes**
    Metzig et al., Pattern Recognition Letters, 2020
    *Uses BFDB; shows folk tunes cluster by geographic origin. Your embeddings should do this better.*

14. **Fine-Grained Music Plagiarism Detection (BMM-Det)**
    arXiv: 2107.09889
    *Bipartite graph matching for symbolic plagiarism. Your embedding approach should outperform this.*

15. **ChatMusician: Understanding and Generating Music Intrinsically with LLMs**
    ACL Findings 2024
    *Shows ABC as a first-class LLM format. Context for why ABC is the right choice.*

---

## 7. Differentiation Strategy vs. CLaMP

The reviewers' first question will be: *"CLaMP already does symbolic music retrieval in ABC. What's new?"*

Your answers:

**1. Music-only, no text dependency**
CLaMP requires text descriptions at inference for semantic search. ABC2Vec produces tune-level embeddings purely from notation. No text metadata needed. Useful for the vast majority of folk tunes that have no Wikipedia article.

**2. Folk-specific pre-training objectives**
CLaMP uses generic masked modeling. ABC2Vec uses folk-aware objectives: transposition invariance, section contrastive learning, variant-pair supervision. These are structurally motivated by folk music practice.

**3. Folk-specific evaluation benchmark**
CLaMP evaluates on WikiMT (classical/jazz/pop lead sheets). No prior work evaluates on same-tune-family retrieval at scale. You create this benchmark as a public release.

**4. Computational musicology application**
Using the trained embeddings to map folk tune evolution, detect melodic lineage, and visualize cross-tradition borrowing is a novel application not demonstrated in any prior work.

**5. Interpretable embedding dimensions**
Probe the embedding space: do axes correspond to mode (Dorian vs. Mixolydian), rhythm type (jig vs. reel), or regional origin? Probing analysis is a differentiating section.

---

## 8. Execution Roadmap

### Phase 1 — Foundation (Weeks 1–4)
- [ ] Set up data pipeline: download IrishMAN / abc_cc, normalize, deduplicate
- [ ] Implement ABC tokenizer + bar patchifier (reference TunesFormer/CLaMP code)
- [ ] Reproduce a miniature CLaMP encoder as your baseline
- [ ] Establish retrieval evaluation harness using The Session variant pairs

### Phase 2 — Core Model (Weeks 5–10)
- [ ] Implement ABC2Vec encoder with MMM pre-training
- [ ] Add transposition invariance contrastive objective
- [ ] Add section contrastive learning (A/B part pairs)
- [ ] Pre-train on IrishMAN (~216K tunes), evaluate on variant retrieval

### Phase 3 — Ablations + Applications (Weeks 11–16)
- [ ] Ablation study: each pre-training objective in isolation
- [ ] Clustering analysis: reel/jig/hornpipe separation
- [ ] Cross-tradition retrieval: Irish → Scottish → Appalachian
- [ ] Folk evolution visualization: BFDB dated tunes projected in 2D
- [ ] Probing experiments: what do embedding dimensions encode?

### Phase 4 — Write-up (Weeks 17–20)
- [ ] ISMIR paper: 6 pages + references
- [ ] Release model weights + evaluation benchmark on HuggingFace
- [ ] Release processed evaluation dataset (same-tune pairs from The Session)

---

## 9. Technical Stack

```
Language:         Python 3.10+
Framework:        PyTorch + HuggingFace Transformers
ABC Processing:   music21, abcjs (for validation), xml2abc
Data:             IrishMAN (HuggingFace datasets)
Tokenization:     Custom ABC bar patchifier (reference CLaMP's patchilizer)
Training:         Single A100 or 2x 3090 — folk tunes are short, feasible without large cluster
Evaluation:       FAISS for approximate nearest-neighbor retrieval
Visualization:    UMAP / t-SNE for embedding space analysis
Baseline:         CLaMP-S/512 (pretrained weights available on GitHub)
```

---

## 10. Key Risk Factors

| Risk | Mitigation |
|---|---|
| CLaMP already solves retrieval well enough | Focus on the no-text, folk-specific claim; create a benchmark where CLaMP fails |
| The Session scraping becomes unavailable | Use IrishMAN (already includes The Session data, pre-processed) |
| Variant labels are noisy | Use conservative ground truth (only tunes with 3+ settings on The Session) |
| Pre-training doesn't improve over CLaMP | Ablate aggressively; publish the benchmark even if model results are mixed |
| ISMIR scope too narrow for evolution analysis | Frame musicology angle as a "use case" section, not the main contribution |

---

## 11. Summary Positioning

> **ABC2Vec** is the first music-only self-supervised embedding model specifically designed for folk music in ABC notation. Trained on 200K+ public-domain folk tunes using folk-aware objectives (transposition invariance, section contrastive learning, variant supervision), it produces tune-level embeddings that capture melodic similarity, regional style, and structural variation without requiring text metadata. We introduce the first public benchmark for folk tune family retrieval and demonstrate applications in similarity search, clustering, plagiarism detection, and tune evolution analysis.

This is a 6-page ISMIR paper with a dataset contribution (benchmark release) that makes it robustly publishable even if the model doesn't obliterate CLaMP on every metric — because the benchmark itself is a novel artifact.

---

*Document prepared March 2026. Field is moving fast — check arXiv cs.SD and ISMIR 2025 proceedings for new preprints before submission.*
