# ABC2Vec Paper - Compilation Instructions

## Files Created

1. **abc2vec_paper.tex** - Main paper LaTeX source
2. **abc2vec.bib** - Bibliography file with all references
3. **Images copied from results:**
   - umap_tune_type.png
   - umap_mode.png
   - tune_type_confusion.png
   - dimension_importance.png
   - probing_summary.png

## To Compile

### Option 1: Using pdflatex + bibtex

```bash
cd Taylor___Francis_LaTeX_template_p_reference_style_/
pdflatex abc2vec_paper.tex
bibtex abc2vec_paper
pdflatex abc2vec_paper.tex
pdflatex abc2vec_paper.tex
```

### Option 2: Using latexmk (recommended)

```bash
cd Taylor___Francis_LaTeX_template_p_reference_style_/
latexmk -pdf abc2vec_paper.tex
```

### Option 3: Using Overleaf

1. Create new project in Overleaf
2. Upload abc2vec_paper.tex
3. Upload abc2vec.bib
4. Upload interact.cls and tfp.bst
5. Upload all .png images
6. Compile

## Missing Figure

**training_loss_placeholder.pdf** - You need to generate this manually or use the provided graph1.eps/graph2.eps as placeholders.

To generate the training loss plot, run:

```python
import json
import matplotlib.pyplot as plt

with open('../checkpoints/training_history.json') as f:
    history = json.load(f)

steps = [item['step'] for i, item in enumerate(history) if i % 100 == 0]
mmm = [item['mmm'] for i, item in enumerate(history) if i % 100 == 0]
ti = [item['ti'] for i, item in enumerate(history) if i % 100 == 0]
total = [item['total'] for i, item in enumerate(history) if i % 100 == 0]

plt.figure(figsize=(10, 6))
plt.plot(steps, mmm, label='MMM Loss', linewidth=2.5)
plt.plot(steps, [t*20 for t in ti], label='TI Loss (×20)', linewidth=2.5)
plt.plot(steps, total, label='Total Loss', linewidth=2.5, linestyle='--')
plt.axvline(x=15000, color='green', linestyle=':', label='Best Model')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('ABC2Vec Training Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss_placeholder.pdf', dpi=300, bbox_inches='tight')
```

## Paper Structure

The paper includes:

✅ Abstract (200 words)
✅ Introduction with 4 research questions
✅ Related Work (CLaMP, MusicBERT, MelodyT5, etc.)
✅ Methodology with detailed formulas:
   - Bar patchification equations
   - Transformer encoder architecture  
   - Masked Music Modeling (MMM) loss
   - Transposition Invariance (TI) contrastive loss
   - Combined training objective
✅ Results with 6 tables and 5+ figures
✅ Discussion answering all 4 research questions
✅ Limitations section
✅ Conclusion and Future Work
✅ 30+ properly formatted references

## Key Results Summary

- **Tune Type Classification:** 78.3% ± 0.8%
- **Mode Classification:** 80.5% ± 0.4%  
- **Variant Detection:** 0.676 similarity (vs 0.253 for unrelated)
- **Linear Probing:** 77.4% tune type, 73.6% tune length, 56.4% key root
- **Clustering:** Poor (negative silhouette), indicating distributed representations

## Citation Style

Uses Taylor & Francis reference style (numbered) with natbib.

