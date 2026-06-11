# Rongorongo

Casual experiments in using NLP to see whether modern embedding methods can say anything useful about **Rongorongo**, the undeciphered script of Easter Island (Rapa Nui).

This is not a decipherment claim, a published method, or a finished pipeline. It is exploratory work: tokenize transliterations, learn projections into English transformer space, nudge a handful of known glosses as anchors, and inspect neighbors, clusters, and tuning runs. Most results so far are collapse artifacts, anchor overfitting, or generic English words—not recovered meaning.

---

## What is Rongorongo?

**Rongorongo** is a writing system used on Rapa Nui (Easter Island), a remote Polynesian island in the southeastern Pacific. It is one of the few independent script traditions in Oceania and remains **largely undeciphered**.

### The artifacts

Surviving texts are inscribed on wooden tablets and related objects, mostly in the 19th century after European contact. The corpus is small: on the order of **two dozen authentic tablets** and roughly **15,000–20,000 glyphs** in total. Lines are read in **reverse boustrophedon** (alternating rows, one direction reversed). Many glyphs are pictographic or otherwise visually distinct; the script appears to mix logographic and possibly syllabic or rebus-like elements, but that is still debated.

### History and context

Oral tradition on the island associated rongorongo with elites, genealogy, and ritual knowledge. Production may have declined before or around the 1860s—periods of population collapse, labor raids, and cultural disruption. Collectors and missionaries removed most surviving pieces to museums in Europe and the Americas. No fluent reader is known to have been recorded in enough detail to unlock the system.

### Transliteration

Because the script is unread, modern study relies on **transliterations**: glyph-by-glyph notation of what appears on each line. Common schemes (e.g. Barthel’s catalog) assign numeric codes to glyph shapes and use letters for variant forms (size, orientation, or sub-glyphs). Those strings are the input data here—not translations.

### Why decipherment is hard

- Very **small corpus**; statistical methods have little to work with.
- **Unknown language** (likely related to Rapa Nui language, but not proven glyph-by-glyph).
- **Unclear structure**: prose, chant, genealogy, calendar, or mixed genres.
- Only a **few contested partial readings** (e.g. lunar or genealogical interpretations); nothing like a Rosetta parallel text.

---

## What this repo does (briefly)

| Area | Role |
|------|------|
| `rr_tablets/` | Transliterated tablet texts and tokenized variants |
| `utils/` | Tokenization strategies (simple, Barthel, suffix, glyph+variants, etc.) |
| `embed/` | Project Rongorongo tokens into frozen English embedding space (e5-small, etc.), gloss anchors, hyperparameter tuning, neighbor JSONs and plots |
| `glyph_variants` | Structured tokenizer (`{glyph, variant, modifiers}`) with dedicated projection learner (`learn_glyph_variant_projection.py`) |

Generated artifacts (`outputs/`, `embed/tuning/trial_outputs/`, etc.) are local experiment output and are gitignored.

### Glyph variants pipeline

```bash
python utils/tokenize_transliterated.py --strategy glyph_variants
python embed/run_embedding_pipeline.py \
  --tokenized-dir rr_tablets/transliterated/complete/tokenized/glyph_variants \
  --transformer-preset e5-small \
  --epochs 12 --anchor-weight 8 --projected-cooccurrence-weight 0.5 \
  --collapse-weight 0.15 --anchor-passes 12
```

Vocab keys flatten structured tokens (e.g. `003`, `003:a`, `003:a+x`) so `003:a` and `003:b` share glyph `003` but train as distinct rows.

```bash
python embed/tune_projection_hyperparams.py --grid glyph_variants_focused --resume
```

---

## Expectations

Treat any English “neighbor” of a glyph as a **hypothesis to debug**, not a reading. Useful signals so far are mostly: whether supervised glosses stick, whether co-occurrence clusters glyphs together in embedding space, and which tokenizations/hyperparameters avoid collapsing every glyph onto one English word.

If you are looking for a serious decipherment effort, start with standard references on Rongorongo epigraphy and Rapa Nui history—not this repository.
