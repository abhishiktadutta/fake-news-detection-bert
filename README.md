# 🗞️ Fake News Detection using BERT Fine-tuning
### A Comparative Study of BERT and DistilBERT on the LIAR2 Dataset

**Author:** Abhishikta Dutta  
**Institution:** Siliguri Institute of Technology (MAKAUT)  
**Project Type:** NLP Research Project  

---

## 📌 Abstract

This project investigates the effectiveness of transformer-based language models — BERT and DistilBERT — for automated detection of political misinformation. Using the LIAR2 dataset (~23,000 fact-checked political statements), both models were fine-tuned on a binary classification task (FAKE vs REAL) and compared across accuracy, F1 score, and inference efficiency. A per-class and confidence-based error analysis reveals two key insights: model performance degrades sharply on ambiguous "barely-true" statements that sit at the FAKE/REAL boundary, and both models are well-calibrated, expressing lower confidence on incorrect predictions. DistilBERT was found to match BERT's accuracy at roughly 40% fewer parameters, suggesting it is the more practical choice for deployment.

---

## 🔬 Research Questions

1. Can fine-tuned BERT reliably classify political misinformation under a binary scheme?
2. How does DistilBERT compare to BERT in accuracy versus computational efficiency?
3. Which types of statements do the models misclassify, and what does this reveal about the task?

---

## 📊 Dataset

**LIAR2 Dataset** — an enhanced version of the original LIAR benchmark (Wang, 2017), re-labeled by professional fact-checkers.

- Source: [chengxuphd/liar2 on HuggingFace](https://huggingface.co/datasets/chengxuphd/liar2)
- ~23,000 short political statements from PolitiFact
- 6 original veracity labels: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`
- Binary grouping used here: FAKE = {pants-fire, false, barely-true}, REAL = {half-true, mostly-true, true}

| Split      | Samples |
|------------|---------|
| Train      | 18,369  |
| Validation | 2,297   |
| Test       | 2,296   |

Training class balance: FAKE 10,591 / REAL 7,778 (~58/42).

---

## 🏗️ Project Structure

```
fake_news_detection/
│
├── Week1_EDA_and_Setup.ipynb        # Data loading, EDA, tokenizer analysis
├── Week2_BERT_Finetuning.ipynb      # Fine-tuning BERT & DistilBERT + comparison
├── Week3_Error_Analysis.ipynb       # Error analysis, calibration, live demo
│
├── charts/                          # Saved visualizations (.png)
└── README.md
```

> Note: This project was developed on Google Colab using the free T4 GPU. The notebooks load LIAR2 directly from HuggingFace at runtime — no manual dataset download is required.

---

## ⚙️ Setup & Running

Designed to run on **Google Colab (free tier)**. Enable GPU via `Runtime → Change runtime type → T4 GPU`.

Dependencies (auto-installed in the notebooks): `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.

Run order: Week 1 → Week 2 (GPU on) → Week 3. In practice, Weeks 2 and 3 can be run in a single session so the trained models stay in memory.

---

## 🧠 Models

| Model | Parameters | Architecture |
|-------|-----------|--------------|
| `bert-base-uncased` | 110M | 12-layer transformer encoder |
| `distilbert-base-uncased` | 66M | 6-layer distilled transformer |

**Training configuration:** 3 epochs · batch size 16 · learning rate 2e-5 · max sequence length 128 · AdamW optimizer with linear warmup · gradient clipping (max norm 1.0).

---

## 📈 Results

| Model | Test Accuracy | Test F1 (weighted) |
|-------|--------------|-------------------|
| BERT-base | 0.7008 | 0.7025 |
| DistilBERT-base | 0.6990 | 0.7007 |

Both models reach ~70% accuracy on LIAR2 binary classification. Notably, DistilBERT performs within 0.2% of BERT despite being ~40% smaller, making it the more efficient option for real-world use.

---

## 🔍 Key Findings

**1. Performance collapses at the FAKE/REAL boundary.**  
Per-class analysis shows both models perform well on extreme veracity labels (BERT: `pants-fire` 86%, `true` 78%) but degrade sharply on `barely-true` statements (44%) — below random chance. These "barely-true" statements were binned into FAKE but are the least fake of that category, containing partial truths that linguistically resemble accurate statements. This indicates the principal challenge in automated misinformation detection lies not in flagrant falsehoods but in partially-true claims near the decision boundary.

**2. Both models are well-calibrated.**  
Mean prediction confidence is higher on correct predictions (BERT 0.87, DistilBERT 0.85) than on errors (0.79, 0.76). The models express appropriate uncertainty when wrong, suggesting a confidence threshold could be used in a deployed system to flag uncertain predictions for human review.

**3. DistilBERT is competitive at lower cost.**  
DistilBERT matches BERT's accuracy and exhibits the same per-class error pattern while requiring substantially fewer parameters, favoring it for latency- or resource-constrained deployment.

---

## ⚠️ Limitations

- Binary grouping discards nuance from the original 6-class labels; a 6-class model may be more informative.
- LIAR2 covers U.S. political statements; generalization to other domains (health, science) is untested.
- No metadata (speaker history, party, context) was used; incorporating it could improve accuracy, as the original LIAR paper showed.

---

## 🚀 Future Work

- Train directly on the 6-class labels and compare against the binary setup.
- Incorporate speaker and context metadata as auxiliary features.
- Evaluate RoBERTa or DeBERTa for potential accuracy gains.
- Add attention-based explainability (which tokens drive a FAKE prediction).

---

## 📚 References

- Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL 2017.
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT.* NeurIPS 2019 EMC2 Workshop.
- LIAR2 dataset: *An Enhanced Fake News Detection System With Fuzzy Deep Learning.*

---

## 📄 License

MIT License — free to use for academic and research purposes.
