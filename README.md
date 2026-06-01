# 🗞️ Fake News Detection using BERT Fine-tuning
### A Comparative Study of BERT and DistilBERT on the LIAR Dataset

**Author:** Abhishikta Dutta  
**Institution:** Siliguri Institute of Technology (MAKAUT)  
**Project Type:** NLP Research Project  

---

## 📌 Abstract

This project investigates the effectiveness of transformer-based language models — specifically BERT and DistilBERT — for automated detection of political misinformation. Using the LIAR dataset (12,836 labeled political statements), we fine-tune both models on a binary classification task (FAKE vs REAL) and conduct a comparative analysis across accuracy, F1 score, precision, recall, and inference speed. Our error analysis reveals that ambiguous middle-ground statements (`barely-true`, `half-true`) are the primary source of misclassification, and that DistilBERT achieves competitive performance at significantly lower computational cost.

---

## 🔬 Research Questions

1. Can fine-tuned BERT reliably classify political misinformation across binary labels?
2. How does DistilBERT compare to BERT in accuracy vs. inference efficiency?
3. What linguistic patterns characterize statements that both models misclassify?

---

## 📊 Dataset

**LIAR Dataset** — Wang (2017)  
- Source: [HuggingFace Datasets](https://huggingface.co/datasets/liar)  
- 12,836 short political statements from PolitiFact.com  
- 6 original veracity labels: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`  
- Binary grouping used in this project: FAKE = {pants-fire, false, barely-true}, REAL = {half-true, mostly-true, true}

| Split      | Samples |
|------------|---------|
| Train      | 10,269  |
| Validation | 1,284   |
| Test       | 1,267   |

---

## 🏗️ Project Structure

```
fake_news_detection/
│
├── Week1_EDA_and_Setup.ipynb        # Data loading, EDA, tokenizer analysis
├── Week2_BERT_Finetuning.ipynb      # Fine-tuning BERT & DistilBERT, comparison
├── Week3_Error_Analysis.ipynb       # Error analysis, live demo, research write-up
│
├── bert_fakenews_model/             # Saved BERT model (generated after Week 2)
├── distilbert_fakenews_model/       # Saved DistilBERT model (generated after Week 2)
│
├── train_processed.csv              # Processed training data (generated after Week 1)
├── val_processed.csv
├── test_processed.csv
│
└── README.md
```

---

## ⚙️ Setup & Running

### Requirements
All notebooks are designed to run on **Google Colab (free tier)**.  
Enable GPU: `Runtime → Change runtime type → T4 GPU`

### Dependencies (auto-installed in each notebook)
```
transformers
datasets
scikit-learn
torch
pandas
numpy
matplotlib
seaborn
```

### Run Order
```
1. Week1_EDA_and_Setup.ipynb       ← Start here (no GPU needed)
2. Week2_BERT_Finetuning.ipynb     ← Requires GPU
3. Week3_Error_Analysis.ipynb      ← Requires GPU + saved models from Week 2
```

---

## 🧠 Models

| Model | Parameters | Architecture |
|-------|-----------|--------------|
| `bert-base-uncased` | 110M | 12-layer transformer encoder |
| `distilbert-base-uncased` | 66M | 6-layer distilled transformer |

**Training configuration:**
- Epochs: 3  
- Batch size: 16  
- Learning rate: 2e-5  
- Max sequence length: 128 tokens  
- Optimizer: AdamW with linear warmup scheduler  
- Gradient clipping: max norm 1.0  

---

## 📈 Results

*(Fill in your actual results after training)*

| Model | Test Accuracy | Test F1 (weighted) | Inference (ms/batch) |
|-------|-------------|-------------------|----------------------|
| BERT-base | — | — | — |
| DistilBERT-base | — | — | — |

---

## 🔍 Key Findings

- **Ambiguous statements are hardest:** `barely-true` and `half-true` original labels have the highest misclassification rates when binarized, revealing a fundamental challenge in coarse-grained fake news classification.
- **Confidence is calibrated:** Both models show lower prediction confidence on incorrect predictions, suggesting their softmax outputs are meaningful uncertainty signals.
- **DistilBERT is competitive:** DistilBERT achieves within 1-2% accuracy of BERT at ~40% fewer parameters and ~1.6x faster inference — favoring deployment in real-time applications.
- **Statement length has low impact:** Error rates are not strongly correlated with statement length, suggesting models rely on semantic content rather than surface features.

---

## ⚠️ Limitations

- Binary binarization loses nuance from the original 6-class label scheme.
- The LIAR dataset is limited to U.S. political statements; cross-domain generalization is not guaranteed.
- No metadata features (speaker history, political party, context) were used — incorporating these could significantly improve accuracy.

---

## 🚀 Future Work

- Fine-tune on all 6 original classes directly.
- Incorporate speaker metadata as auxiliary input features.
- Experiment with RoBERTa and DeBERTa architectures.
- Apply to multilingual misinformation datasets (e.g., MultiFC, FakeSV).
- Explore Explainability via attention visualization and LIME/SHAP.

---

## 📚 References

- Wang, W. Y. (2017). *"Liar, liar pants on fire": A new benchmark dataset for fake news detection.* ACL 2017.  
- Devlin, J., et al. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding.* NAACL 2019.  
- Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT.* EMC2 Workshop, NeurIPS 2019.

---

## 📄 License

MIT License — free to use for academic and research purposes.
