# Bank Marketing: Data Science Blog Post

## Motivation
Answer three marketing questions with the CRISP-DM process: prior campaign success, repeated contact, and pre-call prospect ranking. The article presents findings for a nontechnical reader and separates associations from causal claims.

## Files
- `Bank_Marketing_CRISP_DM.ipynb`: executed notebook covering business understanding, data understanding, preparation, modeling, evaluation, and deployment considerations.
- `analysis.py`: documented reusable analysis functions.
- `data/bank-full.csv`: unchanged 45,211-row UCI source data.
- `data/bank-names.txt`: original data documentation.
- `results/`: calculated rates, model metrics, and two charts.
- `ARTICLE.md` and `article.html`: reader-friendly article, ready for public publication.
- `test_analysis.py`: meaningful data-integrity and leakage/holdout tests.

## Install and run
Python 3.12 is recommended. Install with `pip install -r requirements.txt`.
Run `python analysis.py` from this folder, then `python -m unittest -v test_analysis`.
Open the notebook using Jupyter; restart and run all cells to reproduce the analysis.

## Results
Overall subscription rate is 11.70%. A successful previous campaign is associated with 64.73% subscription versus 9.16% for an unknown previous outcome. Contact-count groups show declining observed rates, but reverse causation is possible. The chronological holdout model has ROC AUC 0.716 and average precision 0.547; its top-ranked 20% has 1.93× observed lift. The holdout positive rate is 31.58%, illustrating a significant period shift.

## Limits
Results are historical and observational. Contact channel must be known before scoring. No profitability, causal uplift, or modern deployment suitability is asserted. Call duration and same-campaign count are excluded from predictive features. Unknown categorical values are treated as missing for modeling; imputation is fitted within the training pipeline. The pdays sentinel -1 is separated into a contact-history flag and a missing elapsed-day value.

## Acknowledgments and license
Data: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306. CC BY 4.0; data redistributed unchanged with attribution. Code and article authored with AI assistance, explicitly disclosed in the article. Analysis outputs were executed and verified.

## Publication
Public article: [https://bank-marketing-project-analysis.dmexpert015.chatgpt.site](https://bank-marketing-project-analysis.dmexpert015.chatgpt.site). The published page presents the three business questions, calculated findings, visualizations, caveats, data source and AI assistance disclosure. Include this public URL in the submission reviewer notes.
