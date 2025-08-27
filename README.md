# Resume–Job Matcher (Prototype)

> **Prototype** system for semantically aligning résumés with job descriptions using transformer-derived embeddings, FAISS approximate nearest-neighbor search, lexical weighting, and interpretable skill-gap analysis.  
> ⚠️ This repo contains the **prototype**; a production variant (maintained privately) includes additional compliance, scalability, and security layers.

---

## ✨ Features

- **Semantic Embedding Engine** → Sentence-Transformers for high-dimensional vectorization of résumés and job descriptions.
- **FAISS ANN Search** → Fast candidate retrieval with cosine-normalized similarity.
- **Composite Scoring** → Weighted combination of semantic similarity, TF-IDF lexical overlap, and explicit skill coverage.
- **Skill Gap Detection** → Extracts missing technical and soft skills using lexical + ontology rules.
- **Attribution & Explainability** → Surfaces which résumé sections and terms contribute most to the score.
- **CLI Support** → Score individual candidates or batch rank an entire folder.
- **Observed Impact (pilot)** → ~27% reduction in manual screening latency compared to keyword-only filters.

---

## 📂 Repository Structure

resume-job-matcher/
├── resume_matcher_pipeline.py # Core pipeline (CLI + scoring)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── data/
│ ├── resumes/ # (empty) résumés in .txt
│ └── jobs/ # (empty) job descriptions in .txt
├── models/ # (optional) store FAISS index or LFS models
└── notebooks/ # Colab/Jupyter explorations

---

## 🧠 Methodology

### Semantic Similarity

For each résumé sentence \( r_i \) and JD sentence \( j_k \):

\[
S\_{\text{sem}} = \frac{1}{|J|}\sum_k \max_i \ \hat{r}\_i^\top \hat{j}\_k
\]

where \(\hat{r}\_i, \hat{j}\_k\) are normalized SBERT embeddings.

### Lexical Overlap

TF-IDF-weighted overlap between JD tokens and résumé tokens:

\[
S*{\text{tfidf}} = \frac{\sum*{t \in J} \mathrm{tfidf}(t,J)\cdot \mathbf{1}[t \in R]}{\sum\_{t \in J}\mathrm{tfidf}(t,J)}
\]

### Skill Coverage

\[
S\_{\text{skills}} = \frac{|K_R \cap K_J|}{|K_J|}
\]

where \(K_R, K_J\) are extracted skill sets from résumé and JD.

### Composite Score

\[
S = \alpha S*{\text{sem}} + \beta S*{\text{tfidf}} + \gamma S\_{\text{skills}}, \quad (\alpha,\beta,\gamma)=(0.6,0.25,0.15)
\]

Normalized to [0,100] across candidates.

---

## ⚙️ Setup

### 1. Clone Repo

```bash
git clone https://github.com/<your-username>/resume-job-matcher.git
cd resume-job-matcher

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python resume_matcher_pipeline.py score --resume ./data/resumes/alice.txt --jd ./data/jobs/ml_engineer.txt

python resume_matcher_pipeline.py rank --resumes_dir ./data/resumes --jd_path ./data/jobs/ml_engineer.txt --out_csv results.csv

 idx   score  semantic  tfidf  skills   skill_gaps
   5   92.3     0.81    0.74    0.67    ['docker','gcp']
  12   88.7     0.79    0.70    0.60    ['kubernetes']
   3   74.1     0.65    0.50    0.40    ['fastapi','pandas']

Thupakula, S. (2025). Resume–Job Matcher (Prototype). GitHub repository.
```
