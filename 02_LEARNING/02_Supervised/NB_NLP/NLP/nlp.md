# Learning Brief — Natural Language Processing (NLP)

Dokumen ini menjelaskan **tujuan, prasyarat, dataset, alur belajar, dan output** untuk materi **Natural Language Processing dengan Naive Bayes** pada data teks sesuai dengan RPS dan Sub-CPMK.

---

## Tujuan Pembelajaran

Setelah menyelesaikan fase ini, mahasiswa mampu:

* Memahami konsep text preprocessing untuk NLP (cleaning, tokenization, normalization)
* Mengimplementasi text vectorization dengan Bag of Words (BoW) dan TF-IDF
* Memahami perbedaan Multinomial Naive Bayes dengan Gaussian Naive Bayes
* Melakukan sentiment analysis dan text classification
* Menerapkan stopwords removal dan stemming/lemmatization
* Menangani text preprocessing untuk bahasa Indonesia
* Menganalisis distribusi kata dengan WordCloud
* Mengevaluasi model NLP dengan classification metrics
* Membandingkan performa BoW vs TF-IDF

**Sub-CPMK terkait:**

> Mahasiswa mampu mengimplementasi dan mengevaluasi model naive bayes untuk text classification dan NLP. (C5, P4, A4)

---

## Prasyarat (SUDAH TERPENUHI)

✔ Naive Bayes fundamentals (Bayes theorem, conditional probability)
✔ Classification metrics (precision, recall, F1, ROC-AUC)
✔ Train-test split
✔ Feature engineering
✔ Label encoding

---

## Konsep Inti yang Dipelajari

### 1. Text Preprocessing

* **Text Cleaning** - Menghapus noise dari teks (URL, HTML tags, punctuation, numbers)
* **Lowercasing** - Normalisasi huruf besar/kecil
* **Stopwords Removal** - Menghapus kata-kata umum yang tidak informatif
* **Stemming** - Memotong kata ke bentuk dasar (SnowballStemmer)
* **Lemmatization** - Mengubah kata ke bentuk lemma (WordNetLemmatizer)
* **Emoji Removal** - Menghapus emoji dari teks
* **Slang Normalization** - Mengubah bahasa tidak baku ke baku

### 2. Text Vectorization

* **Bag of Words (BoW)** - CountVectorizer untuk word frequency
* **TF-IDF** - Term Frequency-Inverse Document Frequency untuk word importance
* **N-grams** - Unigram dan bigram untuk context
* **Max Features** - Limiting vocabulary size untuk efficiency
* **Sparse Matrix** - Representasi efisien untuk high-dimensional data

### 3. Multinomial Naive Bayes

* **Discrete Features** - Untuk data count-based (word frequency)
* **Probability Distribution** - Multinomial distribution untuk word counts
* **Smoothing** - Laplace smoothing untuk zero probability
* **Binary vs Multiclass** - Binary classification dan multiclass classification

### 4. Text Analysis

* **WordCloud** - Visualisasi distribusi kata
* **Text Length Distribution** - Analisis panjang teks per class
* **Vocabulary Analysis** - Analisis kata-kata dominan per sentiment
* **Token Distribution** - Distribusi token setelah preprocessing

---

## Alur Belajar (STEP-BY-STEP)

---

## STEP 1 — SMS Spam Classification

### Tujuan

Memahami binary text classification dengan Naive Bayes pada dataset SMS spam menggunakan Bag of Words.

### Dataset

* **SMS Spam Collection Dataset** (UCI ML Repository)
* Source: Kaggle (uciml/sms-spam-collection-dataset)
* Target: label (ham atau spam)
* Fitur: message (text SMS)
* Total: 5572 SMS messages
* Distribution: Imbalanced (ham >> spam)

### Data Information

* Columns: v1 (label), v2 (message), Unnamed columns
* No missing values setelah cleaning
* Feature engineering: message_length (jumlah kata)
* Binary classification problem

### Feature Engineering

* **Text Cleaning**
  - Lowercase conversion
  - Remove URLs, HTML tags
  - Remove punctuation
  - Remove numbers
  - Remove brackets

* **Stopwords Removal**
  - English stopwords from NLTK
  - Custom stopwords: 'u', 'im', 'c'
  - Remove common non-informative words

* **Stemming**
  - SnowballStemmer untuk bahasa Inggris
  - Reduce words to root form

* **Label Encoding**
  - ham → 0
  - spam → 1

### Model

* **Vectorizer**: CountVectorizer (Bag of Words)
  - No max_features limit
  - Default n-gram (unigram)
  - Fit pada training set
  - Transform pada test set

* **Classifier**: MultinomialNB
  - Default parameters
  - No hyperparameter tuning

### Pipeline

* Manual pipeline (no sklearn Pipeline):
  1. Text cleaning
  2. Stopwords removal
  3. Stemming
  4. Label encoding
  5. Train-test split (80-20)
  6. CountVectorizer fit_transform
  7. MultinomialNB fit
  8. Predict dan evaluate

### Hasil Training

* Metrics (pada test set):
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

* Visualizations:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
  - WordCloud (before dan after preprocessing)
  - Text length distribution per label

### Fokus Analisis

* Impact stemming pada vocabulary
* Distribusi panjang SMS spam vs ham
* Common words dalam spam messages
* Model performance pada imbalanced data
* Precision-Recall trade-off untuk spam detection

### Output

* Notebook: `SMS_Spam_NB_NLP.ipynb`
* Pemahaman text preprocessing pipeline
* Implementasi CountVectorizer
* Baseline spam classification model

---

## STEP 2 — Movie Review Sentiment Analysis

### Tujuan

Mengaplikasikan Naive Bayes untuk sentiment analysis pada movie reviews dengan membandingkan Bag of Words dan TF-IDF.

### Dataset

* **IMDB Dataset of 50K Movie Reviews**
* Source: Kaggle (lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* Target: sentiment (positive atau negative)
* Fitur: review (text review)
* Total: 50,000 reviews
* Distribution: Balanced (25K positive, 25K negative)

### Data Information

* Columns: review, sentiment
* No missing values
* Feature engineering: review_len (jumlah kata)
* Binary sentiment classification
* Longer reviews dibanding SMS

### Feature Engineering

* **Text Cleaning**
  - Lowercase conversion
  - Remove URLs, HTML tags (penting untuk web-scraped data)
  - Remove punctuation
  - Remove numbers
  - Remove brackets

* **Stopwords Removal**
  - English stopwords from NLTK
  - Custom stopwords: 'u', 'im', 'c'

* **Lemmatization** (berbeda dari SMS)
  - WordNetLemmatizer (bukan stemming)
  - Preserve word meaning lebih baik
  - Lebih cocok untuk longer text

* **Label Encoding**
  - negative → 0
  - positive → 1

### Model

* **Model 1: Bag of Words**
  - CountVectorizer dengan max_features=5000
  - ngram_range=(1,2) - unigram dan bigram
  - MultinomialNB classifier

* **Model 2: TF-IDF**
  - TfidfVectorizer dengan max_features=5000
  - ngram_range=(1,2) - unigram dan bigram
  - MultinomialNB classifier

### Pipeline

* Dua pipeline terpisah untuk BoW dan TF-IDF:
  1. Text cleaning
  2. Stopwords removal
  3. Lemmatization
  4. Label encoding
  5. Train-test split (80-20)
  6. Vectorization (BoW atau TF-IDF)
  7. MultinomialNB training
  8. Prediction dan evaluation
  9. Comparison BoW vs TF-IDF

### Hasil Training

* **BoW Model Performance**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC
  - Precision-Recall Curve

* **TF-IDF Model Performance**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC-AUC
  - Precision-Recall Curve

* **Comparison**:
  - BoW vs TF-IDF metrics
  - Performance differences
  - Kelebihan masing-masing method

### Fokus Analisis

* Lemmatization vs Stemming untuk longer text
* Impact bigrams pada classification
* BoW vs TF-IDF performance comparison
* Handling balanced sentiment data
* Vocabulary analysis positive vs negative reviews
* Max_features impact pada model size

### Output

* Notebook: `Movie_review_BoW_TFIDF.ipynb`
* Comparison BoW vs TF-IDF
* Lemmatization implementation
* N-gram analysis

---

## STEP 3 — Tokopedia Review Sentiment Analysis (Bahasa Indonesia)

### Tujuan

Implementasi NLP untuk bahasa Indonesia dengan multiclass sentiment classification dan handling Indonesian text preprocessing.

### Dataset

* **Tokopedia Product Reviews 2025**
* Source: Kaggle (salmanabdu/tokopedia-product-reviews-2025)
* Target: sentiment_label (positive, negative, neutral)
* Fitur: review_text
* Multiple columns: product info, shop info, rating, sold_count
* Multiclass classification problem

### Data Information

* Original columns:
  - review_id, review_date
  - product_name, product_category, product_variant, product_price, product_url, product_id
  - shop_id, rating, sold_count
  - review_text, sentiment_label

* Dropped columns: semua kecuali review_text dan sentiment_label
* Feature engineering: review_len (jumlah kata)
* Three-class classification: positive, negative, neutral
* Indonesian language text

### Feature Engineering

* **Text Cleaning**
  - Lowercase conversion
  - Remove URLs, HTML tags
  - Remove punctuation
  - Remove numbers
  - Remove brackets

* **Emoji Removal**
  - Custom regex untuk Unicode emoji ranges
  - Important untuk social media text

* **Slang Normalization** (khusus Bahasa Indonesia)
  - Dictionary mapping: gk→tidak, ga→tidak, nggak→tidak
  - bgt→banget, bgus→bagus, mantul→mantap
  - tdk→tidak, yg→yang, dgn→dengan
  - krn→karena, jg→juga, sdh→sudah
  - sy→saya, tp→tapi, dr→dari, brg→barang

* **Stopwords Removal**
  - Sastrawi library untuk Indonesian stopwords
  - Custom StopWordRemoverFactory

* **Lemmatization**
  - WordNetLemmatizer (English fallback)
  - Ideally should use Indonesian stemmer (Sastrawi)

* **Label Encoding**
  - negative → 0
  - neutral → 1
  - positive → 2

### Model

* **Model 1: Bag of Words**
  - CountVectorizer dengan max_features=5000
  - ngram_range=(1,2)
  - MultinomialNB classifier
  - Multiclass classification

* **Model 2: TF-IDF**
  - TfidfVectorizer dengan max_features=5000
  - ngram_range=(1,2)
  - MultinomialNB classifier
  - Multiclass classification

### Pipeline

* Preprocessing pipeline:
  1. Text cleaning (lowercase, remove noise)
  2. Emoji removal
  3. Slang normalization
  4. Stopwords removal (Sastrawi)
  5. Lemmatization
  6. Label encoding
  7. Train-test split (80-20)

* Model training:
  - BoW vectorization → MultinomialNB
  - TF-IDF vectorization → MultinomialNB

### Hasil Training

* **BoW Model Performance**:
  - Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted)
  - Classification Report (per-class metrics)
  - Confusion Matrix (3x3)
  - ROC-AUC (multiclass) dengan Yellowbrick

* **TF-IDF Model Performance**:
  - Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted)
  - Classification Report (per-class metrics)
  - Confusion Matrix (3x3)
  - ROC-AUC (multiclass) dengan Yellowbrick

* **Visualizations**:
  - WordCloud untuk positive, negative, neutral (sebelum dan sesudah preprocessing)
  - Histogram distribusi panjang review per sentiment
  - ROC curves untuk multiclass (One-vs-Rest)

### Fokus Analisis

* Challenges preprocessing Bahasa Indonesia
* Slang normalization importance
* Multiclass vs binary classification
* Neutral class performance (often harder to predict)
* BoW vs TF-IDF untuk Indonesian text
* Weighted metrics untuk imbalanced multiclass
* Yellowbrick ROCAUC untuk multiclass visualization

### Output

* Notebook: `tokped_review_nbnlp.ipynb`
* Indonesian text preprocessing
* Multiclass sentiment analysis
* Sastrawi integration
* Slang dictionary implementation

---

## Evaluasi Kritis & Insight

### Perbedaan Gaussian vs Multinomial Naive Bayes

* **Gaussian NB** (untuk data tabular):
  - Continuous features
  - Normal distribution assumption
  - Used in NB_Learn, CC_NB, Diabetes_NB

* **Multinomial NB** (untuk text data):
  - Discrete features (word counts)
  - Multinomial distribution
  - Used in SMS, Movie, Tokped reviews

### Text Preprocessing Comparison

* **Stemming vs Lemmatization**:
  - Stemming: faster, aggressive, may lose meaning
  - Lemmatization: slower, preserves meaning, better untuk longer text
  - SMS Spam: stemming (short text, speed priority)
  - Movie Review: lemmatization (longer text, meaning priority)
  - Tokopedia: lemmatization

* **Language-Specific Challenges**:
  - English: NLTK stopwords, SnowballStemmer, WordNetLemmatizer
  - Indonesian: Sastrawi stopwords, slang normalization, emoji handling

### BoW vs TF-IDF

* **Bag of Words**:
  - Simple word frequency
  - Good untuk short text
  - Tidak mempertimbangkan word importance
  - Faster computation

* **TF-IDF**:
  - Word importance berdasarkan document frequency
  - Better untuk longer text
  - Downweight common words
  - More sophisticated

* **Performance**:
  - Dataset-dependent
  - Movie Review: compare BoW vs TF-IDF
  - Tokopedia: compare BoW vs TF-IDF
  - Biasanya TF-IDF slightly better untuk balanced data

### Binary vs Multiclass Classification

* **Binary** (SMS Spam, Movie Review):
  - Simpler decision boundary
  - Standard precision-recall trade-off
  - Clear ROC-AUC interpretation

* **Multiclass** (Tokopedia):
  - More complex decision boundaries
  - Weighted averaging untuk metrics
  - One-vs-Rest ROC curves
  - Neutral class often hardest

### Kelebihan Naive Bayes untuk NLP

* Fast training dan prediction
* Works well dengan high-dimensional sparse data
* Good baseline untuk text classification
* Probabilistic output untuk confidence
* Tidak butuh banyak training data
* Handles large vocabulary well

### Keterbatasan Naive Bayes untuk NLP

* Asumsi independence antar kata (tidak realistis)
* Tidak capture word order
* Tidak handle context dan semantics
* Limited untuk complex language understanding
* Zero probability problem (butuh smoothing)

### Kapan Menggunakan NB untuk NLP

✔ Spam detection
✔ Sentiment analysis (basic)
✔ Topic classification
✔ Quick baseline model
✔ Real-time classification
✔ Limited computational resources
✔ Small to medium datasets

### Kapan TIDAK Menggunakan NB untuk NLP

✖ Complex semantic understanding
✖ Need untuk capture word order (use RNN, LSTM)
✖ Sarcasm detection
✖ Context-dependent meaning
✖ Need state-of-the-art accuracy (use transformers)

---

## Text Preprocessing Best Practices

### Essential Steps

1. **Lowercase** - Always normalize case
2. **Remove Noise** - URLs, HTML, special characters
3. **Tokenization** - Split text into words
4. **Stopwords Removal** - Remove non-informative words
5. **Normalization** - Stemming atau lemmatization

### Optional Steps (Context-Dependent)

* Emoji removal (for formal text)
* Slang normalization (for social media)
* Number removal (if not informative)
* Punctuation handling (keep for some tasks)

### Language-Specific

* **English**: NLTK, SpaCy
* **Indonesian**: Sastrawi, custom dictionaries
* **Multilingual**: Consider language detection first

---

## Struktur Direktori

```
02_LEARNING/02_Supervised/NB_NLP/NLP/
│
├── SMS_Spam_NB_NLP.ipynb          # Binary classification - SMS spam
├── Movie_review_BoW_TFIDF.ipynb   # Binary sentiment - BoW vs TF-IDF
├── tokped_review_nbnlp.ipynb      # Multiclass sentiment - Bahasa Indonesia
└── nlp.md                         # Dokumentasi pembelajaran (file ini)
```

---

## Kriteria "SUDAH PAHAM"

Kamu boleh lanjut ke materi berikutnya kalau:

* Bisa menjelaskan perbedaan Gaussian vs Multinomial Naive Bayes
* Paham text preprocessing pipeline (cleaning, stopwords, stemming/lemmatization)
* Bisa mengimplementasi CountVectorizer dan TfidfVectorizer
* Paham perbedaan BoW dan TF-IDF
* Bisa handling binary dan multiclass text classification
* Paham n-grams dan max_features impact
* Bisa preprocessing bahasa Indonesia dengan Sastrawi
* Paham weighted metrics untuk multiclass
* Bisa membaca confusion matrix untuk multiclass
* Paham kapan pakai stemming vs lemmatization

Kalau belum → **ulangi dan pelajari lebih dalam**

---

## Progression Path

### Sudah Dikuasai

✔ Naive Bayes fundamentals (tabular data)
✔ Text preprocessing (cleaning, tokenization, normalization)
✔ Text vectorization (BoW, TF-IDF)
✔ Multinomial Naive Bayes
✔ Binary dan multiclass classification
✔ Indonesian text processing

### Next Steps

1. **Advanced NLP**:
   - Word embeddings (Word2Vec, GloVe)
   - Deep learning for NLP (RNN, LSTM, GRU)
   - Transformer models (BERT, GPT)
   - Named Entity Recognition (NER)
   - Topic modeling (LDA)

2. **Other Classification Algorithms**:
   - Support Vector Machines (SVM)
   - Decision Trees
   - Random Forest
   - XGBoost

3. **Advanced Text Features**:
   - Character n-grams
   - POS tagging
   - Dependency parsing
   - Sentiment lexicons

---

## Catatan

> Naive Bayes untuk NLP = **fast baseline untuk text classification**
> Multinomial NB = untuk count-based features (word frequency)
> Text preprocessing sangat penting untuk performa model
> BoW vs TF-IDF = trade-off simplicity vs sophistication
> Bahasa Indonesia butuh special handling (Sastrawi, slang normalization)

---

## Key Takeaways

### Dataset Characteristics

* **SMS Spam**: Short text, binary, imbalanced, English
* **Movie Review**: Long text, binary, balanced, English, HTML tags
* **Tokopedia**: Medium text, multiclass, Indonesian, slang & emoji

### Preprocessing Techniques

* **All**: Cleaning, lowercase, remove punctuation
* **SMS**: Stemming (SnowballStemmer)
* **Movie**: Lemmatization (WordNetLemmatizer)
* **Tokopedia**: Slang normalization, Sastrawi, emoji removal

### Vectorization

* **SMS**: BoW only (baseline)
* **Movie**: BoW vs TF-IDF comparison
* **Tokopedia**: BoW vs TF-IDF comparison
* **All**: N-grams (1,2), max_features control

### Classification

* **Binary**: SMS (ham/spam), Movie (pos/neg)
* **Multiclass**: Tokopedia (pos/neg/neutral)
* **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
* **Multiclass Metrics**: Weighted averaging

### Performance Factors

* Text length (short vs long)
* Class balance (balanced vs imbalanced)
* Language (English vs Indonesian)
* Vocabulary size (max_features)
* N-grams (unigram vs bigram)
* Vectorization method (BoW vs TF-IDF)

---
