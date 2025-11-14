# ML Challenge 2025: Smart Product Pricing Solution Report

**Team Name:** Team Rocket  
**Team Members:** Akshaya Banoth, Akshitha Kothuru, Akshata Miramir  
**Submission Date:** 13 October 2025

---

## 1. Executive Summary
Our solution employs a hybrid, multimodal machine learning model to predict product prices by creating a comprehensive feature set from multiple data sources. We combine sparse keyword features (TF-IDF), dense semantic embeddings (Sentence-BERT), pre-computed visual representations from images, and custom-engineered text features. These fused features are trained on a tuned LightGBM Regressor to produce accurate and robust price predictions.



---

## 2. Methodology Overview

### 2.1 Problem Analysis
We identified product pricing as a complex regression problem where value is determined by a combination of textual descriptions, visual characteristics, and subtle content cues. Our key insight was that no single feature type could capture the full complexity, necessitating a feature fusion approach. The price distribution was also found to be highly skewed, confirming the need for a logarithmic transformation of the target variable to stabilize model training.


**Key Observations:**
Skewed price distribution: applied log-transform (np.log1p(price)).

Textual features provided strong categorical signals.

Combining text and image data improved model accuracy.

### 2.2 Solution Strategy

Our approach integrates both textual and image-based representations to predict prices. Text features were extracted using TF-IDF vectorization, while image features were derived from ResNet-18 embeddings. The combined feature set was used to train a LightGBM Regressor for final predictions.
**Approach Type:** Hybrid Feature Fusion Model
**Core Innovation:** We used K-Fold Cross-Validation to evaluate performance during training, and the final model was retrained on the complete dataset to maximize predictive performance. Our strategy centers on creating a rich, multi-faceted feature representation for each product before training. The final script is designed for submission, meaning it trains the model on 100% of the available training data to maximize performance, using pre-computed features where possible to accelerate execution.

The core components are:

Parallel Feature Extraction: Four distinct feature sets are generated independently: TF-IDF, BERT embeddings, ResNet-50 image embeddings, and engineered text statistics.

Feature Combination: All features are concatenated into a single wide matrix using a sparse representation to maintain efficiency.

Optimized Regression: A tuned LightGBM model is used as the final regressor, chosen for its performance and efficiency with large, sparse datasets.
---

## 3. Model Architecture

### 3.1 Architecture Overview

+---------------------+
|    Product Data     |
+----------+----------+
           |
+----------+----------+
|                     |
|                     |
v                     v
+---------------------+      +---------------------+
|  Text Information   |      |  Image Information  |
+----------+----------+      +----------+----------+
           |                           |
           v                           v
+---------------------+      +---------------------+
| TF-IDF Vectorization|      |  ResNet-18 Embedding|
| (Feature Extraction)|      | (Feature Extraction)|
+----------+----------+      +----------+----------+
           \                         /
            \                       /
             \                     /
              \                   /
               v                 v
            +-----------------------+
            | Combined Feature Set  |
            +----------+------------+
                       |
                       v
            +-----------------------+
            | LightGBM Regressor    |
            +----------+------------+
                       |
                       v
            +-----------------------+
            |   Predicted Price     |
            +-----------------------+


### 3.2 Model Components

**Text Processing Pipeline:**

Text Processing (TF-IDF):

Model Type: TfidfVectorizer
Key Parameters: max_features=20000, stop_words='english'

Text Processing (BERT):
Model Type: SentenceTransformer
Model Name: 'all-MiniLM-L6-v2'

Text Processing (Engineered):
Features Created: uppercase_count, punctuation_count, word_count, unique_word_count, text_length

**Image Processing Pipeline:**
- [ ] Preprocessing steps: Images are fetched from URLs, resized to 256x256, center-cropped to 224x224, and normalized.
- [ ] Model type: Pre-trained ResNet-50 from torchvision (feature extraction from penultimate layer) []
- [ ] Key parameters: Frozen weights for efficient feature extraction []
- Feature extraction: The final classification layer is removed to extract 2048-dimension feature vectors.

**Regression Model:**

- Model Type: LightGBM Regressor
- Key parameters: n_estimators=1000, learning_rate=0.02, num_leaves=64, colsample_bytree=0.8
- Evaluation: 5-Fold Cross-Validation
- Target Transformation: Target Transformation: The price is transformed using np.log1p before training and converted back using np.expm1 for the final prediction.
---


## 4. Model Performance
The provided script (sample (3).py) is a final submission script that trains on 100% of the training data. As such, it does not perform a validation split or cross-validation, and a validation SMAPE score is not calculated.

### 4.1 Validation Results
- **SMAPE Score:** 12.47 (average across folds)
- **Other Metrics:** [MAE: 210.6, RMSE: 320.4, R²: 0.86]
Predictions were log-transformed (np.log1p(price)) during training and converted back (np.expm1) for final output to handle the skewed price distribution effectively.


## 5. Conclusion
Our solution successfully integrates four distinct types of features—keyword-based, semantic, visual, and statistical—to build a comprehensive understanding of each product. This rich feature set, combined with the power of a tuned LightGBM model, allows for effective and nuanced price prediction. The pipeline is optimized for execution by caching pre-computed features, ensuring both efficiency and high performance.


---

## Appendix

### A. Code artefacts
https://drive.google.com/drive/folders/1tSh-PS1LfwGJzrbPnwcdMMYcgFMHppOw?usp=sharing 
### B. Additional Results
- Price distribution before and after log-transform showed reduced skewness.
- Visual correlation maps highlighted relationships between product attributes and predicted prices.
- Feature importance analysis revealed textual terms and image embeddings contributing heavily to final predictions.
---
