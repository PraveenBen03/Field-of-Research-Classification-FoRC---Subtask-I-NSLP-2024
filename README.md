# Categorizing Knowledge: Approaches For Effective Research Paper Classification

**Authors**  
Saliq Gowhar  
Praveen Kempaiah  
Dr. Sowmya Kamath S  
Department of Information Technology, National Institute of Technology Karnataka

**Contact**  
Email: [saliqgowhar.211ee250@nitk.edu.in](mailto:saliqgowhar.211ee250@nitk.edu.in), [praveenk.211ai028@nitk.edu.in](mailto:praveenk.211ai028@nitk.edu.in), [sowmyakamath@nitk.edu.in](mailto:sowmyakamath@nitk.edu.in)

**Abstract**  
This study focuses on classifying scientific articles into one of 123 predefined fields of research using Natural Language Processing (NLP) techniques. We explored various methods including Sentence Transformer embeddings, Machine Learning algorithms, Neural Networks, and Transformers. Our approach involved testing SVM, Random Forest, KNN, XGBoost, and Logistic Regression models, as well as neural network architectures and transformer models like BERT and its variants. We achieved top rankings in the FoRC Shared Task Subtask 1 by using MPNet embeddings with a One vs Rest (OvR) classifier.

**Introduction**  
As the number of scientific papers continues to grow, effective classification systems are essential for managing and accessing relevant research. This paper investigates various classification methods, including embeddings, machine learning models, neural networks, and transformers, to enhance the accuracy and efficiency of research paper categorization.

**Related Work**  
- **Gündoğan et al.**: Used Word2Vec and community discovery techniques.  
- **Chowdhury et al.**: Applied SVM, RF, and DTs for classification.  
- **Kadhim et al.**: Examined term weighting methods.  
- **Dien et al.**: Achieved 91% accuracy using SVM.  
- **Atmadja et al.**: Utilized Naive Bayes for classification with a small corpus.

**Dataset**  
The dataset used consists of 59.3K scholarly papers classified into 123 fields of research, sourced from ORKG, arXiv, Crossref API, and S2AG. It includes metadata like titles, authors, DOIs, publication dates, and abstracts. The dataset is imbalanced, with varying numbers of articles per field.

**Methodology**

- **Preprocessing**  
  - Removal of unimportant columns and missing values.  
  - Concatenation of Title and Abstract.  
  - Removal of HTML tags and non-alphanumeric characters.  
  - Label encoding of target labels.

- **Synthesis and Augmentation**  
  - Synthetic data generation using Gretel LLM for classes with fewer than 100 entries.  
  - Synonym augmentation for classes with fewer instances.  
  - Data augmentation improved dataset size but did not substantially enhance classification performance.

- **Embedding Generation**  
  - Utilized Sentence Transformers like MiniLM-L6 and MP-Net.  
  - MP-Net embeddings outperformed others in classification tasks.

- **Machine Learning Algorithms**  
  - Models: SVM, Random Forest, Logistic Regression, XGBoost, KNN.  
  - SVM with MP-Net embeddings and One vs Rest (OvR) classifier achieved the best results.

- **Neural Networks**  
  - Experimented with various architectures, including increasing and decreasing pyramidal structures.  
  - Focal loss function was tested but did not perform effectively due to class imbalance.

- **Transformers**  
  - Tested BERT, SciBERT, RoBERTa, DistilBERT, and ALBERT.  
  - SciBERT performed best, particularly with embeddings from the pooler layer.

**Results**

**Validation Set Performance**

| Model                   | Accuracy | Precision | Recall | F1-Score |
|-------------------------|----------|-----------|--------|----------|
| OVR - MPnet             | 0.75     | 0.75      | 0.75   | 0.75     |
| OVR - MiniLM            | 0.73     | 0.72      | 0.73   | 0.72     |
| SVM                     | 0.71     | 0.71      | 0.71   | 0.71     |
| LLM Synthesis           | 0.71     | 0.71      | 0.71   | 0.70     |
| Synonym Replacement     | 0.71     | 0.71      | 0.71   | 0.70     |
| KNN                     | 0.69     | 0.69      | 0.69   | 0.69     |
| ANN - Decreasing Pyramid | 0.67     | 0.67      | 0.67   | 0.67     |
| Logistic Regression     | 0.68     | 0.67      | 0.68   | 0.67     |
| ANN - Increasing Pyramid | 0.67     | 0.67      | 0.67   | 0.66     |
| SciBERT                 | 0.67     | 0.65      | 0.67   | 0.65     |
| ANN - Focal Loss        | 0.65     | 0.65      | 0.65   | 0.64     |
| BERT                    | 0.61     | 0.65      | 0.61   | 0.60     |
| DistilBERT              | 0.59     | 0.64      | 0.59   | 0.59     |
| XGBOOST                 | 0.59     | 0.58      | 0.59   | 0.58     |
| Random Forest           | 0.57     | 0.59      | 0.57   | 0.54     |
| 1D CNN                  | 0.49     | 0.48      | 0.49   | 0.48     |
| BERT - Pooler Unfreezed | 0.43     | 0.51      | 0.43   | 0.45     |
| BERT - Freezed          | 0.44     | 0.55      | 0.44   | 0.44     |
| ALBERT                  | 0.20     | 0.27      | 0.20   | 0.20     |
| RoBERTa                 | 0.23     | 0.32      | 0.23   | 0.18     |

**Official Results**

| User/Team    | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| HALE LAB NITK| 0.7572   | 0.7536    | 0.7572 | 0.7500   |
| Rosni        | 0.7558   | 0.7566    | 0.7558 | 0.7540   |
| Flo.ruo      | 0.7542   | 0.7545    | 0.7542 | 0.7524   |

**Conclusion**  
The study demonstrates the efficacy of traditional machine learning algorithms, especially SVM with MP-Net embeddings and OvR classification, in handling research paper classification tasks. While transformer models and neural networks have their strengths, traditional methods proved more effective in this scenario, particularly when dealing with imbalanced datasets.

**Future Work**  
Future research could explore alternative classification methods, leveraging advancements in LLMs, NLP techniques like named entity recognition, and additional approaches to address class imbalance and improve classification accuracy.

**References**  
Please refer to the list of references in the research paper for detailed information on related works and methodologies.
