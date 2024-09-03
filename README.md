# Automated-Essay-Scoring-using-LGBM-BERT

This project was developed as part of the Kaggle Competition: Learning Agency Lab - Automated Essay Scoring 2.0. The objective of the competition was to create a model capable of assigning scores to student essays in a manner that closely aligns with human grading. The challenge involved processing and analyzing textual data to generate meaningful predictions.

In this project, I implemented a hybrid approach that combines both traditional machine learning methods and state-of-the-art deep learning models. Specifically, the solution leverages:

* **LightGBM (LGBM):** A highly efficient and scalable gradient boosting framework that was utilized for its ability to handle large datasets and its superior performance on structured data.

* **Bidirectional Encoder Representations from Transformers (BERT):** A transformer-based model pre-trained on a vast corpus of text. BERT's ability to capture contextual nuances in language made it an ideal choice for understanding and scoring essays based on their content, structure, and coherence.

## Key Highlights of the Project:
* Comprehensive Text Preprocessing Pipeline:
  - The raw essay texts underwent a series of preprocessing steps to enhance model performance. This included tasks such as:
    - **HTML Tag Removal:** Stripping out any residual HTML tags from the text.
    - **Contraction Expansion:** Expanding common English contractions (e.g., "can't" to "cannot") to standardize the text.
    - **Lowercasing:** Converting all text to lowercase to ensure uniformity.
    - **Special Character and Number Removal:** Eliminating irrelevant characters, numbers, and URLs to reduce noise in the data.
    - **Spelling Error Detection:** Using Spacy to detect and quantify spelling errors, which could serve as an additional feature for grading.
    - **Punctuation Removal:** Stripping punctuation to further clean the text data.
* Feature Engineering:

  - In addition to raw text processing, the project explored various feature extraction techniques to convert the cleaned essays into a format suitable for model training. These features included:
    - **Count Vectorization:** Converting the text into a matrix of token counts.
    - **TF-IDF Vectorization:** Capturing the importance of words relative to their frequency across all essays.
    - **Custom Features:** Introducing additional features like the count of spelling errors, which could influence the essay score.
      
* Model Development:

  - The core of the project involved training multiple models to predict essay scores:
    - **DeBERTa for Sequence Classification:** Fine-tuned on the essay data to capture deep contextual relationships within the text, enabling it to predict scores with high accuracy.
    - **LightGBM:** Utilized for its speed and efficiency, particularly in handling the structured features derived from text vectorization.
    - **Ensemble of Machine Learning Models:** A combination of traditional models (e.g., Random Forest, AdaBoost) was trained alongside BERT and LGBM to create a robust ensemble, potentially improving generalization and reducing overfitting.
      
* Hyperparameter Tuning & Model Optimization:

  - The project included rigorous hyperparameter tuning using GridSearchCV and RandomizedSearchCV to optimize model parameters. This was crucial for enhancing the predictive power of the models and ensuring they performed well on unseen data.

* Evaluation and Performance:

  - After training, the models were evaluated using appropriate metrics (e.g., accuracy, F1-score) to ensure they met the competition’s requirements. The ensemble approach, along with the advanced NLP techniques employed, culminated in the project achieving a Bronze medal in the competition, placing it in the top tier of submissions.

## Competition Link:
The Link to the competition can be found at (https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)
## Additional Inputs:

* Due to the lack of executive GPUs for pretraining tasks, I only used GPUs offered by Kaggle to do inference and fine-tuning tasks and used additional resources that have been created, (or) trained and published by others on the Kaggle as the inputs. The links to the relevant inputs are listed below:

  -**English_Word_HX** : (https://www.kaggle.com/datasets/xianhellg/english-word-hx)
  
  -**AES2-400-FEs-202404291649** :(https://www.kaggle.com/datasets/hideyukizushi/aes2-400-fes-202404291649)
  
  -**AES2-400-20240419134941**: (https://www.kaggle.com/datasets/hideyukizushi/aes2-400-20240419134941/data)
  
  **-autogluon_cpu**: (https://www.kaggle.com/code/aikhmelnytskyy/autogluon-cpu)

  -**autogluon_gpu**:(https://www.kaggle.com/code/aikhmelnytskyy/autogluon-gpu)
