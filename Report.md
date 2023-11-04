# NLP Project: Pretrained Transformer Models Evaluation

## Overview
This is my NLP project report where I've explored and compared the performance of pretrained transformer models. In this project, I conducted evaluations and made observations on several pretrained models, focusing on precision, recall, F1-score, and accuracy. 

### Pretrained Models and Observations
I evaluated the following pretrained models and recorded their performance:

| Model                                           | Precision | Recall | F1-score | Accuracy |
| ----------------------------------------------- | --------- | ------ | -------- | -------- |
| Meohong/distilbert-base-uncased-finetuned-imdb  | 56.25     | 78     | 65.36    | 58.66    |
| fabriceyhc/bert-base-uncased-imdb               | 88.74     | 89.33  | 89.03    | 89       |
| aychang/roberta-base-imdb                       | 92        | 94     | 93.33    | 93.33    |

### Analysis and Observations
The evaluation metrics used for model comparison are Precision, Recall, F1-score, and Accuracy. Here are the observations for each model:

**Meohong/distilbert-base-uncased-finetuned-imdb:**
This model exhibited lower performance compared to the others. It has a low precision and accuracy, suggesting a high false positive rate and reduced overall accuracy. Despite relatively high recall, its low precision may lead to misclassification of reviews.

**fabriceyhc/bert-base-uncased-imdb:**
This model showed relatively high performance, with impressive precision, recall, F1-score, and accuracy. The accuracy score of 89% indicates a strong classification capability. High precision and recall demonstrate accurate identification of positive and negative reviews.

**aychang/roberta-base-imdb:**
Among the three models, this one performed the best. It boasted the highest precision, recall, F1-score, and accuracy. Its impressive precision and recall scores indicate precise identification of positive and negative reviews, with a remarkable accuracy rate of 93.33%.

In summary, based on the evaluation metrics, the order of performance is aychang/roberta-base-imdb > fabriceyhc/bert-base-uncased-imdb > Meohong/distilbert-base-uncased-finetuned-imdb. 

An additional Distilbert pretrained model, huggingface-course/distilbert-base-uncased-finetuned-imdb, was also tested, which achieved an accuracy of 52%. It's worth noting that the DistilBert pretrained models did not perform very well in this context.

### Dataset and Preprocessing
- The IMDB dataset was used.
- Due to resource limitations, the training dataset was reduced to 300 samples.
- The dataset was shuffled to maintain randomness in selecting 300 rows.
- Google Colab with GPU support was used to expedite processing.
- Auto Tokenizer functions from the Transformers library were used to create necessary tokens for the models.

### Additional Analysis
Intrigued by the model performance, additional analysis was conducted. Here are the results:

| Model       | Precision | Recall | F1-score | Accuracy |
| ----------- | --------- | ------ | -------- | -------- |
| Roberta     | 89.80     | 94     | 91.85    | 91.6     |
| Bert        | 88.31     | 90.66  | 89.47    | 89.33    |
| Distilbert  | 86.07     | 90.65  | 88.31    | 88       |

**Analysis and Observations (Trained Models - Self-Training):**
- The IMDB dataset was used, but the training dataset was reduced to 3000 samples.
- Data was shuffled for randomization.
- Google Colab with GPU support was used for faster processing.
- AutoTokenizer functions were employed for tokenization.

**Findings:**
- Roberta achieved the highest performance, but it was the slowest.
- Two runs showed consistent performance with the order: Roberta > Bert > Distilbert.

**Limitations:**
- Time constraints limited the evaluation of all available pretrained models on Hugging Face.
- There may be more models with superior performance that were not tested, particularly for Distilbert.
- Pretrained models may perform differently when trained on a larger dataset compared to the small dataset used for fine-tuning.

### Future Usage of Pretrained Models
- Hugging Face offers a wide range of pretrained models on IMDb, Twitter, and other datasets.
- For future sarcasm detection on Twitter data, using a pretrained sarcasm model could reduce computational expenses compared to training a model from scratch.
- Pretrained models can be valuable for future sentiment analysis and sarcasm detection research, enhancing model accuracy.

Feel free to reach out for any questions or further information.

