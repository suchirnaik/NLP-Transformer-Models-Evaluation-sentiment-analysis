# HW7 - Sentiment Analysis Comparison

## Pretrained Models

| Model                                           | Precision | Recall | F1-score | Accuracy |
|-------------------------------------------------|-----------|--------|----------|----------|
| Meohong/distilbert-base-uncased-finetuned-imdb  | 56.25     | 78     | 65.36    | 58.66    |
| fabriceyhc/bert-base-uncased-imdb              | 88.74     | 89.33  | 89.03    | 89       |
| aychang/roberta-base-imdb                      | 92        | 94     | 93.33    | 93.33    |

## Analysis and Observations

The evaluation metrics used to compare the models are Precision, Recall, F1-score, and Accuracy. Let's analyze each of these metrics for each model:

### Meohong/distilbert-base-uncased-finetuned-imdb

This model has the lowest performance compared to the other two models. It has a relatively low precision and accuracy, indicating that it has a high false positive rate and low overall accuracy. Although the recall is relatively high, the low precision score means that the model may be incorrectly classifying many reviews as positive when they are negative.

### fabriceyhc/bert-base-uncased-imdb

This model has a relatively high performance compared to the first model, with high precision, recall, F1-score, and accuracy. The accuracy score indicates that this model has correctly classified 89% of the reviews in the dataset. The precision and recall scores indicate that the model is able to correctly identify positive and negative reviews with high accuracy.

### aychang/roberta-base-imdb

This model has the highest performance compared to the other two models. It has the highest precision, recall, F1-score, and accuracy. The precision and recall scores indicate that the model is able to correctly identify positive and negative reviews with high accuracy. The accuracy score indicates that this model has correctly classified 93.33% of the reviews in the dataset.

In summary, based on the evaluation metrics provided, the aychang/roberta-base-imdb model has the highest performance, followed by fabriceyhc/bert-base-uncased-imdb, and Meohong/distilbert-base-uncased-finetuned-imdb has the lowest performance.

I also tested an additional Distilbert pretrained model called as huggingface-course/distilbert-base-uncased-finetuned-imdb. This gave me an accuracy of 52%.

The DistilBert pre-trained models did not seem to perform very well.

## Dataset and Preprocessing

- I have used the IMDB dataset.
- It was difficult to use the whole test dataset; therefore, I reduced the train dataset to 300 sample size.
- Also, the dataset had all labels with entries 0 entered in the first and labels with 1 in the second half. Therefore, we shuffled the dataset to select a random of 300 rows.
- Also, I ran the code on Colab, which helped me use the GPU for running the transformer code. Using the GPU helped considerably reduce the test time. It is recommended to use the GPU for training the data in this assignment.
- Also, I set the tokenizer required for each of the models to understand the input during classification. We used the Auto Tokenizer function from Transformers to aid us in creating the necessary tokens for the three different models.

## Additional Analysis

I conducted an additional analysis out of curiosity to see how the models would perform when I trained them on my own.

### Observation Table (Trained models - self training)

| Model    | Precision | Recall | F1-score | Accuracy |
|----------|-----------|--------|----------|----------|
| Roberta  | 89.80     | 94     | 91.85    | 91.6     |
| Bert     | 88.31     | 90.66  | 89.47    | 89.33    |
| Distilbert | 86.07  | 90.65  | 88.31    | 88       |

## Analysis and Observations

I fine-tuned the model and experimented with the trainer module to see how adjusting the parameters helps reduce the processing complexity.

I observed that Roberta performs the best out of the three models. It also took the longest to run out of the three models.

I ran the models twice to see if the results were consistent since the scores for DistilBert and Bert were very close to each other.

In both cases, the following was the order of performance (best to least):
1. Roberta
2. Bert
3. Distilbert

I trained the models and ran the test since DistilBert pre-trained models were not performing too well.

Training the model and running the tests proved to be more fruitful in my case for DistilBert.

## Limitations

- There are many pre-trained models on Hugging Face, and I was not able to test all of them due to the time constraint.
- There could have been more models that could have performed better in the list of pre-trained models on Hugging Face, especially in the case of Distilbert.
- However, the other two pretrained models performed better than when I trained the models on my own. This was because probably the pre-trained models were trained on the entire IMDB dataset of 25000 rows rather than my dataset for training which contained only 3000 rows.

## How I Could Use These Pre-trained Models in the Future

- There are many pre-trained models on IMDB, Twitter, etc. on Hugging Face.
- Since we are implementing sarcasm detection on Twitter data, probably using a pre-trained model for sarcasm can be very fruitful for us rather than training a model on our own. This can help us reduce the computational expenses required for running a transformer model on a large dataset.
- Also Hugging Face has many pre-trained models on different datasets. They can be useful to us in the future for sentiment analysis and sarcasm detection.
- We plan to continue our Sarcasm detection project in the future. Right now we have only trained and tested non-transformer models. Our hypothesis is that the transformer models would have a higher accuracy in detecting sarcasm. Using pre-trained models will turn out to be very helpful for us in the future.
