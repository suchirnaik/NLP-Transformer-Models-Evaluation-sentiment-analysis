import torch
torch.cuda.is_available()

#Assuming that the following libraries are installed
# Install the libraries
#!pip install datasets transformers huggingface_hub
#!apt-get install git-lfs

#Preprocess the data

# Load data
from datasets import load_dataset
imdb = load_dataset("imdb")

# Create a smaller training dataset due to resource limitations
#small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])
#print(small_train_dataset[0])
#print(small_test_dataset[0])

# Set DistilBERT tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Prepare the text inputs for the model
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

#tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# Use data_collector to convert our samples to PyTorch tensors and concatenate them with the correct amount of padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Training the model

#Distilbert

# Define DistilBERT as our base model:
from transformers import AutoModelForSequenceClassification,TFAutoModelForSequenceClassification
#model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
#model = AutoModelForSequenceClassification.from_pretrained("huggingface-course/distilbert-base-uncased-finetuned-imdb", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("Meohong/distilbert-base-uncased-finetuned-imdb", num_labels=2)

#model = TFAutoModelForSequenceClassification.from_pretrained('nateraw/bert-base-uncased-imdb', from_pt=True)

import numpy as np
from datasets import load_metric

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    load_precision = load_metric("precision")
    load_recall = load_metric("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    precision = load_precision.compute(predictions=predictions, references=labels)["precision"]
    recall = load_recall.compute(predictions=predictions, references=labels)["recall"]
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Define a new Trainer with all the objects we constructed so far
from transformers import TrainingArguments, Trainer

repo_name = "finetuning-sentiment-model-3000-samples"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch", 
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=tokenized_train,
    train_dataset=None,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
#trainer.train()



# Compute the evaluation metrics



evaluation_results = trainer.evaluate()
print(f"\n Meohong/distilbert-base-uncased-finetuned-imdb Evaluation Results:\n")
print(evaluation_results)


#Bert

# Define DistilBERT as our base model:
from transformers import AutoModelForSequenceClassification
#model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-imdb", num_labels=2)

# Set BERT tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Prepare the text inputs for the model
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

#tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)



# Define a new Trainer
from transformers import TrainingArguments, Trainer

repo_name = "finetuning-sentiment-model-3000-samples"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch", 
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=tokenized_train,
    train_dataset = None,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
#trainer.train()

# Compute the evaluation metrics

evaluation_results = trainer.evaluate()
print(f"\n fabriceyhc/bert-base-uncased-imdb Evaluation Results:\n")
print(evaluation_results)

#Roberta

import numpy as np
from datasets import load_metric

def compute_metrics(eval_pred):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
    load_precision = load_metric("precision")
    load_recall = load_metric("recall")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    precision = load_precision.compute(predictions=predictions, references=labels)["precision"]
    recall = load_recall.compute(predictions=predictions, references=labels)["recall"]
    
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

# Define DistilBERT as our base model:
from transformers import AutoModelForSequenceClassification
#model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("aychang/roberta-base-imdb", num_labels=2)

# Set BERT tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# Prepare the text inputs for the model
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

#tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# Define a new Trainer with all the objects we constructed so far
from transformers import TrainingArguments, Trainer

repo_name = "finetuning-sentiment-model-3000-samples"

training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch", 
    #push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=tokenized_train,
    train_dataset=None,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
#trainer.train()

evaluation_results = trainer.evaluate()
print(f"\n aychang/roberta-base-imdb Evaluation Results:\n")
print(evaluation_results)







