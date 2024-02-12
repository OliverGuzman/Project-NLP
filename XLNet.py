#%%
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification, create_optimizer
import evaluate
from transformers.keras_callbacks import KerasMetricCallback
import tensorflow as tf
import pandas as pd
import numpy as np

'''numpy lastest version does not have attribute .object'''
np.object = np.object_

#%%
'''Load the dataset using a transformers function'''
dataset = load_dataset("imdb")

#%%
'''Automatically tokenize the input according to the model'''
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

#%%
'''Create a preprocessing function to tokenize text'''
def preprocess_function(examples):
    return tokenizer(examples["text"])

#%%
'''It applies the preprocesing function over the dataset and it is faster because 
it parallelizes the tokenization of all the examples in a batch'''
tokenized_imdb = dataset.map(preprocess_function, batched=True)

#%%
'''It dynamically pads the sentences to the longest length in a batch during collation "building a batch"'''
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

#%%
'''Load metric for model evaluation'''
accuracy = evaluate.load("accuracy")

#%%
'''create a function that passes your predictions and labels to compute to calculate the accuracy'''
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

#%%
'''create a map of the expected ids to their labels with id2label and label2id'''
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

#%%
'''optimizer function, learning rate schedule,and batch size'''
batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

#%%
'''This will instantiated the xlnet-base-cased model, it will pass
the number of columms and a maps for converting the labels created above'''
model = TFAutoModelForSequenceClassification.from_pretrained(
    "xlnet-base-cased", num_labels=2, id2label=id2label, label2id=label2id
)

#%%
'''Convert datasets to the tf.data.Dataset format for the model'''
tf_train_set = model.prepare_tf_dataset(
    tokenized_imdb["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)
tf_validation_set = model.prepare_tf_dataset(
    tokenized_imdb["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

#%%
'''Default loss function for the xlnet-base-cased model and base optimizer class.
Additionally, the metric accuracy is past so iti is shown during training'''
model.compile(optimizer=optimizer,metrics=['accuracy'])

#%%
'''Receive the function for calculating the accuracy and the validation 
dataset which are used to compute the mettric at the end of
each epoch'''
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
callbacks = [metric_callback]

#%%
'''Train the model'''
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)

#Using this model, kaggle was not able to allocate enough resources for its training
