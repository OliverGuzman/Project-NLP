#%%
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification, create_optimizer
import evaluate
from transformers.keras_callbacks import KerasMetricCallback
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import datasets

'''numpy lastest version does not have attribute .object'''
np.object = np.object_

#%%
'''Load the dataset using a transformers function'''
dataset = load_dataset("imdb")

#%%
'''Automatically tokenize the input according to the model'''
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

#%%
'''Create a preprocessing function to tokenize text and truncate sequences to be no 
longer than distilroberta-base maximum input length'''
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

#%%
'''It applies the preprocesing function over the dataset and it is faster because  it parallelizes the tokenization of all the examples in a batch'''
tokenized_imdb = dataset.map(preprocess_function, batched=True)

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
'''optimizer selected for the task'''
optimizer = Adam(2e-5)

#%%
'''This will instantiated the distilroberta-base model, it will pass
the number of columms and a maps for converting the labels created above'''
model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base", num_labels=2, id2label=id2label, label2id=label2id)

#%%
'''It dynamically pads the sentences to the longest length in a batch during collation "building a batch"'''
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

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
'''Default loss function for the distilroberta-base model and base optimizer class.
Additionally, the metric accuracy is past so iti is shown during training'''
model.compile(optimizer=optimizer,metrics=['accuracy'])

#%%
'''Receive the function for calculating the accuracy and the validation 
dataset which are used to compute the mettric at the end of
each epoch'''
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

#This function was created to track the info during training steps
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.losses = []
        self.accuracy = []
        self.val_losses = []
        self.val_accuracy = []

    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracy = []
        
    def on_test_begin(self, batch, logs=None):
        self.val_losses = []
        self.val_accuracy = []
        
    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get("accuracy"))
        
    def on_test_batch_end(self, batch, logs=None):    
        self.val_losses.append(logs.get('loss'))
        self.val_accuracy.append(logs.get("accuracy"))
        
#Through this object, it is later acces the information
cbk = LossHistory()
callbacks = [metric_callback,cbk]

#training step 3, epoc 3 with adam
history = model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)



#%%
'''Plots the results gather in the object cbk'''
steps = list(range(4686))
steps1 = list(range(1563))
plt.plot(steps, cbk.losses)
plt.plot(steps1, cbk.val_losses)

plt.title('model loss')

plt.ylabel('Loss')
plt.xlabel('Steps')

plt.legend(['train',"val"], loc='upper left')
plt.show()



#%%
'''Input dataset from stanford'''
dataset_sf_test = pd.read_csv(".../test_dataset.csv")
dataset_sf_train = pd.read_csv(".../train_dataset.csv")

#%%
'''convert it to datasetDict'''
train_dataset_sf = datasets.Dataset.from_dict(dataset_sf_train)
test_dataset_sf = datasets.Dataset.from_dict(dataset_sf_test)
my_dataset_dict = datasets.DatasetDict({"train":train_dataset_sf,"test":test_dataset_sf})

#%%
'''tokenization of data'''
tokenized_sf_dataset= my_dataset_dict.map(preprocess_function, batched=True)

#%%
'''Convert datasets to the tf.data.Dataset format for the model'''
tf_train_set_sf = model.prepare_tf_dataset(
    tokenized_sf_dataset["train"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set_sf = model.prepare_tf_dataset(
    tokenized_sf_dataset["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

#%%
'''make predictions using the model and the stanford dataset'''
preds = model.predict(tf_validation_set_sf)["logits"]

#%%
'''convert these logits into the model class predictions
by using argmax to find the highest logit'''
class_preds = np.argmax(preds, axis=1)
print(preds.shape, class_preds.shape)

#%%
'''loads the metrics you would like to use and computes it'''
metric = evaluate.load("glue","mrpc")
metric.compute(predictions=class_preds, references=my_dataset_dict["test"]["label"])

metric = evaluate.load("precision")
metric.compute(predictions=class_preds, references=my_dataset_dict["test"]["label"])




#%%
'''import dataset to be analyzed by the model'''
df_marvel = pd.read_csv(".../AvengersEndgame 2019.csv")
df_marvel = df_marvel.drop(["username", "rating", "helpful", "total", "date", "title"],axis=1)
df_marvel.head()

#%%
'''function to analyze each input and associate it with a tag'''
def provide_sentiment_score(data_frame_string):
    input_data = str(data_frame_string)
    encoded_text = tokenizer(input_data, return_tensors='np',truncation=True)
    output = model(**encoded_text)
    scores_string = softmax(output[0].numpy())
    
    if scores_string[0][0] > scores_string[0][1]:
        return 0
    else:
        return 1

#%%
'''applies the function and tags each input'''
df_marvel['Sentiment'] = df_marvel['review'].apply(provide_sentiment_score)
df_marvel.head()