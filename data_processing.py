import os
import csv

'''It takes the Stanford Large Movie Review Dataset and converts into two datasets for training and testing in csv format'''

'''The locations is changed according to the dataset set, it is shown below the root for training'''
directory = '.../aclImdb_v1/aclImdb/train/neg'
test_dict = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        temp_dict = {}
        bla = open(f,"r",encoding='utf-8',errors='ignore')
        temp_dict["text"] = str(bla.read())
        temp_dict["label"] = "0"
        test_dict.append(temp_dict)

'''The locations is changed according to the dataset set, it is shown below the root for training'''
directory = '.../aclImdb_v1/aclImdb/train/pos'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        temp_dict = {}
        bla = open(f,"r",encoding='utf-8',errors='ignore')
        temp_dict["text"] = str(bla.read())
        temp_dict["label"] = "1"
        test_dict.append(temp_dict)

field_names = ["text","label"]

'''creates the file in .csv format'''
with open('train_dataset.csv', 'w',encoding='utf-8',newline= '') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames = field_names) 
    writer.writeheader() 
    writer.writerows(test_dict) 
    