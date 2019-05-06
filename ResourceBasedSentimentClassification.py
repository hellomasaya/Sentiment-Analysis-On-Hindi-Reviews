# This module is written to do a Resource Based Semantic analyasis using hindi sentiwordnet.
import pandas as pd
import codecs
from nltk.tokenize import word_tokenize
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import f1_score
import re
import array as arr
import numpy as np

data = pd.read_csv("HindiSentiWordnet.txt", delimiter=' ')

fields = ['POS_TAG', 'ID', 'POS', 'NEG', 'LIST_OF_WORDS']

#Creating a dictionary which contain a tuple for every word. Tuple contains a list of synonyms,
# positive score and negative score for that word.
words_dict = {}
for i in data.index:
    # print (data[fields[0]][i], data[fields[1]][i], data[fields[2]][i], data[fields[3]][i], data[fields[4]][i])

    words = data[fields[4]][i].split(',')
    for word in words:
        words_dict[word] = (data[fields[0]][i], data[fields[2]][i], data[fields[3]][i])

neg_list = []
negatives_file = codecs.open("negatives.txt", "r", encoding='utf-8', errors='ignore').read()
neg_list = negatives_file.split(',')
neg_array = np.asarray(neg_list)
a=0 #a=number of sentences found w/ desired negation

# This function checks negation
def negation_handling(text):
    shabd = word_tokenize(text)
    array1 = np.asarray(shabd)
    lent = len(array1)
    for x in range(lent):
        if array1[x] == neg_array[0] and (array1[x-1] in words_dict): # 'na' not being matched
            global a
            a+=1
            neg=-1
            return neg

# This function determines sentiment of text.
def sentiment(text):
    negation = negation_handling(text)
    words = word_tokenize(text)
    votes = []
    pos_polarity = 0
    neg_polarity = 0
    #adverbs, nouns, adjective, verb are only used
    allowed_words = ['a','v','r','n']
    for word in words:
        if word in words_dict:
            #if word in dictionary, it picks up the positive and negative score of the word
            pos_tag, pos, neg = words_dict[word]
            # print(word, pos_tag, pos, neg)
            if pos_tag in allowed_words:
                if pos > neg:
                    pos_polarity += pos
                    votes.append(1)
                elif neg > pos:
                    neg_polarity += neg
                    votes.append(0)
    #calculating the no. of positive and negative words in total in a review to give class labels
    pos_votes = votes.count(1)
    neg_votes = votes.count(0)
    if negation == None:
        negation = 1
    if pos_votes > neg_votes:
        return 1*negation
    elif neg_votes > pos_votes:
        return -1*negation
    else:
        if pos_polarity < neg_polarity:
            return -1*negation
        else:
            return 1*negation


pred_y = []
actual_y = []
# to calculate accuracy

pos_reviews = codecs.open("pos_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
for line in pos_reviews.split('$'):
    data = line.strip('\n')
    if data:
        pred_y.append(sentiment(data))
        actual_y.append(1)

neg_reviews = codecs.open("neg_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
for line in neg_reviews.split('$'):
    data=line.strip('\n')
    if data:
        pred_y.append(sentiment(data))
        actual_y.append(-1)

neg_actual=actual_y.count(-1)
neg_pred=pred_y.count(-1)
pos_actual=actual_y.count(1)
pos_pred=pred_y.count(1)

print("\n",'positive:', "actual:", pos_actual, '  ', "predicted:", pos_pred)
print('negative:', "actual:", neg_actual, '  ', "predicted:",neg_pred,"\n")

print("Accuracy%:",accuracy_score(actual_y, pred_y) * 100, "\n")
print('F-measure:  ',f1_score(actual_y,pred_y))

# if __name__ == '__main__':
    #print(sentiment("मैं इस उत्पाद से बहुत खुश हूँ  यह आराम दायक और सुन्दर है  यह खरीदने लायक है "))
