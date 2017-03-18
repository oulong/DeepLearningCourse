from collections import Counter
from strategy import SentimentNetwork
import numpy as np

with open('reviews.txt', 'r') as f:
    reviews = list(map(lambda x : x[0:-1], f.readlines()))

with open('labels.txt', 'r') as f:
    labels = list(map(lambda x : x[0:-1].upper(), f.readlines()))

'''
positive_count = Counter()
negative_count = Counter()
total_count = Counter()

for i in range(0, len(reviews)):

    if  labels[i] == 'POSITIVE':
        for word in reviews[i].split(' ') :
            positive_count[word] += 1
            total_count[word] += 1
    else:
        for word in reviews[i].split(' '):
            negative_count[word] += 1
            total_count[word] += 1


print(positive_count.most_common(30))
'''

'''
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
output_notebook()

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

for i in range(len(reviews)):
    if(labels[i] == 'POSITIVE'):
        for word in reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1

pos_neg_ratios = Counter()

for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio

for word,ratio in pos_neg_ratios.most_common():
    if(ratio > 1):
        pos_neg_ratios[word] = np.log(ratio)
    else:
        pos_neg_ratios[word] = -np.log((1 / (ratio+0.01)))

hist, edges = np.histogram(list(map(lambda x:x[1],pos_neg_ratios.most_common())), density=True, bins=100, normed=True)
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="Word Positive/Negative Affinity Distribution")
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="#555555")
show(p)
'''


#learning_rate 0.1 => 0.01 => 0.001
mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], polarity_cutoff=0.3, min_count=20, learning_rate=0.001)
#mlp.test(reviews[-1000:],labels[-1000:])
mlp.train(reviews[:-1000],labels[:-1000])
mlp.test(reviews[-1000:],labels[-1000:])


