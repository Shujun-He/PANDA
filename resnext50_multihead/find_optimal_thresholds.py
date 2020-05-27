import numpy as np
import pickle
from sklearn import metrics
import itertools
from tqdm import tqdm

# from numba import jit
#
# @jit
# def qwk3(a1, a2, max_rat):
#     assert(len(a1) == len(a2))
#     a1 = np.asarray(a1, dtype=int)
#     a2 = np.asarray(a2, dtype=int)
#
#     hist1 = np.zeros((max_rat + 1, ))
#     hist2 = np.zeros((max_rat + 1, ))
#
#     o = 0
#     for k in range(a1.shape[0]):
#         i, j = a1[k], a2[k]
#         hist1[i] += 1
#         hist2[j] += 1
#         o +=  (i - j) * (i - j)
#
#     e = 0
#     for i in range(max_rat + 1):
#         for j in range(max_rat + 1):
#             e += hist1[i] * hist2[j] * (i - j) * (i - j)
#
#     e = e / a1.shape[0]
#
#     return 1 - o / e


def classification_threshold(raw,thresholds):
    tensor=raw.copy()
    # for i in range(len(tensor)):
    #     if tensor[i] > thresholds[4]:
    #         tensor[i]=5
    #     elif tensor[i] > thresholds[3]:
    #         tensor[i]=4
    #     elif tensor[i] > thresholds[2]:
    #         tensor[i]=3
    #     elif tensor[i] > thresholds[1]:
    #         tensor[i]=2
    #     elif tensor[i] > thresholds[0]:
    #         tensor[i]=1
    #     elif tensor[i] < thresholds[0]:
    #         tensor[i]=0
    # indices=(raw>thresholds[-1])
    # tensor[indices]=len(thresholds)+1
    # for i in reversed(range(len(thresholds)-1)):
    #     indices=(raw<=thresholds[i+1])&(raw>thresholds[i])
    #     tensor[indices]=i+1
    # indices=raw<thresholds[0]
    # tensor[indices]=0
    indices=raw<thresholds[0]
    tensor[indices]=0
    for i in range(len(thresholds)):
        indices=raw>thresholds[i]
        tensor[indices]=i+1


    return tensor.astype('int')


with open('pred_truth.p','rb') as f:
    raw,ground_truths=pickle.load(f)

products=itertools.product(range(1,10),repeat=5)
thresholds=[]
for product in products:
    thresholds.append(list(products))

thresholds=np.asarray(thresholds).squeeze()

best_score=0
#for a in tqdm(thresholds):
for a in thresholds:

    a=a*0.1+np.arange(5)

    classification=classification_threshold(raw,a)
    #classification=classification_threshold(raw)
    score=metrics.cohen_kappa_score(classification,ground_truths,weights='quadratic')
    #print(score)
    #score=qwk3(classification,ground_truths,5)
    if score > best_score:
        best_score=score
        best_thresholds=a
        print("New best thresholds: {}, score: {}".format(best_thresholds,score))
    #break

print("New best thresholds: {}, score: {}".format(best_thresholds,best_score))

with open('best_thresholds.txt','w+') as f:
    for tick in best_thresholds:
        f.write('{}\n'.format(tick))
