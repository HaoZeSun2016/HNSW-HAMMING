# coding: utf-8
import pickle as pkl
import time
import numpy as np
import hnswlib

threads = 25
M = 16
# M = 24
efC = 200
#efC = 1000
efS = int(0.8 * efC)
dim = 384
inp_dir = '/search/odin/data/lizhi-query/lizhi-query-onlytext-seqnash/'
#inp_dir = '/search/odin/data/lizhi-query/lizhi-query-hq-seqnash/'
inp_files = [inp_dir + 'part-' + '%05d' % x for x in range(10)]
print(inp_files)
n = 610000000  # changed by data
#n = 88000000  # changed by data
out_dir = '/search/odin/data/lizhi-query/results-onlytext/'
#out_dir = '/search/odin/data/lizhi-query/results-hq/'

# ------- txt output ----------
g = open(out_dir + 'text', 'wb')
g.close()
g = open(out_dir + 'text', 'a')

#-------- initialize index ------------------
index = hnswlib.Index(space='hamming', dim=dim)
index.set_num_threads(threads)
index.init_index(max_elements=n, ef_construction=efC, M=M)
print('begin circling ...')
# ------- add items n times ----------------
tol_l = 0
for cnt, inp_file in enumerate(inp_files):
    tt = time.time()
    f = open(inp_file, 'rb')
    lines = f.readlines()
    length = len(lines)
    tol_l += length
    f.close()
    embs = [l.strip().split('\t')[0].split(',') for l in lines]
    embs = np.asarray(embs, dtype=np.uint32)
    if cnt == 0:
        embs0 = embs
    text = '\n'.join(['\t'.join(l.strip().split('\t')[1:]) for l in lines])
    g.write(text + '\n') 
    print('deal data ' + inp_file + ' with time: ' + str(time.time() - tt))
    
    tt = time.time()
    index.add_items_uint32(embs)
    print('add index items ' + str(length) + ' with time: ' + str(time.time() - tt))

g.close()
index.save_index(out_dir + 'index')
print('save index with %d items' % tol_l)

index.set_num_threads(1)
n_query = 10000
tt = time.time()
labels, distances = index.knn_query_uint32(embs0[0:n_query, :], k=3)
print('checking ' + str(n_query)+ ' queries with time (1 thread): ' + str(time.time() - tt))
recall = 0.0
for i in range(n_query):
    if labels[i, 0] == i:
        recall += 1.0
print('Recall Top1:' + str(recall / n_query) )
