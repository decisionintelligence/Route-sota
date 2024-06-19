import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, gc
import threading
import multiprocessing 

from multiprocessing import Process
import gzip
from v_opt import VOpt
import json
import operator
import networkx as nx
import pickle
from tqdm import tqdm

class Overlap():
    def __init__(self, filename, subpath, p_path, subpath_range, overlap_name, path_count_name, path_desty_name):
        self.filename = filename
        self.subpath = subpath
        self.p_path = p_path
        self.subpath_range = subpath_range
        self.overlap_name = overlap_name
        self.path_count_name = path_count_name
        self.path_desty_name = path_desty_name
    
    def cal_overlap(self, path_keys):
        path_keys_ = [set([int(c) for c in key.split('-')]) for key in path_keys]
        N = len(path_keys_)
        print(N)
        overlap = {}
        for i in range(N-1):
            if i% 10000 == 0:
                print(i)
            for j in range(i+1, N):
                intes = path_keys_[i].intersection( path_keys_[j])
                if len(intes) > 1 and len(intes) < min(len(path_keys_[i]), len(path_keys_[j])):
                    if i not in overlap:
                        overlap[i] = []
                    if j not in overlap:
                        overlap[j] = []
                    overlap[i].append(j)
                    overlap[j].append(i)
        return overlap

    def norms(self, dicts2, is_norm):
        for e_key in dicts2:
            sums = sum(dicts2[e_key].values())
            for time in dicts2[e_key]:
                dicts2[e_key][time] = round(100.0*dicts2[e_key][time]/sums, 6)
            if is_norm:
                dicts2[e_key] =  dict(sorted(dicts2[e_key].items(), key=operator.itemgetter(1), reverse=True))
        return dicts2

    def get_vopt(self, vopt, dicts, is_path=False, is_norm=False):
        for e_key in tqdm(dicts):
            time_freq = dicts[e_key] 
            time_cost, time_freq = [float(l) for l in time_freq.keys()], [float(l) for l in time_freq.values()]
            pvalues = {}
            if len(time_cost) > 1:
                pvalues = vopt.v_opt(time_cost, time_freq, 3) # pvalues is a dict, contains new key and value
                dicts[e_key] = pvalues
            else:
                if not is_path:
                    k1 = str(int(np.mean(time_cost)))
                    pvalues[k1] = 1.0
        return self.norms(dicts, is_norm)

    def get_inters2(self, path_1, path_2, flag=False):
        end = path_1.find(';')
        lhead = path_1[:end]
        end2 = path_2.find(lhead)
        if end2 == -1:
            if flag: return 0
            return self.get_inters2(path_2, path_1, True)
        else:
            b = min(len(path_2)-end2, len(path_1))
            a = path_2[end2:] == path_1[:b]
            if not a: return 0
            else: return len(path_2[end2:].split(';'))

    def special2(self, path_keys, path_count,way_type={}):
        rev_dict = {}
        N = len(path_keys)
        print(N)
        M = 200   
        for i in range(N):            
            key = path_keys[i]
            key = set(key.split(';'))
            for k_ in key:
                if city=='xa' and k_ not in way_type:
                    continue
                if city=='xa' and way_type[k_]==0:
                    continue
                if k_ not in rev_dict:
                    rev_dict[k_] = []
                rev_dict[k_].append(i)  
        overlap, path_count_ = {}, {}
        for i in range(N):
            overlap[i] = [0] * M
        gc.disable()
        for i in range(N):     
            if i%10000 == 0: print(i)
            key = path_keys[i]
            key2 = set(key.split(';'))
            A = set()
            for k_ in key2:        
                if city=='xa' and k_ not in way_type:
                    continue
                if city=='xa' and way_type[k_]==0:
                    continue
                valu = rev_dict[k_]
                for va in valu: # valu is a int list
                    if va not in A:
                        A.add(va)
            for va in A:   
                if va != i:
                    pat_ = path_count[va]
                    path_len = len(pat_.split(';'))
                    intes = self.get_inters2(key, pat_)  
                    if intes > 0 and intes<min(len(key2), path_len):
                        overlap[i][0] += 1   
                        if overlap[i][0] >= M and overlap[i][0]%M == 0:
                            overlap[i].extend([0]*M)
                        overlap[i][overlap[i][0]] = va    
        gc.enable()
        for k in overlap:
            overlap[k] = overlap[k][1:overlap[k][0]+1]
        overlap1={str(i):list(map(str,overlap[i])) for i in overlap}
        return overlap1

    def get_k(self, key):
        end, j = -1, 0
        key = key.split('-')
        key_ = key[0]
        for kk in key[1:]:
            if j % 2 == 0:
                key_ += '-'
            else:
                key_ += ';'
            key_ += kk
            j += 1
        return key_
    
    def get_subpath(self, ):
        path_desty, path_count = {}, {}
        iters = 0
        way_type={}
        if city=='xa':
            filename = '../data/xa/XIAN_R_type.txt'
            with open(filename) as fn:
                for line in fn:
                    line = line.strip().split('\t')
                    if line[1] in ['nan','unclassified', 'track','residential','service','living_street']:  #'nan',
                        way_type[line[0]] = 0
                    else:
                        way_type[line[0]] = 1
        for i in self.subpath_range:
            if not os.path.exists(self.subpath+self.p_path+str(i)+'.txt'):
                continue
            with open(self.subpath + self.p_path + str(i) + '.txt', 'rb') as f:
                content = f.read()
                a = gzip.decompress(content).decode()
                jsdata = json.loads(a)

                for key in jsdata.keys():
                    key_ = self.get_k(key)
                    skey = key_.split(';')
                    if len(skey) > len(set(skey)):  
                        continue
                    path_desty[key_] = jsdata[key]   
                    path_count[iters] = key_      
                    iters += 1

        overlap = self.special2(list(path_desty.keys()), path_count,way_type)
        return path_desty, path_count, overlap

    def write_json(self, js_dict, fname):

        with open(fname, 'w') as fw:
            json.dump(js_dict, fw, indent=4)
    def write_file(self, js_dict, fname):
        fn = open(fname, 'w')
        for key in js_dict:
            val=','.join(str(l) for l in js_dict[key])
            fn.write(str(key) +':' + val + '\n')
        fn.close()

    def write_gzip(self, js_dict, fname):
        js_dict_ = json.dumps(js_dict)  
        compressed_string = gzip.compress(js_dict_.encode())  
        with open(fname, 'wb') as fw:
            fw.write(compressed_string)


    def main(self, ):
        path_desty, path_count, overlap = self.get_subpath()
        vopt = VOpt()
        path_desty = self.get_vopt(vopt, path_desty, True, True)
        self.write_gzip(path_desty, self.subpath + "norm_desty.txt")
        print('begin to write file ...')
        os.makedirs(self.subpath, exist_ok=True)
        self.write_gzip(path_desty, self.subpath + self.path_desty_name)
        self.write_gzip(path_count, self.subpath+self.path_count_name)
        self.write_gzip(overlap, self.subpath+self.overlap_name)

        

if __name__ == '__main__':
    dinxs = [50]
    # dinxs = [15, 30, 50, 100]
    city='xa'  ##cd aal xa
    flag=1
    datasets=["peak"] #["peak","offpeak"]
    for dataset in datasets:
        for dinx in dinxs:
            if city=='aal':
                if flag==1:
                    filename = "../data/aal/trips_real_"+str(dinx)+"_"+dataset+'.csv'
                else:
                    filename = '../data/aal/trips_real1_"+str(dinx)+"_"+dataset+'.csv'
                subpath = '../data/'+dataset+'_res%d' % dinx+'_%d/' %flag
            elif city=='cd':
                filename = '../data/cd/trips_real_'+str(dinx)+'_'+dataset+'.csv'
                subpath = '../data/'+dataset+'_cd_res%d' % dinx+'_%d/' %flag
            else:
                subpath = '../data/'+dataset+'_xa_new_res%d' % dinx+'_%d/' %flag
                if flag==1:
                    filename = '../data/xa/days_trips_real1_'+str(dinx)+'_'+dataset+'.csv'
                else:
                    filename = '../data/xa/days_trips_real2_'+str(dinx)+'_'+dataset+'.csv'
            p_path = 'path_travel_time_'
            path_count_name = 'path_count%d.txt' % dinx
            path_desty_name = 'path_desty%d.txt'%dinx
            overlap_name = 'overlap%d.txt'%dinx
            subpath_range = [l for l in range(2, 12)]
            overlap = Overlap(filename, subpath, p_path, subpath_range, overlap_name, path_count_name, path_desty_name)
            overlap.main()
    print("finished")


