from os.path import abspath

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy
from os import path
import threading
import multiprocessing 
from multiprocessing import Process
import json
import operator
import gzip
class TPath():
    def __init__(self, filename, subpath, dinx, process_num):
        self.filename  = filename
        self.subpath = subpath
        os.makedirs(self.subpath, exist_ok=True)
        self.dinx = dinx
        self.process_num = process_num
        self.w_data = multiprocessing.Manager().dict()

    def load(self):
        try:
            data = pd.read_csv(self.filename)
            print("finished reading data...")
            return data
        except Exception as e:
            print(e)

    def get_edge_freq(self, data):
        try:
            seg_ids = data.seg_id.values
            N = len(seg_ids)
            edge_freq = {}
            for i in range(N):
                if seg_ids[i] not in edge_freq:
                    edge_freq[seg_ids[i]] = 1
                else:
                    edge_freq[seg_ids[i]] += 1
            # edge_freq_={}
            # for i in edge_freq:
            #     if edge_freq[i]>self.dinx:
            #         edge_freq_[i]=edge_freq[i]
            # 
            return edge_freq
        except Exception as e:
            print(e)

    def get_edge_time_freq(self,data):
        try:
            seg_ids = data.seg_id.values
            travel_time = data.travel_time.values
            N = len(seg_ids)
            edge_time_freq = {}
            for i in range(N):
                str_travel=str(travel_time[i])
                if seg_ids[i] not in edge_time_freq:
                    edge_time_freq[seg_ids[i]] = {str_travel: 1}
                else:
                    if str_travel not in edge_time_freq[seg_ids[i]]:
                        edge_time_freq[seg_ids[i]][str_travel] = 1
                    else:
                        edge_time_freq[seg_ids[i]][str_travel] += 1
            edge_time_freq_= {}
            for key in edge_time_freq.keys():
                for j in edge_time_freq[key]:
                    if edge_time_freq[key][j] > self.dinx:
                        if key not in edge_time_freq_:
                            edge_time_freq_[key]={}
                        edge_time_freq_[key][j]=edge_time_freq[key][j]
            # print(edge_time_freq_.keys())
            return edge_time_freq_
        except Exception as e:
            print(e)


    def full_(self, data):
        N = len(data)
        print('lens N: %d'%N)
        full_len = {} #full path's freq
        last_ntid, path_key = -1, ''
        path_len = {} #full path's length
        lens = 0
        path_time = {}
        one_time = []
        for data_ in data.values:
            #print(data_)
            ntid, path, inx = data_[1], data_[-3], data_[-2]
            t_time = data_[-1]
            if ntid != last_ntid:
                if path_key != '':
                    if path_key in full_len:
                        full_len[path_key] += 1
                    else:
                        full_len[path_key] = 1
                        path_len[path_key] = lens
                    if path_key in path_time:
                        path_time[path_key].append(one_time)
                    else:
                        path_time[path_key] = [one_time]

                path_key = path
                lens = 1
                one_time = [t_time]
            else:
                '''if type(path) != str:
                    path = str(path)
                if type(path_key) != str:
                    path_key = str(path_key)'''
                path_key = str(path_key) + '-' + str(path)
                lens += 1
                one_time.append(t_time)
            last_ntid = ntid
        if path_key in full_len:
            full_len[path_key] += 1
        else:
            full_len[path_key] = 1
            path_len[path_key] = lens
        if path_key in path_time:
            path_time[path_key].append(one_time)
        else:
            path_time[path_key] = [one_time]
        # return full_len, path_len, path_time

        full_len_, path_len_, path_time_={},{},{}
        for key in full_len.keys():
            if full_len[key] > self.dinx:
                full_len_[key],path_len_[key], path_time_[key]=full_len[key],path_len[key], path_time[key]
        print(len(full_len_))
        count_num=[0]*5
        for key in full_len_.keys():
            if path_len_[key]==1:
                count_num[0]+=full_len_[key]
                continue
            if path_len_[key]<=5:
                count_num[1]+=full_len_[key]
                continue
            if path_len_[key]<=10:
                count_num[2]+=full_len_[key]
                continue
            if path_len_[key]<=20:
                count_num[3]+=full_len_[key]
                continue
            count_num[4]+=full_len_[key]
        print(count_num)
        # return full_len_, path_len_, path_time_
        return full_len, path_len, path_time
    
    def cut_it(self, path_freq, path_len, path_time, k):
        new_dict = sorted(path_len.items(),  key=lambda d: d[1], reverse=False)
        keys = [i[0] for i in new_dict]
        # keys = path_len.keys()
        freq_dict = {}
        kk = 2 * k -1

        def cut_it_(paths):
            paths = paths.split('-')
            pls = []
            lens1 = int((len(paths)+1)/2)
            p_lens = lens1 - k + 1
            for i in range(p_lens):
                skey = '-'.join(paths[i*2 + j] for j in range(2 * k))
                pls.append(skey)
            return pls 

               
        def is_full(f_index, inx, s_len):
            iset = set()
            for f_inx in f_index:
                if f_inx != inx:
                    for i in range(k):
                        iset.add(f_inx + i)
                        #print(f_inx+i)
            if len(iset) < s_len:
                return False
            elif len(iset) == s_len:
                return True
            else:
                print('error happend, please check code ...')
                print('error_1 %d %d %d %d' % (len(iset), s_len, inx, k))
                print(f_index)
                print(iset)
                return False

        for key in keys:
            if path_len[key] < k:
                continue
            elif path_len[key] == k:
                freq_dict[key] = path_freq[key]    
            else:
                path1 = cut_it_(key)
                for p1 in path1:
                    if p1 in freq_dict:
                        freq_dict[p1] += 1
                    else:
                        freq_dict[p1] = 1

        cut_path = {}
        path_time_freq = {}
        for key in keys:
            if path_len[key] < k:
                cut_path[key] = path_freq[key]
            elif path_len[key] == k:
                cut_path[key] = path_freq[key]  
                path_time_ = path_time[key]
                p_time = [sum(p_t) for p_t in path_time_]
                if key not in path_time_freq:
                    path_time_freq[key] = {}
                for p_t in p_time:
                    if p_t not in path_time_freq[key]:
                        path_time_freq[key][p_t] = 1
                    else:
                        path_time_freq[key][p_t] += 1
                #     for p_t in p_time:
                #         if p_t not in path_time_freq[key]:
                #             path_time_freq[key][p_t] = 1
                #         else:
                #             path_time_freq[key][p_t] += 1
                # else:
                #     for p_t in p_time:
                #         if p_t not in path_time_freq[key]:
                #             path_time_freq[key][p_t] = 1
                #         else:
                #             path_time_freq[key][p_t] += 1

            else:
                path1 = cut_it_(key)
                freqs = [freq_dict[p1] for p1 in path1]
                index = np.argsort(freqs) 
                f_index = list(index)
                s_len = len(index) + k - 1
                for f_inx in f_index:
                    k1 = path1[f_inx]
                    if k1 in cut_path:
                        cut_path[k1] += 1
                    else:
                        cut_path[k1] = 1
                path_time_ = path_time[key]
                for f_inx in f_index:
                    k1 = path1[f_inx]
                    p_time = [sum(p_t[f_inx:f_inx+k+1]) for p_t in path_time_]
                    if k1 not in path_time_freq:
                        path_time_freq[k1] = {}
                    for p_t in p_time:
                        if p_t not in path_time_freq[k1]:
                            path_time_freq[k1][p_t] = 1
                        else:
                            path_time_freq[k1][p_t] += 1
                    # else:
                    #     for p_t in p_time:
                    #         if p_t not in path_time_freq[k1]:
                    #             path_time_freq[k1][p_t] = 1
                    #         else:
                    #             path_time_freq[k1][p_t] += 1

        for k_1 in path_time_freq:
            path_time_freq[k_1] = dict(sorted(path_time_freq[k_1].items(), key=operator.itemgetter(1), reverse=True))
        flag = self.write2(cut_path, k)
        return flag, cut_path, path_time_freq

    def write(self, paths, k):
        B = [1000, 500, 200, 100, 50, 30, 20, 10, 5, 1, 1]
        A = [[0]*k for i in range(len(B))]
        for key in paths:
            lens = int((len(key.split('-')) + 1)/2)
            for i in range(len(B) -1):
                if paths[key] > B[i]:
                    A[i][lens-1] += 1
            if paths[key] == B[i]:
                A[-1][lens-1] += 1
        strs = '>,'
        strs += ','.join(str(i+1)+'-edge' for i in range(k)) + '\n'
        for i in range(len(B)-1):
            strs += '>%d,'%B[i]
            strs += ','.join(str(A[i][j]) for j in range(k)) + '\n' 
        strs += '=%d,'%B[-1]
        strs += ','.join(str(A[-1][j]) for j in range(k))
        fname = './full_cut/stat_10_k%d.csv'%k
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w') as files:
            files.write(strs)
            files.close()

    def write2(self, paths, k):
        B = [1000, 500, 200, 100, 50, 30, 20, 10, 5, 1, 1]
        A = [0] * len(B)
        for key in paths:
            lens = int((len(key.split('-')) + 1)/2)
            if lens == k:
                for i in range(len(B)-1):
                    if paths[key] > B[i]:
                        A[i] += 1
                        break
                if paths[key] == B[i]:
                    A[-1] += 1
        titles = '%d-edge'%k
        if sum(A[:5]) == 0:
            #fname = './full_cut/all_stat_10_k%d.csv'%k
            #self.w_data.to_csv(fname, index=None)
            return False
        else:
            self.w_data[titles] = A
            return True

    def write_json(self, js_dict, fname):
        #json.dumps(js_dict, fname )
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w') as fw:
            json.dump(js_dict, fw, indent=4)

    def write_json2(self, js_dict, cut_path, fname):
        #json.dumps(js_dict, fname )
        fre_n = self.dinx
        js_dict_ = copy.deepcopy(js_dict)
        for key in js_dict.keys():
            if cut_path[key] <= fre_n:
                del js_dict_[key]
        if len(js_dict_) < 1: return
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'w') as fw:
            json.dump(js_dict_, fw, indent=4)

    def write_gzip2(self, js_dict, cut_path, fname):
            # json.dumps(js_dict, fname )
        fre_n = self.dinx
        js_dict_ = copy.deepcopy(js_dict)
        for key in js_dict.keys():
            if cut_path[key] <= fre_n:
                del js_dict_[key]
        if len(js_dict_) < 1: return
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        js_dict_=json.dumps(js_dict_)
        compressed_string = gzip.compress(js_dict_.encode())
        with open(fname, 'wb') as fw:
            fw.write(compressed_string)


    def thread_fun(self, path_freq, path_len, path_time, inxs_array):
        for inx_k in inxs_array:
            path_time_freq={}
            values, cut_path, path_time_freq = self.cut_it(path_freq, path_len, path_time, inx_k)
            if not values:
                print(inx_k)
                #break
            else:
                # fname = self.subpath+'path_travel_time_%d.json'%inx_k
                # os.makedirs(os.path.dirname(fname), exist_ok=True)
                # self.write_json2(path_time_freq, cut_path, fname)
                fname = self.subpath + 'path_travel_time_%d.txt'%inx_k
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                self.write_gzip2(path_time_freq, cut_path, fname)
            #print('hehe')
        #return path_time_freq
        return path_time_freq

    def write_gzip(self, js_dict, fname):  ##npz
        js_dict_ = json.dumps(js_dict)
        compressed_string = gzip.compress(js_dict_.encode())
        with open(fname, 'wb') as fw:
            fw.write(compressed_string)


    def main(self, ):
        print('load data ...')
        data = self.load()
        print('edge freq ...')
        edge_freq = self.get_edge_freq(data)
        edge_time_freq=self.get_edge_time_freq(data)
        self.write_gzip(edge_time_freq, subpath+'edge_travel_time.txt')
        print('get path len and freq ...')
        path_freq, path_len, path_time = self.full_(data)#full_len is the frequency of a path; path_len is the length of a path.
        k = 0
        B = [1000, 500, 200, 100, 50, 30, 20, 10, 5, 1, 1]
        t1 = ['>%d'%B[l] for l in range(len(B)-1)]
        t1.append('=1')
        self.w_data = multiprocessing.Manager().dict()
        print('cut path ...')
        inxs = [l for l in range(2, 61)]
        # inxs = [l for l in range(1, 61)]
        threads_num = self.process_num
        t_inxs = int(len(inxs) / threads_num) +1
        thread_array = []
        for len_thr in range(threads_num):
            mins = min((len_thr+1)*t_inxs, len(inxs))
            inxs_array = inxs[len_thr * t_inxs : mins]
            print(inxs_array)
            threads_ = Process(target=self.thread_fun, args=(path_freq, path_len, path_time, inxs_array))
            thread_array.append(threads_)
            print('start thread %d' %len_thr)
            threads_.start()

        for len_thr in range(threads_num):
            thread_array[len_thr].join()
        print('write file ...')
        fname = self.subpath+'AAL_stat_%d.csv'%self.dinx
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        ww_data = {}
        kkeys = ['%d-edge'%kke for kke in range(2, 100)]
        for kk in kkeys:
            if kk in self.w_data:
                ww_data[kk] = self.w_data[kk]
        ww_data = pd.DataFrame.from_dict(ww_data)
        ww_data.insert(loc=0, column='>', value=t1)
        ww_data.to_csv(fname, sep=',', index=None)

        
if __name__ == '__main__':
    # dinx = 15
    dinxs = [15,30,50,100]
    # datasets=["offpeak"]
    datasets=["peak","offpeak"]
    process_num = 10
    for dataset in datasets:
        for dinx in dinxs:
            print("dataset:"+dataset+" dinx:"+str(dinx))
            # subpath = '../data/res%d/'%dinx
            # filename = '../data/Aalborg_2007.csv'
            # filename = '../data/AAL_short_50_peak.csv'
            # subpath = '../data/peak_res%d/' % dinx
            filename = '../data/cd/trips_real_'+str(dinx)+'_'+dataset+'.csv'
            subpath = '../data/'+dataset+'_cd_res%d/' % dinx
            tpath = TPath(filename, subpath, dinx, process_num)
            tpath.main()
    print("Finished")
