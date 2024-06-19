import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, time
import threading
import multiprocessing 
from multiprocessing import Process, Manager
from geopy.distance import geodesic
import gzip
import json
import operator
import networkx as nx
from tqdm import tqdm
import pickle
from get_tpath import TPath
from v_opt import VOpt
from math import cos, asin, sqrt, pi
import random


# # haversin
def distance(lat1, lon1, lat2, lon2):
    r = 6731  
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 2 * r * asin(sqrt(a))


# def distance(lat1, lon1, lat2, lon2):  
    # return np.linalg.norm(np.array([lat1, lon1]) - np.array([lat2, lon2]))


class GenVP():
    def __init__(self, filename, subpath, p_path, subpath_range, fpath_desty, fpath_count, foverlap, B, store_desty_fname, threads_num, store_degree_fname, store_degree_fname2, graph_path, graph_store_fname,axes_file , dinx):
        self.filename = filename
        self.subpath = subpath
        os.makedirs(subpath, exist_ok=True)
        self.p_path = p_path
        self.subpath_range = subpath_range
        self.fpath_desty = fpath_desty
        self.fpath_count = fpath_count
        self.foverlap = foverlap
        self.B = B
        self.mid_olp = {}
        self.store_desty_fname = store_desty_fname
        self.threads_num = threads_num
        self.store_degree_fname = store_degree_fname 
        self.store_degree_fname2 = store_degree_fname2
        self.graph_path = graph_path
        self.graph_store_fname = graph_store_fname
        self.axes_file = axes_file
        self.dinx = dinx
        self.mergeres = {"desty": 0, "freq": 0, "err":0}
        self.edges = set()

    def get_speed(self, ):
        speed_dict = {}
        with open(self.graph_path) as fn:
            for line in fn:
                line = line.strip().split('\t')
                speed_dict[line[0]] = {int(3600*float(line[1])/float(line[2])): 1.0}  
        return speed_dict

    def load(self, ):
        data = pd.read_csv(self.filename)
        return data

    def get_axes(self, ):
        fo = open(self.axes_file)
        points = {}
        for line in fo:
            line = line.split('\t')
            points[line[0]] = (float(line[2]), float(line[1]))
        return points

    def get_distance(self, points, point):
        (la1, lo1) = points[point[0]]
        (la2, lo2) = points[point[1]]

        return distance(la1, lo1, la2, lo2)

    def get_distance2(self, points, path_1, path_2):
        a = path_1.find('-')
        point1 = path_1[:a]
        a = path_1.rfind('-')
        point2 = path_1[a+1:].split(";")[0]
        (la1, lo1) = points[point1]
        (la2, lo2) = points[point2]
        d1=distance(la1, lo1, la2, lo2)
        a = path_2.find('-')
        point3 = path_2[:a]
        a = path_2.rfind('-')
        point4 = path_2[a+1:].split(";")[0]
        (la3, lo3) = points[point3]
        (la4, lo4) = points[point4]
        d2=distance(la3, lo3, la4, lo4)
        d3 = distance(la1, lo1, la4, lo4)
        d4 = distance(la2, lo2, la3, lo3)
        dd = max(d3, d4)     
        if dd < d2 or dd < d1:
            return True
        return False

    def get_graph(self,):
        fn = open(self.graph_path)
        anodes, aedges = set(), set()
        # TODO out_degree
        for line in fn:
            line = line.split()
            ali = line[0].split('-')
            aedges.add((ali[0], ali[1]))
            anodes.add(ali[0])
            anodes.add(ali[1])
        return list(anodes), list(aedges)

    def get_overlap(self, ):
        with open(self.subpath + self.foverlap, 'rb') as f:
            content = f.read()
            a = gzip.decompress(content).decode()
            overlap = json.loads(a)
        return overlap

    def get_dict(self, ):
        
        with open(self.subpath + self.fpath_desty, 'rb') as f:
            content = f.read()
            a = gzip.decompress(content).decode()
            path_desty = json.loads(a)

        with open(self.subpath + self.fpath_count, 'rb') as f:
            content = f.read()
            a = gzip.decompress(content).decode()
            path_count = json.loads(a)

        TP = {}
        for path in path_count.values():
            a = path.find('-')
            s = path[:a]
            b = path.rfind('-')
            e = path[b+1:]
            tp = s+'-'+e
            TP[tp] = [path, path_desty[path]]

        overlap = self.get_overlap()
        i = 0
        path_num = {}
        self.nodes = set()
        for key in path_desty.keys():
            path_num[key] = i
            i += 1
            for ke in key.split('-'):
                self.nodes.add(ke)
        self.nodes = list(self.nodes)
        print('set node %d'%len(self.nodes))
        return path_desty, path_count, path_num, overlap, TP

    def check_loop(self, path_):
        path_1 = path_.split(';')
        path_2 = set(path_1)
        if len(path_1) > len(path_2):
            return True
        start = path_.find('-')
        head = path_[:start]
        end = path_.rfind('-')
        tail = path_[end+1:]
        if head in path_[start:] or tail in path_[:end]:
            return True
        return False

    def compose_path2(self, path_1, path_2, flag=False):
        end = path_1.find(';')
        lhead = path_1[:end]
        end2 = path_2.find(lhead)
        if end2 == -1:
            if flag: 
                return '', '', -1, False, ''
            return self.compose_path2(path_2, path_1, True)
        else:
            b = min(len(path_2)-end2, len(path_1))
            a = path_2[end2:] == path_1[:b]
            if len(path_2)-end2 == len(path_1): a = False
            psi = path_2 + ';' + path_1[b+1:]
            e1 = path_2.find('-')
            e2 = path_1.rfind('-')
            head, tail = path_2[:e1], path_1[e2+1:]  
            return psi, path_2[end2:], flag, a, head+'-'+tail

    def merge(self, p1, p2, f, s, vopt, path_desty, edge_time_freq,speed):

        def get_w(w1, w2, ws):
            wk1, wk2 = list(w1.keys()), list(w2.keys())
            N, J = len(wk1), len(wk2)
            new_w = {}
            if type(ws) == int or type(ws) == str:
                wks = 1
                for i in range(N):
                    for j in range(J):
                            n_k = str(int(wk1[i]) + int(wk2[j]) - int(ws))
                            if n_k in new_w:
                                new_w[n_k] += float(w1[wk1[i]]) * float(w2[wk2[j]]) / float(ws)
                            else:
                                new_w[n_k] = float(w1[wk1[i]]) * float(w2[wk2[j]]) / float(ws)
            else:
                wks = list(ws.keys())
                M = len(wks)
                for i in range(N):
                    for j in range(J):
                        for l in range(M):
                            n_k = str(int(wk1[i]) + int(wk2[j]) - int(float(wks[l])))
                            if n_k in new_w:
                                new_w[n_k] += float(w1[wk1[i]]) * float(w2[wk2[j]]) / float(ws[wks[l]])
                            else:
                                new_w[n_k] = float(w1[wk1[i]]) * float(w2[wk2[j]]) / float(ws[wks[l]])
            return new_w

        def get_ew(ww):
            for ew in ww:
                kew = self.edge_time_freq[ew]
                for kew_ in kew:
                    vew = kew[kew_]
                    
        
        w1 = path_desty[p1]
        w2 = path_desty[p2]
        if s in path_desty:
            ws = path_desty[s]
        else:
            if s in edge_time_freq:
                ws = edge_time_freq[s]    
            else:
                if s in speed:
                    ws=speed[s]
                else:
                    ws=1
        new_w = get_w(w1, w2, ws)
        if len(new_w) > 10:
            time_cost, time_freq = [float(l) for l in new_w.keys()], [float(l) for l in new_w.values()]
            new_w = vopt.v_opt(time_cost, time_freq, self.B) # p
        return new_w

    def check_domin(self, AA, BB):
        B1, B2 = list(map(int,BB.keys())), list(map(float,BB.values()))
        dom, B_len = 0, len(B1)
        for (_, Q) in AA:
            q_len = len(Q)
            M = min(B_len, q_len)
            A1, A2 = list(map(int,Q.keys())), list(map(float,Q.values()))
            a, b = 0, 0
            for m in range(M):
                if A1[m] > B1[m]:
                    a += 1
                elif A1[m] < B1[m]:
                    b += 1
                else:
                    if A2[m] > B2[m]:
                        a += 1
                    else:
                        b += 1
            if a < b: return False
        return True


    def get_tpaths(self, p_in):
        if p_in[0] == 'v':
            return p_in.split(';')[1:], 2
        else:
            return [p_in], 0

    def thread_fun(self, in_list_, vopt, path_desty, path_count, path_num, overlap, edge_time_freq, TP, All_Path, points):
        try:
            path_num_, over_ = {}, {}
            all_ =  0
            speed=self.get_speed()
            for p_in in in_list_:
                #print('p_in %s'%p_in)
                tp_in, spoint = self.get_tpaths(p_in)
                for t_p in tp_in:
                    dict_inx = -1
                    if t_p in overlap:
                        T_overlapping = overlap[t_p]   
                    else:
                        # print(t_p)
                        continue
                    if len(T_overlapping) < 1:
                        continue
                    for t_o in T_overlapping: # for it's overlapping paths
                        if t_o in tp_in:
                            continue
                        path_1 = path_count[t_o]
                        path_2 = path_count[p_in]
                        if self.get_distance2(points, path_1, path_2):
                            continue
                        psi, s, fg, is_s, TP_new = self.compose_path2(path_1, path_2)   

                        if psi == '' or not is_s: continue
                        if TP_new in TP: continue   
                        if self.check_loop(psi): continue
                        if len(psi.split(';'))>=150: continue    
                        if psi not in path_num and psi not in over_:
                            if not fg:
                                t_over = self.merge(path_1, path_2, psi, s, vopt, path_desty, edge_time_freq,speed)
                                new_ = 'v;'+t_o+';'+p_in[spoint:]
                            else:
                                t_over = self.merge(path_2, path_1, psi, s, vopt, path_desty, edge_time_freq,speed)
                                new_ = 'v;'+p_in[spoint:]+';'+t_o
                            if TP_new not in All_Path:    
                                All_Path[TP_new] = [(psi, t_over)]
                                over_[psi] = t_over
                                all_ += 1
                                path_num_[psi] = new_
                            else:                     
                                if self.check_domin(All_Path[TP_new], t_over):
                                    All_Path[TP_new].append((psi, t_over))
                                    over_[psi] =  t_over
                                    all_ += 1
                                    path_num_[psi] = new_
            print(f"Finished thread {os.getpid()}")
            return [path_num_, over_]  
        except Exception as e:
            print("There was an error at pid %d %d"%(os.getpid(), all_))
            print(e)
            trace = sys.exc_info()
            traceback.print_exception(*trace)
            del trace



    def collect_res(self, result):
        self.result.append(result)

    def gen(self, vopt, path_desty, path_count, path_num, overlap, edge_time_freq, TP, points):
        in_list = list(path_count.keys())
        '''multi processes 
        '''
        '''kkk = 0
        out_degree = self.mkgraph.G.degree(self.anodes)
        out_degrees = [self.anodes]
        out_degrees.append([l[1] for l in out_degree])
        out_degree = self.mkgraph.G.degree(self.nodes)
        out_degrees2 = [[l[0] for l in out_degree]]
        out_degrees2.append([l[1] for l in out_degree])
        print(len(out_degrees[1]))
        print(len(out_degrees2[1]))
        #print(out_degrees2)
        #sys.exit()'''
        All_Path = Manager().dict()
        linx = 0
        print(len(in_list))
        all_begin_t = time.time()
        while len(in_list) > 0: 
            self.result = []
            print('length of in list %d' %len(in_list))
            print('path desty %d'%(len(path_desty)))
            print('path count %d'%(len(path_count)))
            print('path num %d'%(len(path_num)))
            begin_t = time.time()
            len_in = len(in_list)
            random.shuffle(in_list)     
            if len_in % self.threads_num == 0:
                t_inxs = int(len_in/self.threads_num)
            else:
                t_inxs = int(len_in/self.threads_num) + 1
            pool = multiprocessing.Pool(self.threads_num)
            for len_thr in range(self.threads_num):
                mins = min((len_thr+1)*t_inxs, len_in)
                inxs_array = in_list[len_thr*t_inxs:mins]

                
                pool.apply_async(
                    self.thread_fun,
                    args=(inxs_array,vopt, path_desty, path_count, path_num, overlap, edge_time_freq, TP, All_Path, points),
                    callback=self.collect_res)
            pool.close()
            time.sleep(10)
            pool.join()
            time.sleep(10)

            path_num2, path_count2= {}, {}
            print(f"Got to Res: {len(self.result)}")

            for res in self.result:
                if res==None:
                    print("there is a none")
                    continue
                var1 = res[0]
                var2 = res[1]
                path_num2.update(var1)
                path_desty.update(var2)
            print('len path num2 %d' %len(path_num2))
            in_list = path_num2.values()
            new_edges = set()
            for ke in path_num2.keys():
                va = path_num2[ke]
                path_count2[va] = ke
                a = ke.find('-')
                b = ke.rfind('-')
                new_edges.add((ke[:a], ke[b+1:]))
            self.edges = set.union(self.edges, new_edges)
            path_count.update(path_count2)
            path_num.update(path_num2)
            del path_num2, path_count2
            #print(in_list)
            #print(in_list[0])
            print('len list %d'%len(in_list))
            in_list = list(set(in_list))
            print('len set list %d'%len(in_list))
            end_time=int(time.time()-begin_t)
            print('time cost %d'%end_time)
            print('all time cost %d'%int(time.time()-all_begin_t))
            print("end of while")
        return path_desty, All_Path 
    
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
            # print(time_cost)
            pvalues = {}
            if len(time_cost) > 1:
                pvalues = vopt.v_opt(time_cost, time_freq, self.B) # pvalues is a dict, contains new key and value
                dicts[e_key] = pvalues
            else:
                if not is_path:
                    k1 = str(int(np.mean(time_cost)))
                    pvalues[k1] = 1.0
        return self.norms(dicts, is_norm)

    def write_json(self, js_dict, fname):
        with open(fname, 'w') as fw:
            json.dump(js_dict, fw, indent=4)

    def write_gzip(self, js_dict, fname):
        js_dict_ = json.dumps(js_dict)
        compressed_string = gzip.compress(js_dict_.encode())
        with open(fname, 'wb') as fw:
            fw.write(compressed_string)

    def write_degree(self, out_degrees_, fname):
        temp = pd.DataFrame()
        temp['node'] = out_degrees_[0]
        strs = 0
        for out_degree in out_degrees_[1:]:
            temp[str(strs)] = out_degree
            strs += 1
        tem_str = temp.to_string()  
        compressed_string = gzip.compress(tem_str.encode())
        with open(fname, 'wb') as fw:
            fw.write(compressed_string)

    def store_graph(self, fname):
        strs = ''
        for edge_ in self.edges:
            strs += edge_[0]+'-'+edge_[1]+'\n'
        compressed_string = gzip.compress(strs.encode())
        with open(fname, 'wb') as fw:
            fw.write(compressed_string)

    def get_edge_time_freq(self, file):
        with open(file, 'rb') as fw:
            content = fw.read()
            a = gzip.decompress(content).decode()
            edge_time_freq = json.loads(a)
        
        return edge_time_freq

    def main(self, ):
        data = self.load()
        tpath = TPath(self.filename, self.subpath, self.dinx, self.threads_num)
        # edge_time_freq = tpath.get_edge_freq(data)
        edge_time_freq= self.get_edge_time_freq(subpath+'edge_travel_time.txt')
        anodes, aedges = self.get_graph()
        for edge in aedges:
            self.edges.add(str(edge[0] + '-' + str(edge[1])))
        self.anodes = anodes
        points = self.get_axes()
        path_desty, path_count, path_num, overlap, TP = self.get_dict()
        num=0
        for i in overlap:
            if len(overlap[i])>0:
                num+=1
        print('all edge %d'%len(overlap))
        print('all overlap edge %d'%num)
        vopt = VOpt()
        print('gen ...')
        #self.vopt = vopt
        path_desty, All_Path = self.gen(vopt, path_desty, path_count, path_num, overlap, edge_time_freq, TP, points)
        print('path desty length %d'%len(path_desty))
        self.store_graph(self.subpath+self.graph_store_fname)   
        All_Path_ = {}
        for ak in All_Path:
            All_Path_[ak] = All_Path[ak]
        self.write_gzip(All_Path_, self.subpath+self.store_desty_fname)
        count_num=[0]*6
        for i in All_Path_:
            for j in All_Path_[i]:
                length=len(j[0].split('-'))
                if length==2:
                    count_num[0]+=1
                elif length>2 and length<=10:
                    count_num[1]+=1
                elif length>10 and length<=25:
                    count_num[2]+=1
                elif length>25 and length<=50:
                    count_num[3]+=1
                elif length>50 and length<=100:
                    count_num[4]+=1
                else:
                    count_num[5]+=1
        with open(subpath+'v_path_num.txt', 'w+') as fw:
            fw.write("2\t2-10\t11-25\t25-50\t51-100\t>100\n")
            fw.write("\t".join(list(map(str,count_num))))
        print("Finished")

if __name__ == '__main__':
    dinxs = [50]
    process_num = 10
    flag=1
    datasets=["offpeak"] #["peak","offpeak"]
    city='xa'  ##cd aal xa
    # datasets=["offpeak"]
    for dataset in datasets:
        all_running_time=[]
        for dinx in dinxs:
            if city=='aal':
                filename = "../data/aal/trips_real_"+str(dinx)+"_"+dataset+'.csv'
                subpath = '../data/'+dataset+'_res%d' % dinx+'_%d/' %flag
                subpath = '../data/'+dataset+'_res%d' % dinx+'_%d/' %flag 
                graph_path = '../data/AAL_NGR'
                axes_file = '../data/aal_vertices.txt'
            elif city=='cd':
                filename = '../data/cd/trips_real_'+str(dinx)+'_'+dataset+'.csv'
                subpath = '../data/'+dataset+'_cd_res%d' % dinx+'_%d/' %flag
                graph_path = '../data/full_NGR'
                axes_file = '../data/cd_vertices.txt'
            else:
                subpath = '../data/'+dataset+'_xa_res%d' % dinx+'_%d/' %flag
                if flag==1:
                    filename = '../data/xa/new_days_trips_real_'+str(dinx)+'_'+dataset+'.csv'
                else:
                    filename = '../data/xa/new_days15_trips_real_'+str(dinx)+'_'+dataset+'.csv'
                graph_path = '../data/xa/XIAN_R_new.txt'
                axes_file = '../data/xa/xa_vertices.txt'
            p_path = 'path_travel_time_'
            fpath_count = 'path_count%d.txt' % dinx
            fpath_desty = 'path_desty%d.txt' % dinx
            foverlap = 'overlap%d.txt'% dinx
            # axes_file = '../data/vertices.txt'
            subpath_range = [l for l in range(2, 39)]
            B = 3
            threads_num = 10
            store_desty_fname = 'KKdesty_num_%d.txt' % threads_num    
            store_degree_fname = 'KKdegree_%d.txt' % threads_num
            store_degree_fname2 = 'KKdegree2_%d.txt' % threads_num
            graph_store_fname = 'KKgraph_%d.txt'%threads_num     
            begin_t = time.time()
            genvp = GenVP(filename, subpath, p_path, subpath_range, fpath_desty, fpath_count, foverlap, B, store_desty_fname, threads_num, store_degree_fname, store_degree_fname2, graph_path, graph_store_fname, axes_file, dinx)
            genvp.main()
            end_time=int(time.time()-begin_t)
            print('time cost %d'%end_time)
            all_running_time.append(end_time)
        with(open('../data/'+city+dataset+"_running_time.txt", 'w+')) as fw:
            all_time=""
            for i in range(len(all_running_time)):
                all_time+=str(dinxs[i])+": "+str(all_running_time[i])+"\n"
            fw.write(all_time)


