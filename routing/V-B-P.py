import traceback

import networkx as nx
import numpy as np
import os, sys
import json
import time
import operator
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.stats import entropy
from pqdict import PQDict
import pickle
import copy
import argparse
import gzip
from v_opt import VOpt
import math
from math import cos, asin, sqrt, pi

class Modified():

    def __init__(self, T, fpath_desty, fedge_desty, subpath, axes_file, speed_file, true_path, query_name, sigma):
        self.T = T
        self.sigma = sigma
        self.vopt = VOpt()
        self.B = 3
        self.fpath_desty = fpath_desty
        self.fedge_desty = fedge_desty
        self.subpath = subpath
        os.makedirs(subpath, exist_ok=True)
        self.axes_file = axes_file
        self.pairs_num = 90
        self.speed_file = speed_file
        self.true_path = true_path
        self.speed = 50
        self.query_name = query_name
        self.nodes={}
    
   
    def add_tpath(self,):
        os.makedirs(os.path.dirname(self.subpath + self.true_path), exist_ok=True)
        with open(self.subpath + self.true_path, 'rb') as f:
            content = f.read()
            a = gzip.decompress(content).decode()
            gt_path_ = json.loads(a)
        gt_path = {}
        for gt in gt_path_:
            a = gt.find('-')
            a1 = gt[:a]
            b = gt.rfind('-')
            b1 = gt[b+1:]
            gt_key = a1+'-'+b1
            nw_key = list(gt_path_[gt].keys())
            nw_value = list(gt_path_[gt].values())
            lens1 = len(gt.split(';'))
            if gt_key not in gt_path:
                gt_path[gt_key] = [str(gt), gt_path_[gt], lens1, nw_key[0], a1, b1]
            else:
                ex_path = gt_path[gt_key]
                mins  = min(len(ex_path[1]), len(gt_path_[gt]))
                ex_key = list(ex_path[1].keys())
                ex_value = list(ex_path[1].values())
                a, b = 0, 0
                for i in range(mins):
                    if float(ex_key[i]) < float(nw_key[i]):
                        a += 1
                    elif  float(ex_key[i]) > float(nw_key[i]): 
                        b += 1
                    else:
                        if float(ex_value[i]) > float(nw_value[i]):
                            a += 1
                        else:
                            b += 1
                if lens1 < gt_path[gt_key][2] : b += 1
                elif lens1 > gt_path[gt_key][2] : a += 1
                if a < b: gt_path[gt_key] = [str(gt), gt_path_[gt], lens1, nw_key[0], a1, b1]

        return gt_path,gt_path_


    def get_axes(self, ):
        os.makedirs(os.path.dirname(self.axes_file), exist_ok=True)
        fo = open(self.axes_file)
        points = {}
        for line in fo:
            line = line.split('\t')
            points[line[0]] = (float(line[2]), float(line[1]))
        return points

    def get_distance(self, points, point):
        (la1, lo1) = points[point[0]]
        (la2, lo2) = points[point[1]]
        return geodesic((la1, lo1), (la2, lo2)).kilometers


    def get_dict(self, ):
        path = self.subpath + self.fpath_desty
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path = self.subpath+self.fedge_desty
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.subpath + self.fpath_desty, 'rb') as f:
            content = f.read()
            a = gzip.decompress(content).decode()
            path_desty = json.loads(a)

        with open(self.subpath + self.fedge_desty, 'rb') as f:
            content = f.read()
            a = gzip.decompress(content).decode()
            edge_desty = json.loads(a)
            edge_desty = dict(sorted(edge_desty.items(), key=operator.itemgetter(0)))

        vedge_desty = {}
        for p_k in path_desty:
            vedge_desty[p_k] = path_desty[p_k][0][1]
        return vedge_desty, edge_desty, path_desty

    def get_speed(self, ):
        speed_dict = {}
        os.makedirs(os.path.dirname(self.speed_file), exist_ok=True)
        with open(self.speed_file) as fn:
            for line in fn:
                line = line.strip().split('\t')
                speed_dict[line[0]] = {int(float(line[1])/float(line[2])*3600): 1.0}
        return speed_dict


    def get_graph2(self, edge_desty, gt_path, vedge_desty, r_pairs, points):
        speed_dict = self.get_speed()
        all_nodes, all_edges = set(), set()
        for key in speed_dict:
            line = key.split('-')
            all_nodes.add(line[0])
            all_nodes.add(line[1])
            all_edges.add(key)
        all_nodes, all_edges = list(all_nodes), list(all_edges)
        All_edges = []
        for edge in all_edges:
            edge_ = edge.split('-')
            if edge in edge_desty:
                cost1 = edge_desty[edge].keys()
                cost = min(float(l) for l in cost1)
                All_edges.append((edge_[0], edge_[1], cost))
            elif edge in speed_dict:
                cost1 = speed_dict[edge].keys()
                cost = min(float(l) for l in cost1)
                All_edges.append((edge_[0], edge_[1], cost))
        G2 = nx.DiGraph()
        G2.add_nodes_from(all_nodes)
        G2.add_weighted_edges_from(All_edges)

        temp_edges, temp_nodes = set(), set()
        for pairs in r_pairs:
            for pair in pairs:
                stedge = pair[-2] + '-' + pair[-1]
                #print(stedge)
                temp_edges.add(stedge)
                temp_nodes.add(pair[-2])
                temp_nodes.add(pair[-1])

        for gt_ in gt_path:
            All_edges.append((gt_path[gt_][-2], gt_path[gt_][-1], abs(float(gt_path[gt_][3]))))
            All_edges.append((gt_path[gt_][-1], gt_path[gt_][-2], abs(float(gt_path[gt_][3]))))
            temp_edge = gt_path[gt_][-2] + '-' + gt_path[gt_][-1]
            temp_edge2 = gt_path[gt_][-1] + '-' + gt_path[gt_][-2]
            if temp_edge not in edge_desty:
                edge_desty[temp_edge] = {abs(float(gt_path[gt_][3])):1.0}
            if temp_edge2 not in edge_desty:
                edge_desty[temp_edge2] = {abs(float(gt_path[gt_][3])):1.0}

        all_edges = set(all_edges)
        vedge_desty_ = {}
        for edge in vedge_desty:
            edge_ = edge.split('-')
            edge_2 = edge_[1] + '-' + edge_[0]
            if edge_2 not in vedge_desty:
                vedge_desty_[edge_2] = vedge_desty[edge]
        vedge_desty.update(vedge_desty_)
        for edge in vedge_desty:
            if edge not in edge_desty and edge not in speed_dict:
                edge_ = edge.split('-')
                cost1 = vedge_desty[edge].keys()
                cost = min(float(l) for l in cost1)
                All_edges.append((edge_[0], edge_[1], cost))
                all_edges.add(edge)
        all_edges = list(all_edges)

        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        G.add_weighted_edges_from(All_edges)
        for edge in speed_dict:
            if edge not in edge_desty:
                edge_desty[edge] = speed_dict[edge]
        self.nodes=all_nodes
        return all_edges, all_nodes, G, G2, speed_dict

    def conv(self, A, B):
        D = {}
        for a_k in A:
            for b_k in B:
                n_k = float(a_k) + float(b_k)
                n_v = float(A[a_k]) * float(B[b_k])
                D[n_k] = n_v
        if len(D) <= 3:
            return D
        D = dict(sorted(D.items(), key=operator.itemgetter(0)))
        new_w = self.vopt.v_opt(list(D.keys()), list(D.values()), self.B) 
        new_w = dict(sorted(new_w.items(), key=operator.itemgetter(0)))
        return new_w

    def get_min(self, U, node):
        return np.argwhere(U[node] > 0)[0][0]

    def cosin_distance(self, vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product / ((normA * normB) ** 0.5)


    def get_modified_one_to_all3(self, G,  target):
        Gk = G.reverse()
        inf = float('inf')
        D = {target:0}
        Que = PQDict(D)
        P = {}
        nodes = Gk.nodes
        U = set(nodes)
        while U:
            #print('len U %d'%len(U))
            #print('len Q %d'%len(Que))
            if len(Que) == 0: break
            (v, d) = Que.popitem()
            D[v] = d
            U.remove(v)
            #if v == target: break
            neigh = list(Gk.successors(v))
            for u in neigh:
                if u in U:
                    d = D[v] + Gk[v][u]['weight']
                    if d < Que.get(u, inf):
                        Que[u] = d
                        P[u] = v

        return P


    def rout(self, start, desti, edge_desty, vedge_desty, nodes_order, U, G, G2, points, gt_path, gt_path_, pred):
        Q, P_hat = [], []
        neigh = list(G.successors(start))
        start_ax = points[start]
        desti_ax = points[desti]
        s_d_arr = [desti_ax[0] - start_ax[0], desti_ax[1] - start_ax[1]]
        all_expire2 = 0.0

        def get_weight(p_hat):
            w_p_hat = {}
            if p_hat in edge_desty:
                w_p_hat = edge_desty[p_hat]
            elif p_hat in gt_path:
                w_p_hat = gt_path[p_hat][1]
            elif p_hat in vedge_desty:
                w_p_hat = vedge_desty[p_hat]
            else:
                print('other %s' %p_hat)
            if len(w_p_hat) == 0:
                print('zero %s' %p_hat)
            return w_p_hat

        def get_maxP3(w_p_, vv, vi_getmin):
            p_max = 0.0 
            for pk in w_p_:
                w_pk = w_p_[pk]
                pk_ = int(int(pk) / self.sigma)
                if int(pk) % self.sigma == 0: pk_ += 1
                if (1.0*self.T - float(pk)) > vi_getmin:
                    p_max += float(w_pk)
            return p_max

        def get_vigetmin2(vv):
            start2 = time.time()
            v = vv 
            path_ = [v]
            while v != desti:
                if v not in pred:
                    expire_time = time.time()-start2
                    return -1,expire_time
                v = pred[v]#[2]
                path_.append(v)
            paths_ = []
            if len(path_) == 2: paths_ = [path_[0]+'-'+path_[1]]
            else:
                paths_ = [path_[i-1]+'-'+path_[i] for i in range(1, len(path_))]
            spath = path_[0]
            vi_getmin = 0.0
            flag, iters = False, -1
            for l in range(len(paths_)):
                key = ';'.join(p for p in paths_[l:])
                if key in gt_path_:
                    vi_getmin += min(abs(float(ll)) for ll in gt_path_[key].keys())
                    flag = True
                    iters = 0
                    break
            if not flag:
                for l in range(len(paths_), 1, -1):
                    key = ';'.join(p for p in paths_[:l+1])
                    if key in gt_path_:
                        vi_getmin += min(abs(float(ll)) for ll in gt_path_[key].keys())
                        flag = True
                        iters = 1
                        break
            if not flag: 
                s, l = 0, len(path_)
            else:
                if iters == 0:
                    s = 0
                    l = l
                elif iters == 1:
                    s = l+1
                    l = len(paths_)
                #print('got shot')
            for key in paths_[s:l]:
                #key = spath+'-'+epath
                if key in edge_desty:
                    vi_getmin += min(abs(float(ll)) for ll in edge_desty[key].keys())
                elif key in vedge_desty:
                    vi_getmin += min(abs(float(ll)) for ll in vedge_desty[key].keys())
                else:
                    vi_getmin += min(abs(float(ll)) for ll in gt_path[key][1].keys())
            expire_time = time.time()-start2
            return vi_getmin, expire_time

        has_visit = set()
        has_visit.add(start)
        Que = PQDict.maxpq()
        Q = {}
        for vi in neigh:
            if vi in has_visit: continue
            else: has_visit.add(vi)
            p_hat = start +'-'+ vi
            w_p_hat = get_weight(p_hat)
            w_min = min([float(l) for l in w_p_hat.keys()])
            p_order = nodes_order[vi] #p_hat]
            vi_getmin, ex_p = get_vigetmin2(vi)
            if vi_getmin==-1:
                break
            all_expire2 += ex_p
            cost_time = w_min + vi_getmin
            if cost_time <= self.T:
                p_max = get_maxP3(w_p_hat, vi, vi_getmin)   
                Que[p_hat] = p_max
                Q[p_hat] = (p_max, w_p_hat, cost_time)
        QQ = {}
        p_best_p, flag = None, False
        p_best_p, p_max_m, p_best_cost, p_w_p  = None, -1, -1, -1 
        if len(Q) == 0: return None, -1, -1, -1, all_expire2, -1
        all_rounds = 0
        while len(Q) != 0:
            (p_hat, pqv) = Que.popitem()
            all_rounds += 1
            (p_max, w_p_hat, cost_time) = Q[p_hat]
            del Q[p_hat]
            a = p_hat.rfind('-')
            v_l = p_hat[a+1:]
            if v_l == desti:
                p_best_p = p_hat
                p_max_m = p_max
                p_best_cost = cost_time
                p_w_p = w_p_hat
                flag = True
                break
            neigh = list(G.successors(v_l))
            if len(w_p_hat.keys()) == 0:
                print(w_p_hat)
                print(p_hat)
            cost_sv = min([float(l) for l in w_p_hat.keys()])
            vd_d_arr = [points[desti][0]-points[v_l][0], points[desti][1]-points[v_l][1]]
            for u in neigh:
                if u == desti:
                    vu = v_l + '-' + u
                    w_vu = get_weight(vu)
                    if len(w_vu) == 0: cost_vu = 0
                    else: cost_vu = min([float(l) for l in w_vu.keys()])
                    vi_getmin, ex_p = get_vigetmin2(u)
                    all_expire2 += ex_p
                    p_best_p = p_hat + ';' + vu
                    p_w_p = self.conv(w_p_hat, w_vu)
                    p_max_m = get_maxP3(w_p_hat, u, vi_getmin)
                    p_best_cost = cost_sv + cost_vu + vi_getmin
                    flag = True
                    break
                if u in has_visit: 
                    #print('u1 %s'%u)
                    continue
                else: has_visit.add(u)
                if u in p_hat: 
                    #print('u2 %s'%u)
                    continue
                vu = v_l + '-' + u
                w_vu = get_weight(vu)
                if len(w_vu) == 0:
                    #print('vu %s'%vu)
                    continue
                cost_vu = min([float(l) for l in w_vu.keys()])
                vi_getmin, ex_p = get_vigetmin2(u)
                all_expire2 += ex_p
                cost_time = cost_sv + cost_vu + vi_getmin
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    p_hat_max = get_maxP3(w_p_hat, u, vi_getmin)
                    QQ[p_hat_p] = (p_hat_max, w_p_hat_p, cost_time)
            if flag: break
            if len(Q) == 0:
                Q = copy.deepcopy(QQ)
                for qqk in QQ:
                    Que[qqk] = QQ[qqk][0]
                QQ = {}
        return p_best_p, p_max_m, p_best_cost, p_w_p, all_expire2, all_rounds

    def get_dijkstra3(self, G,  target):
        Gk = G.reverse()
        inf = float('inf')
        D = {target:0}
        Que = PQDict(D)
        P = {}
        nodes = Gk.nodes
        U = set(nodes)
        while U:
            #print('len U %d'%len(U))
            #print('len Q %d'%len(Que))
            if len(Que) == 0: break
            (v, d) = Que.popitem()
            D[v] = d
            U.remove(v)
            neigh = list(Gk.successors(v))
            for u in neigh:
                if u in U:
                    d = D[v] + Gk[v][u]['weight']
                    if d < Que.get(u, inf):
                        Que[u] = d
                        P[u] = v
        return P


    def main(self, ):
        vedge_desty, edge_desty, path_desty = self.get_dict()
        gt_path, gt_path_ = self.add_tpath()
        os.makedirs(os.path.dirname(self.query_name), exist_ok=True)
        df2 = open(self.query_name, 'rb')
        r_pairs = pickle.load(df2)
        df2.close()
        points = self.get_axes()

        edges, nodes, G, G2, speed_dict = self.get_graph2(edge_desty, gt_path, vedge_desty, r_pairs, points)
        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1
        plot_data1, plot_kl, plot_lcs = [], [], []
        all_mps, all_mps2 = [], []
        ls = 0
        PT = 5
        plot_data1, plot_data2,  sums = [0]*PT, [0]*PT, [0]*PT
        one_plot, one_plot2 = [], []
        All_rounds = np.zeros(20).reshape(4, 5)
        One_Plot  = np.zeros(20).reshape(4, 5)
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        cate = ['0-5km', '5-10km', '10-25km', '25-35km']
        for pairs in r_pairs:
            one_dis += 1
            print('distance category %s'%cate[one_dis])
            tstart2 = time.time()
            #print('len pairs %d'%len(pairs))
            sums2  = 0
            all_expires = 0.0
            mps, mps2 = 0.0, 0.0
            cost_t, cost_t2 = 0, 0
            
            for pair_ in pairs:
                #print(pair_)
                print('o-d pair: %s'%pair_[0]+'-'+pair_[1])
                start, desti = pair_[-2], pair_[-1]
                pred2 = self.get_modified_one_to_all3(G, desti)
                path_2 , st1 = [start], start

                U = ''
                pred = self.get_dijkstra3(G, desti)
                path_, st1 = [start], start
                distan2 = 0
                while st1 != desti:
                    if st1 not in pred:
                        distan2=-1
                        break
                    st1 = pred[st1]
                    path_.append(st1)
                if distan2==-1:
                    continue
                st1, time_budget = start, 0.0
                for st2 in path_[1:]:
                    sedge = st1+'-'+st2
                    if sedge in edge_desty:
                        speed_key = list([float(l) for l in edge_desty[sedge].keys()])
                        time_budget += max(speed_key)
                    elif sedge in speed_dict:
                        speed_key = list([float(l) for l in speed_dict[sedge].keys()])
                        time_budget += max(speed_key)
                    elif sedge in vedge_desty:
                        speed_key = list([float(l) for l in vedge_desty[sedge].keys()])
                        time_budget += max(speed_key)
                    else: 
                        print(' edge: %s not in speed_dict, exit'%sedge)
                        sys.exit()
                        #continue
                    st1 = st2 
                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    tstart = time.time()
                    self.T = time_budget * t_b
                    best_p, max_m, best_c, best_pw, all_expire, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, nodes_order,U,G, G2, points, gt_path, gt_path_, pred2)
                    if all_expire < 0: continue
                    if best_p == None: continue
                    all_expires += all_expire

                    tend = time.time()
                    plot_data1[t_b_] += tend - tstart
                    plot_data2[t_b_] += tend - tstart - all_expire
                    sums[t_b_] += 1

                    One_Plot[one_dis][t_b_] += tend - tstart 
                    One_Plot2[one_dis][t_b_] += tend - tstart - all_expire
                    One_Sums[one_dis][t_b_] += 1
                    All_rounds[one_dis][t_b_] += all_rounds

                    if t_b_ == 2:
                        cost_t += tend - tstart 
                        cost_t2 += tend - tstart - all_expire
                        sums2 += 1
            print(One_Plot)
            print(One_Plot2 )
            print(One_Sums)
        for i in range(PT):
            if sums[i] == 0:
                print('zero %d'%i)
                continue
            plot_data1[i] /= sums[i]
            plot_data2[i] /= sums[i]
        for i in range(len(One_Plot2)):
            for j in range(len(One_Plot2[0])):
                if One_Sums[i][j]!=0:
                    One_Plot2[i][j] = One_Plot2[i][j] / One_Sums[i][j]
                    One_Plot[i][j] = One_Plot[i][j] / One_Sums[i][j]
        One_Plot = np.nan_to_num(One_Plot)
        One_Plot2 = np.nan_to_num(One_Plot2)
        print('The success account')
        print(One_Sums)
        print('The time cost for routing')
        print(One_Plot2)
        print('Time cost for budget: 50%, 75%, 100%, 125%, 150%')
        print(One_Plot2.mean(0))
        print('Time cost for distance: 0-5km, 5-10km, 10-25km, 25-35km')
        print(One_Plot2.mean(1))

        with(open(subpath+"V-B-P-result.txt", 'a+')) as fw:
            fw.write(str(sigma)+'\n')
            fw.write('The success account\n')
            fw.write(";".join(map(str,list(One_Sums)))+'\n')
            fw.write('The time cost for routing\n')
            fw.write(";".join(map(str,list(One_Plot2)))+'\n')
            fw.write('Time cost for budget: 50%, 75%, 100%, 125%, 150%\n')
            fw.write(";".join(map(str,list(One_Plot2.mean(0))))+'\n')
            fw.write('Time cost for distance: 0-5km, 5-10km, 10-25km, 25-35km\n')
            fw.write(";".join(map(str,list(One_Plot2.mean(1))))+'\n')

   
if __name__ == '__main__':
    try:

        parser = argparse.ArgumentParser(description='T-BS')
        parser.add_argument('--sig', default=2, type=int)
        args = parser.parse_args()
        if args.sig == 0:
            sigma, eta = 10, 800
        elif args.sig == 1:
            sigma, eta = 30, 333
        elif args.sig == 2:
            sigma, eta = 60, 170
        elif args.sig == 3:
            sigma, eta = 90, 111
        else:
            print('wrong sig , exit')
            sys.exit()
        
        threads_num = 10
        dinx = 50
        city='xa'
        dataset='peak'
        flag=1
        if city=='aal':
            filename = "../data/aal/trips_real_"+str(dinx)+"_"+dataset+'.csv'
            subpath = '../data/'+dataset+'_res%d' % dinx+'_%d/' %flag
            speed_file = '../data/AAL_NGR'
            axes_file = '../data/aal_vertices.txt'
            query_name = "../data/queries.txt"
        elif city=='cd':
            filename = '../data/cd/trips_real_'+str(dinx)+'_'+dataset+'.csv'
            subpath = '../data/'+dataset+'_cd_res%d' % dinx+'_%d/' %flag
            speed_file = '../data/full_NGR'
            axes_file = '../data/cd_vertices.txt'
            query_name = "../data/cd_queries.txt"
        else:
            filename = '../data/xa/new_days_trips_real_'+str(dinx)+'_'+dataset+'.csv'
            subpath = '../data/'+dataset+'_xa_res%d' % dinx+'_%d/' %flag
            speed_file = '../data/xa/XIAN_R_new.txt'
            axes_file = '../data/xa/xa_vertices.txt'
            query_name = "../data/xa/xa_new_queries.txt"

        true_path = 'path_desty%d.txt' % dinx
        fpath_desty = 'KKdesty_num_%d.txt' % threads_num  # 'new_path_desty1.json'
        fedge_desty = 'M_edge_desty.txt'


        time_budget = 5000
        rout = Modified(time_budget, fpath_desty, fedge_desty, subpath, axes_file, speed_file, true_path, query_name, sigma)
        rout.main()
        print("finished")
    except Exception as e:
        print(e)
        trace = sys.exc_info()
        traceback.print_exception(*trace)
        del trace

