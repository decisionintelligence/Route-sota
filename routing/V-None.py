import traceback

import numpy as np
import networkx as nx
import numpy as np
import os, sys, copy
import json
import time
import operator
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pickle
from pqdict import PQDict
from v_opt import VOpt
import copy
import argparse
import gzip
class PaceRout():
    def __init__(self, fpath, T, fpath_desty, fedge_desty, subpath, axes_file, speed_file, true_path, sigma, eta, query_name):
        self.fpath = fpath
        self.hU = {}
        self.T = T
        self.sigma = sigma
        self.eta = eta
        self.vopt = VOpt()
        self.B = 3
        self.fpath_desty = fpath_desty
        self.fedge_desty = fedge_desty
        self.subpath = subpath
        self.axes_file = axes_file
        self.pairs_num = 90
        self.speed_file = speed_file
        self.true_path = true_path
        self.speed = 50
        self.query_name = query_name


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
        return geodesic((lo1, la1), (lo2, la2)).kilometers

    def get_dict(self, ):
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
        with open(self.speed_file) as fn:
            for line in fn:
                line = line.strip().split('\t')
                speed_dict[line[0]] = {3600*float(line[1])/float(line[2]): 1.0}
        return speed_dict


    def get_graph2(self, edge_desty, vedge_desty, r_pairs, points):
        speed_dict = self.get_speed()
        all_nodes, all_edges = set(), set()
        for key in speed_dict:
            line = key.split('-')
            all_nodes.add(line[0])
            all_nodes.add(line[1])
            all_edges.add(key)
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
        for edge in edge_desty:
            if edge not in speed_dict:
                print(edge)
        G2 = nx.DiGraph()
        G2.add_nodes_from(all_nodes)
        G2.add_weighted_edges_from(All_edges)

        temp_edges = set()
        for pairs in r_pairs:
            for pair in pairs:
                stedge = pair[-2] + '-' + pair[-1]
                #print(stedge)
                temp_edges.add(stedge)

        for edge in vedge_desty:
            if edge not in edge_desty and edge not in speed_dict:
                edge_ = edge.split('-')
                cost1 = vedge_desty[edge].keys()
                cost = min(float(l) for l in cost1)
                All_edges.append((edge_[0], edge_[1], cost))
                all_edges.add(edge)
        print('len of vedge_desty 2: %d'%len(vedge_desty))
        all_nodes, all_edges = list(all_nodes), list(all_edges)
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        G.add_weighted_edges_from(All_edges)
        return all_edges, all_nodes, G, G2, speed_dict

    def conv(self, A, B):
        D = {}
        for a_k in A:
            for b_k in B:
                n_k = int(float(a_k) + float(b_k))
                n_v = float(A[a_k]) * float(B[b_k])
                D[n_k] = n_v
        if len(D) <= 3:
            return D
        D = dict(sorted(D.items(), key=operator.itemgetter(0)))
        new_w = self.vopt.v_opt(list(D.keys()), list(D.values()), self.B)
        new_w = dict(sorted(new_w.items(), key=operator.itemgetter(0)))
        return new_w

    def path_conv(self, prob1, prob2, edge, edge_desty):
        DD = {}
        edge_w = edge_desty[edge]
        wk1, wk2, wk3 = list(prob1.keys()), list(prob2.keys()), list(edge_w.keys())
        N, J, M = len(prob1), len(prob2), len(edge_w)
        for n in range(N):
            for j in range(J):
                for m in range(M):
                    n_k = int(wk1[n]) + int(wk2[j]) - int(wk3[m])
                    if n_k < 0: continue
                    if n_k in DD:
                        DD[n_k] += float(prob1[wk1[n]]) * float(prob2[wk2[j]]) / float(edge_w[wk3[m]])
                    else:
                        DD[n_k] = float(prob1[wk1[n]]) * float(prob2[wk2[j]]) / float(edge_w[wk3[m]])
        if len(DD) <= 3: return DD
        #print(DD)
        DD = dict(sorted(DD.items(), key=operator.itemgetter(0)))
        DD = self.vopt.v_opt(list(DD.keys()), list(DD.values()), self.B)
        DD = dict(sorted(DD.items(), key=operator.itemgetter(0)))
        return DD

 
    def get_dijkstra(self, G, source, target):
        path = nx.dijkstra_path(G, source, target)
        return path

    def get_last_edge(self, apath):
        a = apath.rfind(';')
        if a == -1: return apath
        else: return apath[a+1:]

    def rout(self, start, desti, edge_desty, vedge_desty, path_desty, nodes_order, U, G, points, speed_dict):
        all_expire = 0.0 
        neigh = list(G.successors(start))
        has_visit = set()
        has_visit.add(start)
        Que = PQDict.maxpq()
        Q, QQ = {}, []
        print('neigh : %d'%len(neigh))
        for vi in neigh: 
            if vi in has_visit: continue
            else: has_visit.add(vi)
            candidite = start +'-'+ vi
            if candidite in edge_desty:
                w_can = edge_desty[candidite]
            elif candidite in speed_dict:
                w_can = speed_dict[candidite]
            elif candidite in vedge_desty:
                w_can = vedge_desty[candidite]
            w_min = min([float(l) for l in w_can.keys()])
            w_exp = 0.0
            for w_k, w_v in w_can.items():
                w_exp += float(w_k)*float(w_v)
            if w_min <= self.T:
                Que[candidite] = w_exp
                Q[candidite] = w_can
        Qkey = list(Q.keys())
        flag = False
        best_path, best_w = None, -1
        bts = time.time()
        all_rounds = 0
        while len(Q) != 0:
            (cand, pqv) = Que.popitem()
            all_rounds += 1
            w_can = Q[cand]
            del Q[cand]
            a = cand.rfind('-')
            v_l = cand[a+1:]
            if v_l == desti:
                best_path = cand
                best_w = w_can
                flag = True
                break
            if desti in cand:
                a = cand.find('-'+desti)
                best_path = cand[:a+len(desti)+1]
                best_w = w_can
                flag = True
                break
            neigh = list(G.successors(v_l))
            cost_sv = min([float(l) for l in w_can.keys()])
            for u in neigh:
                if u == desti:
                    next_cand = v_l + '-' + u
                    if next_cand in edge_desty:
                        next_w = edge_desty[next_cand]
                    elif next_cand in speed_dict:
                        next_w = speed_dict[next_cand]
                    elif next_cand in vedge_desty:
                        next_w = vedge_desty[next_cand]
                    tcost1 = time.time()
                    next_w_min = min([float(l) for l in next_w.keys()])
                    best_path = cand + ';'+ next_cand
                    best_w = self.conv(w_can, next_w)
                    all_expire += time.time() - tcost1
                    flag = True
                    break
                if u in has_visit: continue
                else: has_visit.add(u)
                if u in cand: continue
                next_cand = v_l + '-' + u
                if next_cand in vedge_desty:
                    next_w = vedge_desty[next_cand]
                elif next_cand in edge_desty:
                    next_w = edge_desty[next_cand]
                elif next_cand in speed_dict:
                    next_w = speed_dict[next_cand]

                next_w_min = min([float(l) for l in next_w.keys()])
                cost_time = cost_sv + next_w_min
                #if next_w_min <= self.T:
                if cost_time  <= self.T:
                    new_path = cand + ';' + next_cand
                    temp_1 = cand.rfind(';')
                    if temp_1 != -1:
                        new_p = cand[temp_1+1:]+';'+next_cand
                        if new_p in path_desty:
                            new_path_w = self.path_conv(w_can, path_desty[new_p], cand[temp_1+1:], edge_desty)
                        else:
                            new_path_w = self.conv(w_can, next_w)
                    else:
                        new_path_w = self.conv(w_can, next_w)

                    w_exp = 0.0
                    for w_k, w_v in w_can.items():
                        w_exp += float(w_k)*float(w_v)
                    QQ.append((new_path, new_path_w, w_exp))
            last_edge = self.get_last_edge(cand)
            if flag: break
            if True:
                for QQ_ in QQ:
                    Q[QQ_[0]] = QQ_[1]
                    Que[QQ_[0]] = QQ_[2] 
                QQ = []
        return best_path, best_w, all_rounds, all_expire

    def get_dijkstra3(self, G,  target):
        Gk = G.reverse()
        inf = float('inf')
        D = {target:0}
        Que = PQDict(D)
        P = {}
        nodes = Gk.nodes
        UU = set(nodes)
        while UU:
            if len(Que) == 0: break
            (v, d) = Que.popitem()
            D[v] = d
            UU.remove(v)
            neigh = list(Gk.successors(v))
            for u in neigh:
                if u in UU:
                    d = D[v] + Gk[v][u]['weight']
                    if d < Que.get(u, inf):
                        Que[u] = d
                        P[u] = v
        return P


    def get_U2(self, u_name):
        if u_name in self.hU:
            return self.hU[u_name]
        if not os.path.isfile(self.fpath+u_name):
            print(u_name)
            return {}
        with open(self.subpath + u_name, 'rb') as f:
            content = f.read()
            fn = gzip.decompress(content).decode()
        U = {}
        for line in fn:
            line = line.strip().split(';')
            U[line[0]] = np.zeros(self.eta)
            if line[1] == '-1' and line[2] == '-1':
                pass
            elif line[1] == '-1' and line[2] == '0':
                U[line[0]] = np.ones(self.eta)
            else:
                for i in range(int(line[1])+1, int(line[2])):
                    t = 3 + i - int(line[1]) - 1
                    U[line[0]][i] = float(line[t])
                for i in range(int(line[2]), self.eta):
                    U[line[0]][i] = 1.0
        self.hU[u_name] = U
        return U

    def main(self, ):
        df2 = open(self.query_name, 'rb')
        r_pairs = pickle.load(df2)
        df2.close()

        vedge_desty, edge_desty, path_desty = self.get_dict()
        points = self.get_axes()
        edges, nodes, G, G2, speed_dict = self.get_graph2(edge_desty, vedge_desty, r_pairs, points)

        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1

        plot_data1 = []
        U = ''
        all_mps, all_mps2 = [], []
        ls = 0
        PT = 5
        sums, plot_data1 = [0] * PT, [0] * PT
        one_plot = []
        All_rounds = np.zeros(20).reshape(4, 5)
        One_Plot = np.zeros(20).reshape(4, 5)
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        cate = ['0-5km', '5-10km', '10-25km', '25-35km']
        for pairs in r_pairs:
            one_dis += 1
            print('distance category %s'%cate[one_dis])
            tstart2 = time.time()
            sums2 = 0
            mps, mps2 = 0.0, 0.0
            alls = 0
            cost_t = 0
            for pair_ in pairs:
                print('o-d pair: %s'%pair_[0]+'-'+pair_[1])
                start, desti = pair_[-2], pair_[-1]
                if start == desti: 
                    print('wrong: start and desti is same')
                    print(pair_)
                    continue
                start, desti = pair_[-2], pair_[-1]
                pred = self.get_dijkstra3(G, desti)
                path_, st1 = [start], start
                while st1 != desti:
                    st1 = pred[st1]
                    path_.append(st1)
                st1, time_budget = start, 0.0
                for st2 in path_[1:]:
                    sedge = st1+'-'+st2
                    if sedge in edge_desty:
                        speed_key = list([abs(float(l)) for l in edge_desty[sedge].keys()])
                        time_budget += max(speed_key)
                    elif sedge in speed_dict:
                        speed_key = list([abs(float(l)) for l in speed_dict[sedge].keys()])
                        time_budget += max(speed_key)
                    elif sedge in vedge_desty:
                        speed_key = list([abs(float(l)) for l in vedge_desty[sedge].keys()])
                        time_budget += max(speed_key)
                    else: 
                        print(' edge: %s not in speed_dict, exit'%sedge)
                        sys.exit()
                    st1 = st2 

                tend = time.time()
                ls += 1
                print('budget %f'%time_budget)
                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    tstart = time.time()
                    self.T = time_budget * t_b

                    best_p, best_w, all_rounds, all_expire = self.rout(start, desti, edge_desty, vedge_desty, path_desty, nodes_order, U, G, points, speed_dict)
                    if best_p == None: continue
                    tend = time.time()
                    plot_data1[t_b_] += tend - tstart
                    sums[t_b_] += 1

                    One_Plot[one_dis][t_b_] += tend - tstart 
                    One_Plot2[one_dis][t_b_] += tend - tstart - all_expire
                    One_Sums[one_dis][t_b_] += 1
                    All_rounds[one_dis][t_b_] += all_rounds

                    if t_b_ == 2:
                        cost_t += tend - tstart
                        sums2 += 1
        for i in range(PT):
            if sums[i] == 0:
                continue
            plot_data1[i] /= sums[i]
        One_Plot = One_Plot / One_Sums 
        One_Plot2 = One_Plot2 / One_Sums 
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
        elif args.sig == 4:
            sigma, eta = 120, 80
        elif args.sig == 5:
            sigma, eta = 240, 33
        else:
            print('wrong sig , exit')
            sys.exit()

        dinx = 50
        threads_num = 10
        city='aal'
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
        fpath_desty = 'KKdesty_num_%d.txt' % threads_num  # 'new_path_desty1.json'
        fedge_desty = 'M_edge_desty.txt'
        fpath =  subpath + 'u_mul_matrix_sig%d/'%sigma
        true_path = 'path_desty%d.txt' % dinx

        time_budget = 5000
        pace_rout = PaceRout(fpath, time_budget, fpath_desty, fedge_desty, subpath, axes_file, speed_file, true_path, sigma, eta, query_name)
        pace_rout.main()
        print("finished")
    except Exception as e:
        print(e)
        trace = sys.exc_info()
        traceback.print_exception(*trace)
        del trace


