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
import pickle
from pqdict import PQDict
import copy
import argparse
import gzip
from v_opt import VOpt
import math
from math import cos, asin, sqrt, pi

class Rout():
    def __init__(self, fpath, T, fpath_desty, fedge_desty, subpath, axes_file,  speed_file, sigma, eta, query_name):
        self.fpath = fpath
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        self.hU = {}
        self.T = T
        self.sigma = sigma
        self.eta = eta
        self.vopt = VOpt()
        self.B = 3
        self.fpath_desty = fpath_desty
        self.fedge_desty = fedge_desty
        self.subpath = subpath
        os.makedirs(subpath, exist_ok=True)
        self.axes_file = axes_file
        self.pairs_num = 90
        self.speed_file = speed_file
        self.query_name = query_name
        self.nodes={}
    

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
        return self.distance(la1, lo1, la2, lo2)
    
    def distance(self,lat1, lon1, lat2, lon2):
        r = 6731  
        p = pi / 180
        a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        return 2 * r * asin(sqrt(a))

    def get_U2(self, u_name):
        nodes=self.nodes
        if u_name in self.hU:
            return self.hU[u_name]
        if not os.path.isfile(self.fpath+u_name):
            print(u_name)
            return {}
        with open(self.fpath + u_name, 'rb') as f:
            content = f.read()
            fn = gzip.decompress(content).decode()
        U = {}
        j=0
        fn=fn.split('\n')
        for line in fn[:-1]:
            line = line.strip().split(';')
            U[nodes[j]] = np.zeros(self.eta)
            if line[0] == '0':
                U[nodes[j]] = np.ones(self.eta)
            elif len(line)==1:
                for i in range(int(line[0]),self.eta):
                    U[nodes[j]][i] = 1.0
            elif line[1]=='':
                for i in range(int(line[0]),self.eta):
                    U[nodes[j]][i] = 1.0
            else:
                for i in range(int(line[0]), int(line[0])+len(line)-1):
                    U[nodes[j]][i] = float(line[i-int(line[0])+1])
                for i in range(int(line[0])+len(line), self.eta):
                    U[nodes[j]][i] = 1.0
            j+=1
        self.hU[u_name] = U
        return U

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

        temp_edges, temp_nodes = set(), set()
        for pairs in r_pairs:
            for pair in pairs:
                stedge = pair[-2] + '-' + pair[-1]
                temp_edges.add(stedge)
                temp_nodes.add(pair[-2])
                temp_nodes.add(pair[-1])

        A = [0] * 4
        for edge in vedge_desty:
            if edge not in edge_desty and edge not in speed_dict:
                if edge in temp_edges: 
                    continue
                edge_ = edge.split('-')
                if np.random.rand() > 0.5: continue
                cost1 = vedge_desty[edge].keys()
                cost = min(float(l) for l in cost1)
                distan = self.get_distance(points, (edge_[0], edge_[1]))
                if distan < 5: A[0] += 1
                elif distan < 10: A[1] += 1
                elif distan < 25: A[2] += 1
                else: A[3] += 1
                if distan > 17: continue 
                All_edges.append((edge_[0], edge_[1], cost))
                all_edges.add(edge)
        for edge in temp_edges:
            if edge in vedge_desty:
                del vedge_desty[edge]
        print('len of vedge_desty 2: %d'%len(vedge_desty))
        all_nodes, all_edges = list(all_nodes), list(all_edges)
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        G.add_weighted_edges_from(All_edges)
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

 
    def rout(self, start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, U, G, points):
        path = []
        Q, P_hat = [], []
        neigh = list(G.successors(start))
        print('neigh %d'%len(neigh))
        start_ax = points[start]
        desti_ax = points[desti]
        s_d_arr = [desti_ax[0] - start_ax[0], desti_ax[1] - start_ax[1]]
        all_expire = 0.0 
        def get_weight(p_hat):
            w_p_hat = {}
            if p_hat in edge_desty:
                w_p_hat = edge_desty[p_hat]
            elif p_hat in vedge_desty:
                w_p_hat = vedge_desty[p_hat]
            elif p_hat in speed_dict:
                w_p_hat = speed_dict[p_hat]
            return w_p_hat

        def get_maxP(w_p_, vv):
            p_max = 0.0
            for pk in w_p_:
                try:
                    w_pk = w_p_[pk]
                    pk_ = int(int(pk) / self.sigma)
                    if int(pk) % self.sigma == 0: pk_ += 1
                    if len(U[vv]) <= pk_ or pk_ <= 0:
                        pk_ = len(U[vv]) - 1
                    p_max += float(w_pk ) * U[vv][pk_]
                except Exception as e:
                    print(e)
            return p_max
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
            tcost1 = time.time()
            inx_min = np.argwhere(U[vi] > 0)
            if len(inx_min) == 0: continue
            inx_min = inx_min[0][0]
            all_expire += time.time() - tcost1
            cost_time = w_min + inx_min*self.sigma
            if cost_time <= self.T:
                p_max = get_maxP(w_p_hat, vi)   
                Que[p_hat] = p_max
                Q[p_hat] = (p_max, w_p_hat, cost_time)
        print('len Q %d'%len(Q))
        QQ = {}
        p_best_p, flag = 'none', False
        p_max_m, p_best_cost, p_w_p = -1, -1, -1
        all_rounds = 0
        if len(Q) == 0: return 'none1', -1, -1, -1, all_expire, all_rounds 
        while len(Q) != 0:
            (p_hat, pqv) = Que.popitem()
            all_rounds += 1
            if p_hat not in Q:
                (p_max, w_p_hat, cost_time)
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
            cost_sv = min([float(l) for l in w_p_hat.keys()])
            vd_d_arr = [points[desti][0]-points[v_l][0], points[desti][1]-points[v_l][1]]
            for u in neigh:
                if u == desti:
                    vu = v_l + '-' + u
                    w_vu = get_weight(vu)
                    if len(w_vu) == 0: cost_vu = 0
                    else: cost_vu = min([float(l) for l in w_vu.keys()])
                    tcost1 = time.time()
                    inx_min = np.argwhere(U[u] > 0)
                    inx_min = inx_min[0][0]
                    p_best_p = p_hat + ';' + vu
                    p_w_p = self.conv(w_p_hat, w_vu)
                    p_max_m = get_maxP(w_p_hat, u)
                    all_expire += time.time() - tcost1
                    p_best_cost = cost_sv + cost_vu + inx_min*self.sigma
                    flag = True
                    break
                if u in has_visit: 

                    continue
                else: has_visit.add(u)
                if u in p_hat: 
                    continue
                vu = v_l + '-' + u
                w_vu = get_weight(vu)
                if len(w_vu) == 0:
                    continue
                cost_vu = min([float(l) for l in w_vu.keys()])
                p_order = nodes_order[u] #p_hat]
                tcost1 = time.time()
                inx_min = np.argwhere(U[u] > 0)
                if len(inx_min) == 0 : 
                    continue
                inx_min = inx_min[0][0]
                all_expire += time.time() - tcost1
                cost_time = cost_sv + cost_vu + inx_min*self.sigma
                if cost_time <= self.T:
                    p_hat_p = p_hat + ';' + vu
                    w_p_hat_p = self.conv(w_p_hat, w_vu)
                    p_hat_max = get_maxP(w_p_hat, u)
                    QQ[p_hat_p] = (p_hat_max, w_p_hat_p, cost_time)
            if flag: break
            if len(Q) == 0:
                Q = copy.deepcopy(QQ)
                for qqk in QQ:
                    Que[qqk] = QQ[qqk][0]
                QQ = {}

        return p_best_p, p_max_m, p_best_cost, p_w_p, all_expire, all_rounds
        
    def main(self, ):
        vedge_desty, edge_desty, path_desty = self.get_dict()
        points = self.get_axes()
        df2 = open(self.query_name, 'rb')
        r_pairs = pickle.load(df2)
        df2.close()
        edges, nodes, G, G2, speed_dict = self.get_graph2(edge_desty, vedge_desty,r_pairs, points)

        nodes_order, i = {}, 0
        for node in nodes:
            nodes_order[node] = i
            i += 1
        PT = 5
        plot_data1, plot_kl, plot_lcs, plot_gt_kl= [0.0]*PT, [0.0]*PT, [0.0]*PT, [0.0]*PT
        sums = [0] * PT
        all_iters  = 0
        all_mps = [0.0]*PT
        all_mps2 = [0.0]*PT
        ls = 0
        one_plot = []
        All_rounds = np.zeros(20).reshape(4, 5)
        One_Plot = np.zeros(20).reshape(4, 5)
        One_Plot2 = np.zeros(20).reshape(4, 5)
        One_Sums = np.zeros(20).reshape(4, 5)
        one_dis = -1
        stores = {}
        cate = ['0-5km', '5-10km', '10-25km', '25-35km']
        for pairs in r_pairs[1:]:
            one_dis += 1
            print('distance category %s'%cate[one_dis])
            for pair_ in pairs:
                _ = self.get_U2(pair_[-1])
            dij_time = 0.0
            sums2 = 0
            cost_t2 = 0
            for pair_ in pairs:
                all_iters += 1
                tstart = time.time()
                print('o-d pair: %s'%pair_[0]+'-'+pair_[1])
                start, desti = pair_[-2], pair_[-1]
                pred = self.get_dijkstra3(G, desti)
                path_, st1 = [start], start
                distan2 = 0
                if start not in pred:
                    continue
                while st1 != desti:
                    st2 = st1
                    st1 = pred[st1]
                    path_.append(st1)
                    distan2 += self.get_distance(points, (st2, st1))
                distan = self.get_distance(points, (start, desti))
                st_key = start + '+' + desti + ':'+str(distan)+':'+str(distan2)
                stores[st_key] = {}
                st1, time_budget = start, 0.0
                for st2 in path_[1:]:
                    sedge = st1+'-'+st2
                    if sedge in edge_desty:
                        speed_key = list([abs(float(l)) for l in edge_desty[sedge].keys()])
                        time_budget += (min(speed_key)+ max(speed_key))/2
                    elif sedge in vedge_desty:
                        speed_key = list([abs(float(l)) for l in vedge_desty[sedge].keys()])
                        time_budget += (min(speed_key) + max(speed_key))/2
                    elif sedge in speed_dict:
                        speed_key = list([abs(float(l)) for l in speed_dict[sedge].keys()])
                        time_budget += (min(speed_key) + max(speed_key)) / 2
                    else: 
                        print(' edge: %s not in speed_dict, exit'%sedge)
                        sys.exit()
                    st1 = st2 
                print('time budget: %f'%time_budget)
                U = self.get_U2(desti)
                if len(U) == 0: continue
                tend = time.time()
                ls += 1
                all_kl_ = []
                for t_b_, t_b in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
                    tstart = time.time()
                    self.T = time_budget * t_b
                    best_p, max_m, best_c, best_pw, all_expires, all_rounds = self.rout(start, desti, edge_desty, vedge_desty, speed_dict, nodes_order, U, G, points)
                    stores[st_key][str(t_b)] = [time.time()-tstart-all_expires, all_rounds]
                    if best_p == 'none1' or best_p == 'none2' or best_p == 'none':
                        all_rounds = 0
                        continue
                    tend = time.time()
                    plot_data1[t_b_] += tend - tstart
                    sums[t_b_] += 1

                    One_Plot[one_dis][t_b_] += tend - tstart 
                    One_Plot2[one_dis][t_b_] += tend - tstart - all_expires
                    One_Sums[one_dis][t_b_] += 1
                    All_rounds[one_dis][t_b_] += all_rounds

                    if t_b_ == 2:
                        sums2 += 1
                        cost_t2 += tend - tstart 
            print(One_Plot)
            print(One_Plot2 )
            print(One_Sums)
        for i in range(PT):
            if sums[i] == 0:
                continue
            plot_data1[i] /= sums[i]
        # One_Plot = One_Plot / One_Sums 
        # One_Plot2 = One_Plot2 / One_Sums 
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

        with(open(subpath+"V-BS-result.txt", 'a+')) as fw:
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
        elif args.sig == 4:
            sigma, eta = 120, 80
        elif args.sig == 5:
            sigma, eta = 240, 33
        else:
            print('wrong sig , exit')
            sys.exit()

        threads_num = 10
        dinx = 50
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

        time_budget = 5000
        rout = Rout(fpath, time_budget, fpath_desty, fedge_desty, subpath, axes_file, speed_file, sigma, eta, query_name)
        rout.main()
        print("finished")
    except Exception as e:
        print(e)
        trace = sys.exc_info()
        traceback.print_exception(*trace)
        del trace
    print("finished")