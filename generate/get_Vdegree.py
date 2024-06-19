import time
import traceback

import numpy as np
import queue 
import json 
import sys, os
import multiprocessing
from multiprocessing import Process
from pqdict import PQDict
import pickle 
import argparse
import gzip,math
import networkx as nx
from get_tpath import TPath
from tqdm import tqdm

class Rout:
    #def __init__(self, G, maxsize, node_size, eta, node_list, graph_store_name, time_budget ):
    def __init__(self, subpath, graph_store_name, time_budget, filename, fpath_desty, maxsize, fedge_desty, umatrix_path, process_num, speed_file, query_file, eta=12, sigma=3):
        self.maxsize = maxsize
        self.eta = eta
        self.sigma = sigma
        #self.Q = queue.Queue(maxsize=maxsize)
        self.graph_store_name = graph_store_name
        self.time_budget = time_budget 
        self.filename = filename
        self.fpath_desty = fpath_desty
        self.subpath = subpath
        os.makedirs(subpath, exist_ok=True)
        self.fedge_desty = fedge_desty
        self.umatrix_path = umatrix_path
        self.process_num = process_num
        self.speed_file = speed_file
        self.query_file = query_file

    def get_speed(self, ):
        speed_dict = {}
        with open(self.speed_file) as fn:
            for line in fn:
                line = line.strip().split('\t')
                speed_dict[line[0]] = {str(math.ceil(3600*float(line[1])/float(line[2]))): 1.0}   ##根据限速计算最小代价
        return speed_dict

    def get_graph(self, edge_desty):
        with open(self.subpath + self.graph_store_name, 'rb') as f:
            content = f.read()
            fn = gzip.decompress(content).decode()   
        
        lines=set(fn.split('\n'))
        count_num={}
        for i in tqdm(lines):
            if len(i)<=2:
                continue
            out=i.split('-')[0]
            if out not in count_num:
                count_num[out]=0
            count_num[out]+=1
        print(max(count_num.values())/10)
        print(sum(count_num.values())/117415)
        return 0
    
    def get_dict(self, ):
        
        # with open(self.subpath+self.fedge_desty) as js_file:
        #     edge_desty = json.load(js_file)

        with open(self.subpath + self.fedge_desty, 'rb') as f:
            content = f.read()
            a = gzip.decompress(content).decode()
            edge_desty = json.loads(a)

        # f=gzip.open(self.subpath+self.fedge_desty, 'rb')
        # print(f[:100])
        # edge_desty = json.loads(f)
        # f.close()
        return edge_desty


    def compute_one_rowu(self, v_i, v_d, sucpre, edge_desty, nodes_order, U, Q, G, iset, pred):
        st1, vimin, v_s = v_i, 0.0, sucpre[v_i]
        vi_order, vs_order = nodes_order[v_i], nodes_order[v_s]
        while st1 != v_d:
            st2, st1 = st1 , pred[st1]
            tedge = st2 + '-' + st1
            if tedge in edge_desty:
                vimin += min(list([float(l) for l in edge_desty[tedge].keys()]))
            else:
                print('error ... ')
        row_l = vimin
        row_s = self.eta * self.sigma
        for i in range(self.eta):
            if i * self.sigma < row_l:
                U[vi_order][i] = 0.0
        row_x = row_l
        #row_inx = int(row_l / self.sigma)
        while row_x < self.eta * self.sigma:
            i = int(row_x / self.sigma)
            U[vi_order][i] = 0
            for row_c in edge_desty[v_i+'-'+v_s]:
                #inx_xc = int((row_x-float(row_c))/self.sigma) 
                inx_xc =  int((row_x-float(row_c))/self.sigma) 
                if inx_xc < 0 : continue
                U[vi_order][i] = U[vi_order][i] + edge_desty[v_i+'-'+v_s][row_c] * U[vs_order][inx_xc]
            if U[vi_order][i] < 1:
                row_x = row_x + self.sigma
            else:
                break
        row_s = min(row_s, row_x)
        for i in range(int(row_s/self.sigma), self.eta):
            U[vi_order][i] = 1.0 
        return True


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
            #if v == target: break
            neigh = list(Gk.successors(v))
            for u in neigh:
                if u in U:
                    d = D[v] + Gk[v][u]['weight']
                    if d < Que.get(u, inf):
                        Que[u] = d
                        P[u] = v
        return P


    def rout_(self, v_d, edge_desty, edges, nodes, nodes_order, G):
        try:
            pred = self.get_dijkstra3(G, v_d)
            N = len(nodes)
            U = np.zeros(N * self.eta).reshape(N, self.eta)
            U.fill(-1)
            Q = []
            iset = set()
            print('pid %d, v d %s' %(os.getpid(), v_d))
            if len(Q) != 0:
                print('error, Q is not empty, exit')
                sys.exit()
            Q.append(v_d)
            iset.add(v_d)
            for j in range(self.eta):
                U[nodes_order[v_d]][j] = 1.0
            lsucess, lorder = '', -1
            flag = True
            sucpre = {}
            while len(Q) != 0:
                v_i = Q.pop(0)
                v_order = nodes_order[v_i]
                #print('v_order %d'%v_order)
                #print('len of predecessors: %d'%(len(list(self.G.predecessors(v_i)))))
                if v_i == v_d:
                    U[v_order].fill(1) # = 1
                    iset.add(v_i)
                else:
                    #self.computeU(v_i, v_order, edge_desty, nodes_order, U, Q, G, iset)
                    self.compute_one_rowu(v_i, v_d, sucpre, edge_desty, nodes_order, U, Q, G, iset, pred)
                    iset.add(v_i)

                #print(list(G.predecessors(v_i)))
                for v in list(G.predecessors(v_i)):
                    try:
                        val = nodes_order[v]
                        if U[nodes_order[v]][0] == -1:
                            if v not in Q :
                            #if v not in iset:
                                sucpre[v] = v_i
                                Q.append(v)
                    except Exception as e:
                        print(e)
                        trace = sys.exc_info()
                        traceback.print_exception(*trace)
                        del trace
            print('pid %d, v_d %s, len iset %d'%(os.getpid(), v_d, len(iset)))
            U = np.round(U, 4)
            #sys.exit()
            return U
        except Exception as e:
            print(e)
            trace = sys.exc_info()
            traceback.print_exception(*trace)
            del trace

    
    def rout(self, sub_nodes, edge_desty, edges, nodes, nodes_order, G):
        try:
            print('pid %d'%os.getpid())
            for v_d_ in sub_nodes:  ###sub_nodes 
                U_1 = self.rout_(v_d_, edge_desty, edges, nodes, nodes_order, G)
                self.save_matrix(v_d_, U_1, nodes)
                print("saved")
        except Exception as e:
            print(e)
    
    def write_json(self, dicts, name):
        # with open(self.subpath+name, 'w' ) as fw:
        #     json.dump(dicts, fw, indent=4)
        js_dict_ = json.dumps(dicts)
        compressed_string = gzip.compress(js_dict_.encode())
        with open(name, 'wb') as fw:
            fw.write(compressed_string)

    def save_matrix(self, name, U_2, nodes):
        strs = ''
        min_strs=''
        for ix, ky in enumerate(nodes):
            U_ = U_2[ix]
            AL = np.argwhere(U_ > 0)
            AH = np.argwhere(U_ > 0.999999)
            if len(AL) == 0:
                al = 0
                strs += str(0)+'\n'
                min_strs += str(0)+'\n'
            else:
                al = AL[0][0]
                if len(AH) == 0:
                    ah = self.eta
                else:
                    ah = AH[0][0]
                if al==ah:
                    strs += str(al)+'\n'
                else:
                    strs += str(al)+';'+';'.join(str(l) for l in U_[al:ah])+'\n'
                min_strs+=str(al-1)+'\n'
            #strs += ky+';'+';'.join(str(l) for l in U_) + '\n'
        # with(open(self.umatrix_path+name, 'w')) as fw:
        #     fw.write(strs)
                
                
        # compressed_string = gzip.compress(strs.encode())
        # with open(self.umatrix_path+name, 'wb') as fw:
        #     fw.write(compressed_string)

        os.makedirs(os.path.dirname(self.umatrix_path+"/min/"), exist_ok=True)
        compressed_string = gzip.compress(min_strs.encode())
        with open(self.umatrix_path+"/min/"+name, 'wb') as fw:
            fw.write(compressed_string)

        compressed_string = gzip.compress(strs.encode())
        with open(self.umatrix_path+name, 'wb') as fw:
            fw.write(compressed_string)

    def collect_res(self, sub_res):
        self.result.extend(sub_res)

    def main(self,):
        edge_desty = self.get_dict()
        _= self.get_graph(edge_desty)
        print('end ...')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='policy_U')
    parser.add_argument('--sig', default=3, type=int)
    args = parser.parse_args()
    city='xa'  ##cd aal
    datasets=["peak"]
    # datasets=["offpeak"]
    for dataset in datasets:
        sigma_list = [30, 60, 120, 240]
        eta_list = [333, 170, 80, 33]
        sigma_list = [60]
        eta_list = [170]
        all_running_time=[]
        for i in range(len(eta_list)):
            eta, sigma = eta_list[i], sigma_list[i]
            print('eta: %d, sigma: %d' % (eta, sigma))
            process_num = 10
            time_budget = 1000
            maxsize = 10000
            # dinx = 100
            dinxs = [15, 30, 50, 100]
            # dinxs = [50]
            flag=1
            for dinx in dinxs:
                if city=='aal':
                    filename = "../data/aal/trips_real_"+str(dinx)+"_"+dataset+'.csv'
                    subpath = '../data/'+dataset+'_res%d' % dinx+'_%d/' %flag
                    speed_file = '../data/AAL_NGR'
                    axes_file = '../data/aal_vertices.txt'
                    query_file = "../data/queries.txt"
                elif city=='cd':
                    filename = '../data/cd/trips_real_'+str(dinx)+'_'+dataset+'.csv'
                    subpath = '../data/'+dataset+'_cd_res%d' % dinx+'_%d/' %flag
                    speed_file = '../data/full_NGR'
                    axes_file = '../data/cd_vertices.txt'
                    query_file = "../data/cd_queries.txt"
                else:
                    filename = '../data/xa/new_days_trips_real_'+str(dinx)+'_'+dataset+'.csv'
                    subpath = '../data/'+dataset+'_xa_res%d' % dinx+'_%d/' %flag
                    speed_file = '../data/xa/XIAN_R_new.txt'
                    axes_file = '../data/xa/xa_vertices.txt'
                    query_file = "../data/xa/xa_queries.txt"
                # filename = '../data/AAL_short_50_peak.csv'
                # subpath = '../data/peak_res%d/' % dinx
                # filename = '../data/AAL_short_50_offpeak.csv'
                # subpath = '../data/offpeak_res%d/' % dinx
                # subpath = '../data/res%d/'% dinx
                # filename = subpath + "Aalborg_2007.csv"
                # fpath_desty = 'KKdesty_num_%d.json' % process_num
                fpath_desty = 'KKdesty_num_%d.txt' % process_num
                graph_store_name = 'KKgraph_%d.txt' % process_num
                # fedge_desty = 'M_edge_desty.json'
                fedge_desty = 'M_edge_desty.txt'
                umatrix_path = subpath + 'u_mul_matrix_sig%d/' % sigma
                os.makedirs(subpath + "u_mul_matrix_sig%d/" % sigma, exist_ok=True)
                # speed_file = '../data/AAL_NGR'

                try:
                    begin_t = time.time()
                    rout = Rout(subpath, graph_store_name, time_budget, filename, fpath_desty, maxsize, fedge_desty,
                        umatrix_path, process_num, speed_file, query_file, eta, sigma)
                    rout.main()
                    end_time=int(time.time()-begin_t)
                    print('time cost %d'%end_time)
                    all_running_time.append(end_time)
            #in_list = []
                    # rout = Rout(subpath, graph_store_name, time_budget, filename, fpath_desty, maxsize, fedge_desty, umatrix_path, process_num, speed_file, query_file, eta, sigma)
                    # rout.main()
                except Exception as e:
                    print(e)
    print("finished")

