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
import pickle
from get_tpath import TPath
from v_opt import VOpt
from math import cos, asin, sqrt, pi



def distance(lat1, lon1, lat2, lon2):
    r = 6731  # 地球平均半径
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 2 * r * asin(sqrt(a))
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
                speed_dict[line[0]] = {3600*float(line[1])/float(line[2]): 1.0}
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
        # return geodesic((lo1, la1), (lo2, la2)).kilometers
        return distance(la1, lo1, la2, lo2)

    def get_distance2(self, points, path_1, path_2):
        a = path_1.find('-')
        point1 = path_1[:a]
        a = path_1.rfind('-')
        point2 = path_1[a+1:].split(";")[0]
        (la1, lo1) = points[point1]
        (la2, lo2) = points[point2]
        # d1 = geodesic((lo1, la1), (lo2, la2)).kilometers
        d1=distance(la1, lo1, la2, lo2)
        a = path_2.find('-')
        point3 = path_2[:a]
        a = path_2.rfind('-')
        point4 = path_2[a+1:].split(";")[0]
        (la3, lo3) = points[point3]
        (la4, lo4) = points[point4]
        # d2 = geodesic((lo3, la3), (lo4, la4)).kilometers
        # d3 = geodesic((lo1, la1), (lo4, la4)).kilometers
        # d4 = geodesic((lo2, la2), (lo3, la3)).kilometers
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
        # fn = open(self.subpath+self.foverlap)
        # overlap = {}
        # for line in fn:
        #     temp = line.strip().split(':')
        #     overlap[temp[0]] = temp[1].split(',')
        # fn.close()
        with open(self.subpath + self.foverlap, 'rb') as f:
            content = f.read()
            a = gzip.decompress(content).decode()
            overlap = json.loads(a)
        return overlap

    def get_dict(self, ):
        # with open(self.subpath+self.fpath_desty) as js_file:
        #     path_desty = json.load(js_file)
        #     #path_desty =  dict(sorted(path_desty.items(), key=operator.itemgetter(0), reverse=False))
        # with open(self.subpath+self.fpath_count) as js_file:
        #     path_count = json.load(js_file)
        #     #path_count =  dict(sorted(path_count.items(), key=operator.itemgetter(0), reverse=False))

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
        start = path_.find('-')
        head = path_[:start]
        end = path_.rfind('-')
        tail = path_[end+1:]
        if head in path_[start:] or tail in path_[:end]:
            return True
        path_ = path_.split('-')
        path_2 = set(path_)
        if len(path_) > len(path_2):
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

    def merge(self, p1, p2, f, s, vopt, path_desty, edge_time_freq):

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
                            n_k = str(int(wk1[i]) + int(wk2[j]) - int(wks[l]))
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
                    
        speed=self.get_speed()
        w1 = path_desty[p1]
        w2 = path_desty[p2]
        if s in path_desty:
            ws = path_desty[s]
        else:
            if s in edge_time_freq:
                ws = edge_time_freq[s]
            else:
                #self.mid_olp[ws] = 
                # ws = edge_time_freq[s.split(';')[0]]
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
        B1, B2 = list(BB.keys()), list(BB.values())
        dom, B_len = 0, len(B1)
        for (_, Q) in AA:
            q_len = len(Q)
            M = min(B_len, q_len)
            A1, A2 = list(Q.keys()), list(Q.values())
            a, b = 0, 0
            for m in range(M):
                if int(A1[m]) > int(B1[m]):
                    a += 1
                elif int(A1[m]) < int(B1[m]):
                    b += 1
                else:
                    if float(A2[m]) > float(B2[m]):
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
            for p_in in in_list_:
                #print('p_in %s'%p_in)
                tp_in, spoint = self.get_tpaths(p_in)
                for t_p in tp_in:
                    # if t_p not in overlap :
                        # print(t_p)
                    dict_inx = -1
                    if t_p in overlap:
                        T_overlapping = overlap[t_p]
                    else:
                        # print(t_p)
                        continue
                    # if len(T_overlapping[0]) < 1:  #T_overlapping[0] 是int
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
                        #print('TP_new %s'%TP_new)
                        if psi == '' or not is_s: continue
                        if TP_new in TP: continue
                        if self.check_loop(psi): continue
                        if psi not in path_num and psi not in over_:
                            if not fg:
                                t_over = self.merge(path_1, path_2, psi, s, vopt, path_desty, edge_time_freq)
                                new_ = 'v;'+t_o+';'+p_in[spoint:]
                            else:
                                t_over = self.merge(path_2, path_1, psi, s, vopt, path_desty, edge_time_freq)
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
            return [path_num_, over_]  # , All_Path]
        except Exception as e:
            print("There was an error at pid %d %d"%(os.getpid(), all_))
            print(e)
            trace = sys.exc_info()
            traceback.print_exception(*trace)
            del trace

        #finally:
            # return []


    def collect_res(self, result):
        self.result.append(result)

    def gen(self, vopt, path_desty, path_count, path_num, overlap, edge_time_freq, TP, points):
        in_list = list(path_count.keys())
        #in_list = in_list[:20]
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
        while len(in_list) > 0: 
            self.result = []
            print('length of in list %d' %len(in_list))
            print('path desty %d'%(len(path_desty)))
            print('path count %d'%(len(path_count)))
            print('path num %d'%(len(path_num)))
            begin_t = time.time()
            len_in = len(in_list)
            if len_in % self.threads_num == 0:
                t_inxs = int(len_in/self.threads_num)
            else:
                t_inxs = int(len_in/self.threads_num) + 1
            pool = multiprocessing.Pool(self.threads_num)
            for len_thr in range(self.threads_num):
                mins = min((len_thr+1)*t_inxs, len_in)
                inxs_array = in_list[len_thr*t_inxs:mins]
                # print(type(inxs_array))
                # print(len(inxs_array))
                pool.apply_async(
                    self.thread_fun,
                    args=(inxs_array,vopt, path_desty, path_count, path_num, overlap, edge_time_freq, TP, All_Path, points),
                    callback=self.collect_res)#.get()
            pool.close()
            time.sleep(10)
            pool.join()
            time.sleep(10)
            #print("res: desty={desty}, freq={freq}, err={err}".format
             #     (desty=self.mergeres["desty"],freq=self.mergeres["freq"],err=self.mergeres["err"]))
            #print("result size = %d"%len(self.result))
            path_num2, path_count2= {}, {}
            print(f"Got to Res: {len(self.result)}")
            # print(self.result[0])
            for res in self.result:
                #in_list.extend(res[-1])
                if res==None:
                    print("there is a none")
                    continue
                var1 = res[0]
                var2 = res[1]
                path_num2.update(var1)
                path_desty.update(var2)
                #All_Path.update(res[2])
                #print(res[0])
                #print(res[1])
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
            #in_list = []
            # with(open(self.subpath+"time.txt", 'a+')) as fw:
            #     fw.write(" "+str(end_time))
            '''out_degree = self.mkgraph.G.degree(self.anodes)
            out_degrees.append([l[1] for l in out_degree])
            out_degree = self.mkgraph.G.degree(self.nodes)
            out_degrees2.append([l[1] for l in out_degree])
            #out_degrees.append(out_degree)
            kkk += 1'''
            print("end of while")

        return path_desty, All_Path #out_degrees, out_degrees2
    
    def norms(self, dicts2, is_norm):
        for e_key in dicts2:
            sums = sum(dicts2[e_key].values())
            for time in dicts2[e_key]:
                dicts2[e_key][time] = round(100.0*dicts2[e_key][time]/sums, 6)
            if is_norm:
                dicts2[e_key] =  dict(sorted(dicts2[e_key].items(), key=operator.itemgetter(1), reverse=True))
        return dicts2

    def get_vopt(self, vopt, dicts, is_path=False, is_norm=False):
        for e_key in dicts:
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
        #json.dumps(js_dict, fname)
        with open(fname, 'w') as fw:
            json.dump(js_dict, fw, indent=4)

    def write_gzip(self, js_dict, fname):
        js_dict_ = json.dumps(js_dict)
        compressed_string = gzip.compress(js_dict_.encode())
        with open(fname, 'wb') as fw:
            fw.write(compressed_string)

    # def write_degree(self, out_degrees_, fname):
    #     temp = pd.DataFrame()
    #     temp['node'] = out_degrees_[0]
    #     strs = 0
    #     for out_degree in out_degrees_[1:]:
    #         temp[str(strs)] = out_degree
    #         strs += 1
    #     temp.to_csv(fname, sep=';', index=None)
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
        #for edge_ in self.mkgraph.G.edges():
            strs += edge_[0]+'-'+edge_[1]+'\n'
        # with open(fname, 'w') as fw:
        #     fw.write(strs)
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
        points = self.get_axes()#点的经纬度
        path_desty, path_count, path_num, overlap, TP = self.get_dict()
        
        vopt = VOpt()
        print('v opt edge ...')
        # self.edge_time_freq = multiprocessing.Manager().dict()
        # edge_time_freq = self.get_vopt(vopt, edge_time_freq, False, True) TODO fix this
        path_desty = self.get_vopt(vopt, path_desty, True, True)
        # self.write_json(path_desty, self.subpath + "new_desty_10.json")
        print('gen ...')
        #self.vopt = vopt
        path_desty, All_Path = self.gen(vopt, path_desty, path_count, path_num, overlap, edge_time_freq, TP, points)
        print('path desty length %d'%len(path_desty))
        #self.write_degree(out_degrees, self.subpath+self.store_degree_fname)
        #self.write_degree(out_degrees2, self.subpath+self.store_degree_fname2)
        self.store_graph(self.subpath+self.graph_store_fname)
        All_Path_ = {}
        for ak in All_Path:
            All_Path_[ak] = All_Path[ak]
        self.write_gzip(All_Path_, self.subpath+self.store_desty_fname)
        print("Finished")

if __name__ == '__main__':
    # dinx = 100
    dinxs = [50]
    process_num = 10
    datasets=["peak","offpeak"]
    city='cd'  ##cd aal
    # datasets=["offpeak"]
    for dataset in datasets:
        all_running_time=[]
        for dinx in dinxs:
            if city=='aal':
                filename = '../data/AAL_short_50_'+dataset+'.csv'
                subpath = '../data/'+dataset+'_res%d/' % dinx
                graph_path = '../data/AAL_NGR'
                axes_file = '../data/aal_vertices.txt'
            else:
                filename = '../data/cd/trips_real_'+str(dinx)+'_'+dataset+'.csv'
                subpath = '../data/'+dataset+'_cd_res%d/' % dinx
                graph_path = '../data/full_NGR'
                axes_file = '../data/cd_vertices.txt'
            p_path = 'path_travel_time_'
            # fpath_count = 'path_count%d.json'%dinx
            # fpath_desty = 'path_desty%d.json'%dinx
            fpath_count = 'path_count%d.txt' % dinx
            fpath_desty = 'path_desty%d.txt' % dinx
            foverlap = 'overlap%d.txt'% dinx
            # axes_file = '../data/vertices.txt'
            subpath_range = [l for l in range(2, 12)]
            B = 3
            threads_num = 10
            # store_desty_fname = 'KKdesty_num_%d.json'%threads_num
            # store_degree_fname = 'KKdegree_%d.json'%threads_num
            # store_degree_fname2 = 'KKdegree2_%d.json'%threads_num
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
            #in_list = []
        with(open('../data/'+city+dataset+"_running_time.txt", 'w+')) as fw:
            all_time=""
            for i in range(len(all_running_time)):
                all_time+=str(dinxs[i])+": "+str(all_running_time[i])+"\n"
            fw.write(all_time)


