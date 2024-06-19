import json
import sys
import traceback
import numpy as np
import gzip
from get_tpath import TPath
from v_opt import VOpt
import operator

def norms(dicts2, is_norm):
    for e_key in dicts2:
        sums = sum(dicts2[e_key].values())
        for time in dicts2[e_key]:
            dicts2[e_key][time] = round(100.0*dicts2[e_key][time]/sums, 6)
        if is_norm:
            dicts2[e_key] =  dict(sorted(dicts2[e_key].items(), key=operator.itemgetter(1), reverse=True))
    return dicts2

def get_vopt(vopt, dicts, is_path=False, is_norm=False):
    for e_key in dicts:
        time_freq = dicts[e_key]
        # time_cost, time_freq = [str(l) for l in time_freq.keys()], [float(l) for l in time_freq.values()]
        time_cost, time_freq = [float(l) for l in time_freq.keys()], [float(l) for l in time_freq.values()]
        pvalues = {}
        if len(time_cost) > 1:
            pvalues = vopt.v_opt(time_cost, time_freq, B) # pvalues is a dict, contains new key and value
            dicts[e_key] = pvalues
        else:
            if not is_path:
                k1 = str(int(np.mean(time_cost)))
                pvalues[k1] = 1.0
    dicts_={}
    for i in dicts:
        dicts_[i]={int(j):dicts[i][j] for j in dicts[i]}
    return norms(dicts_, is_norm)

def get_edge_time_freq(data):
    try:
        seg_ids = data.seg_id.values
        travel_time=data.travel_time.values
        N = len(seg_ids)
        edge_time_freq = {}
        for i in range(N):
            if seg_ids[i] not in edge_time_freq:
                edge_time_freq[seg_ids[i]] = {str(travel_time[i]):1}
            else:
                if str(travel_time[i]) not in edge_time_freq[seg_ids[i]]:
                    edge_time_freq[seg_ids[i]][travel_time[i]] = 1
                else:
                    edge_time_freq[seg_ids[i]][travel_time[i]] += 1
        return edge_time_freq
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # dinx = 100
    # fname = '../data/Aalborg_2007.csv'
    #fname = '../data/AAL_short_50_peak.csv'
    #
    fname = '../data/AAL_short_50_offpeak.csv'
    # dinxs = [15, 30, 50, 100]
    dinxs = [50]
    city='xa'  ##cd aal xa
    flag=1
    datasets=["offpeak"]
    for dataset in datasets:
        for dinx in dinxs:
            if city=='aal':
                fname = "../data/aal/trips_real_"+str(dinx)+"_"+dataset+'.csv'
                subpath = '../data/'+dataset+'_res%d' % dinx+'_%d/' %flag
            elif city=='cd':
                fname = '../data/cd/trips_real_'+str(dinx)+'_'+dataset+'.csv'
                subpath = '../data/'+dataset+'_cd_res%d' % dinx+'_%d/' %flag
            else:
                fname = '../data/xa/new_days_trips_real_'+str(dinx)+'_'+dataset+'.csv'
                subpath = '../data/'+dataset+'_xa_res%d' % dinx+'_%d/' %flag
            B = 3
            # subpath='../data/res%d/'%dinx
            # subpath = '../data/peak_res%d/' % dinx
            # subpath = '../data/offpeak_res%d/' % dinx
            tpath = TPath(fname, subpath=subpath, process_num=50, dinx=50)
            data = tpath.load(flag)
            print("done")
            # edge_desty = tpath.get_edge_freq(data)
            # print(edge_desty)
            vopt = VOpt()
            #edge_desty_ = get_vopt(vopt, edge_desty, False, True)
            # with (open("../data/res10/path_desty10.json")) as file:
            #     edge_desty_ = json.load(file)
            with open(subpath+"edge_travel_time.txt", 'rb') as file:
                content = file.read()
                a = gzip.decompress(content).decode()
                edge_desty = json.loads(a)

            # edge_desty = get_edge_time_freq(data)

            edge_desty_ = get_vopt(vopt, edge_desty, False, True)
            keys = list(edge_desty_.keys())
            print(keys[0])
            print(edge_desty_[keys[0]])
            edge_desty_1 = {}
            try:
                for edge in edge_desty_:
                    #ky = list(edge.keys())
                    edge_ = {}
                    ky = edge_desty_[edge]
                    for k in ky.keys():
                        edge_[str(np.abs(int(k)))] = float(ky[k])/100
                    edge_desty_1[edge] = edge_
            except Exception as e:
                print(e)
                trace = sys.exc_info()
                traceback.print_exception(*trace)
                del trace
            #ffname = '../res3/M_vedge_desty_10_2.json'
            # ffname = subpath+'M_edge_desty.json'
            # with open(ffname, 'w') as fw:
            #     json.dump(edge_desty_1, fw, indent=4)

            ffname = subpath+'M_edge_desty.txt'   ###存储每个路径的cost分布
            js_dict_ = json.dumps(edge_desty_1)
            compressed_string = gzip.compress(js_dict_.encode())
            with open(ffname, 'wb') as fw:
                fw.write(compressed_string)
        print("finished")