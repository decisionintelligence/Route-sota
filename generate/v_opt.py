import numpy as np


class VOpt():
    def __init__(self,):
        pass

    def v_opt(self, cost, data, B):
        N = len(data)
        p, pp = [0]*(N+1), [0]*(N+1)
        best_error = np.zeros((N+1)*(B+1)).reshape((B+1), N+1)
        min_index = [-1] * (N+1)
        for i in range(1, N+1):
            p[i] = p[i-1] + data[i-1]
            pp[i] = pp[i-1] + data[i-1] * data[i-1]
        def sq_error(a, b):
            s2 = pp[b] - pp[a-1]
            s1 = p[b] - p[a-1]
            return s2 - s1*s1/(b-a+1)
        min_index[1] = 1
        for k in range(1, B+1):
            for i in range(1, N+1):
                if k == 1:
                    best_error[k][i] = sq_error(1, i)
                else:
                    best_error[k][i] = np.inf
                    for j in range(1, i):
                        if best_error[k-1][j] + sq_error(j+1, i) < best_error[k][i]:
                            best_error[k][i] = best_error[k-1][j] + sq_error(j+1, i)
                            min_index[i] = j + 1

        i = B
        j = N
        cuts = []
        while i >= 2:
            end_point = j
            j = min_index[j]
            #print('[%d .. %d]' % (j, end_point))
            cuts.append((j-1, end_point-1))
            i -= 1
            j -= 1
        #print('[%d .. %d]' % (1, j))
        cuts.append((1-1, j-1))
        #print(cuts)
        cuts.reverse()
        #pvalues = [] 
        pvalues = {}
        for cut in cuts:
            if cut[1] >= cut[0] and cut[0] >=0 :
                sums = 0.0
                coss = 0.0
                for i in range(cut[0], cut[1] + 1):
                    sums += data[i]
                    coss += cost[i]
                coss = int(coss / (cut[1] - cut[0] + 1))
                #pvalues.append('%f:%f'%(coss, sums/sum(data)))
                pvalues[coss] = round(sums/sum(data),6)
                #pvalues[coss] = int(sums/sum(data))
        #pvalues /= N
        #print(pvalues)
        return pvalues





