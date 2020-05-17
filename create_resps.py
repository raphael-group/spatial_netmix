import csv
import addfips
import numpy as np
from KSS_EM import em

####################################################################
# HELPER FUNCTIONS
####################################################################

# shift responsibilities so alpha_est*n of them are positive
def shift_resps(resps, alpha_est):
    n = np.size(resps)
    k = int(round(alpha_est*n))
    sorted_resps = np.sort(resps)[::-1]
    threshold = 0.5*(sorted_resps[max(0, k-1)] + sorted_resps[min(k, n-1)])
    return resps - threshold

# converts column col of some tsv list (no headers) to numpy float array
def tsv_to_np(filename, col=0):
    
    res = []
    
    with open(filename, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
    
        for row in reader:
            if len(row[col]) > 0:
                res.append(float(row[col]))
            
    return np.array(res)

def tsv_to_array(filename, col=0):
    res = []
    with open(filename, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
    
        for row in reader:
            if len(row[col]) > 0:
                res.append(row[col])
            
    return res

####################################################################
# MAIN
####################################################################

# TODO: Fill in paths
expected_path = '' 
observed_path = ''
names_path = ''
resp_path = "path/to/folder/scores_resp.tsv"

C = tsv_to_np(expected_path) 
B = tsv_to_np(observed_path) 
node_names = tsv_to_array(names_path)

qin_est,qout_est,alpha_est,resps_est = em(C,B)
resps_shift = shift_resps(resps_est, alpha_est)

with open(resp_path, "w+") as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    for i,c in enumerate(county_fips):
        tsv_writer.writerow([str(node_names[i]), str(resps_shift[i])])