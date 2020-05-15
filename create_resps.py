import csv
import addfips
import numpy as np
from KSS_EM import em

####################################################################
# HELPER FUNCTION
####################################################################

# shift responsibilities so alpha_est*n of them are positive
def shift_resps(resps, alpha_est):
    n = np.size(resps)
    k = int(round(alpha_est*n))
    sorted_resps = np.sort(resps)[::-1]
    threshold = 0.5*(sorted_resps[max(0, k-1)] + sorted_resps[min(k, n-1)])
    return resps - threshold

####################################################################
# MAIN
####################################################################

C = np.zeros(n) # TODO: Fill in expected counts
B = np.zeros(n) # TODO: Fill in observed counts
node_names = [] # TODO: Fill in with names of each node
resp_path = "path/to/folder/scores_resp.tsv" # TODO: Fill in path to folder


qin_est,qout_est,alpha_est,resps_est = em(C,B)
resps_shift = shift_resps(resps_est, alpha_est)

with open(resp_path, "w+") as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    for i,c in enumerate(county_fips):
        tsv_writer.writerow([str(node_names[i]), str(resps_shift[i])])