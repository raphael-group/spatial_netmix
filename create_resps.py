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
def tsv_to_np(filename, col):
    
    res = []
    
    with open(filename, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
    
        for row in reader:
            if len(row[col]) > 0:
                res.append(float(row[col]))
            
    return np.array(res)

def tsv_to_array(filename, col):
    res = []
    with open(filename, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
    
        for row in reader:
            if len(row[col]) > 0:
                res.append(row[col])
            
    return res


####################################################################
# SCORE GENERATION
####################################################################

c1 = {'808', '341', '110', '147', '807', '135', '146', '364', '400', '303', '136', '401', '125', '344', '465', '141', '124', '532', '363', '343', '342', '123', '144', '109', '362', '536', '145', '509', '142', '399', '533', '105', '508', '302', '839', '143'}
c2 = {'161', '859', '191', '203', '202', '197', '151', '275', '162', '153', '278', '152', '534', '1055', '276', '196', '163', '204', '199', '871', '180', '195', '771', '1011', '164', '280', '860', '432', '870', '277', '154', '229', '198', '1059'}
c3 = {'850', '993', '991', '319', '46', '321', '322', '42', '987', '849', '118', '47', '122', '49', '48', '116', '320', '992', '835', '523', '986', '836'}
c4 = {'86', '936', '31', '33', '943', '786', '872', '933', '940', '88', '30', '87', '941', '873', '945', '942', '32', '784', '944'}

def generate_scores(expected, q_in, q_out, random_seed):
    np.random.seed(random_seed)
    scores = []
    for i in ids:
        b = expected[i]
        
        if i in c1 or i in c2 or i in c3 or i in c4:
            score = np.random.poisson(q_in * b)
        else:
            score = np.random.poisson(q_out * b)
        scores.append(score)
    return scores

####################################################################
# MAIN
####################################################################

num_trials = 10
alphas = np.zeros(10)
qins = np.zeros(10)
qouts = np.zeros(10)

for i in range(num_trials)

# TODO: Fill in paths
expected_path = './expected.tsv' 
#observed_path = './fake_observed.tsv' 
names_path = './geoids.tsv' 
resp_path = './scores_resp.tsv'

#C = tsv_to_np(observed_path, col=1) 
B = tsv_to_np(expected_path, col=1) 
node_names = tsv_to_array(names_path, col=0)

for i in range(num_trials):
    C = generate_scores(expected, 5, 1, i)


    qin_est,qout_est,alpha_est,resps_est = em(C,B)

    qins[i] = qin_est
    qouts[i] = qout_est
    alphas[i] = alpha_est

    #print("qin_est: {}".format(qin_est))
    #print("qout_est: {}".format(qout_est))
    #print("alpha_est {}".format(alpha_est))
   #resps_shift = shift_resps(resps_est, alpha_est)

#with open(resp_path, "w+") as f:
#    tsv_writer = csv.writer(f, delimiter='\t')
#    for i in range(len(node_names)):
#        tsv_writer.writerow([str(node_names[i]), str(resps_shift[i])])