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

def tsv_to_dict(filename, key=0, val=1, float_val=False):
    
    the_dict = {}
    
    with open(filename, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
    
        for row in reader:
            if len(row) > 1: 
                if float_val:
                    the_dict[row[key]] = float(row[val])
                else:
                    the_dict[row[key]] = row[val]
            
    return the_dict

def tsv_to_list(filename, col=0):
    
    the_list = []
    
    with open(filename, 'r') as fp:
        reader = csv.reader(fp, delimiter='\t')
    
        for row in reader:
            if len(row[col]) > 0:
                the_list.append(row[col])
            
    return the_list


####################################################################
# SCORE GENERATION
####################################################################

clusters = clusters = {'48', '523', '119', '826', '194', '107', '188', '298', '500', '84', '898', '765', '270', '901', '902', '121', '924', '750', '272', '49', '835', '945', '458', '977', '897', '357', '83', '33', '226', '120', '85', '513', '47', '271', '46', '336', '358', '825', '143', '196', '195', '524', '815', '899', '784', '149', '31', '926', '846', '159', '337', '192', '834', '273', '177', '150', '338', '148', '140', '426', '151', '102', '275', '42', '314', '32', '200', '900', '896', '845', '106', '763', '141', '812', '115', '813', '920', '144', '522', '122', '201', '30', '133', '317', '356', '393', '847', '894', '158', '921', '199', '925', '785', '132', '138', '521', '339', '113', '315', '142', '395', '844', '297', '139', '161', '499', '975', '193', '316', '394', '160'}
c1 = {'357', '785', '122', '393', '102', '358', '144', '107', '138', '120', '141', '394', '142', '521', '121', '132', '336', '524', '784', '298', '395', '815', '458', '338', '143', '297', '106', '339', '500', '356', '140', '139', '522', '337', '499', '133'}
c2 = {'835', '151', '201', '272', '273', '199', '523', '192', '834', '160', '158', '845', '177', '844', '226', '200', '275', '194', '975', '149', '161', '150', '270', '977', '148', '945', '196', '750', '426', '193', '159', '195', '271', '188'}
c3 = {'317', '115', '315', '49', '48', '47', '812', '513', '925', '314', '316', '113', '119', '920', '926', '813', '825', '921', '826', '42', '46', '924'}
c4 = {'901', '33', '30', '894', '847', '85', '900', '897', '763', '846', '898', '83', '31', '899', '896', '32', '902', '84', '765'}

def generate_scores(names, expected, q_in, q_out, random_seed):
    np.random.seed(random_seed)
    scores = []
    for i in names:
        b = expected[i]
    
        if i in c3:
            score = np.random.poisson(q_in * b)
        else:
            score = np.random.poisson(q_out * b)
        scores.append(score)
    return scores

def run_trial(names, expected, q_in, q_out, random_seed):
    observed = generate_scores(names, names_to_expected, gen_qin, gen_qout, i)

    # C observed, B expected
    qin_est, qout_est, alpha_est, _, _, _ = em(observed,B)
    return qin_est, qout_est, alpha_est

####################################################################
# MAIN
####################################################################

expected_path = '../network-anomalies/cancer/data/expected.tsv'
B = tsv_to_np(expected_path, col=1) 
names = tsv_to_list(expected_path, col=0)

###########
# Standard
###########

observed_path = '../network-anomalies/cancer/data/observed3.tsv' 
resp_path = '../network-anomalies/cancer/data/cancer_resps3.tsv'

C = tsv_to_np(observed_path, col=1) 


qin_est,qout_est,alpha_est,resps_est,nums_est,denoms_est = em(C,B)

print("qin_est: {}".format(qin_est))
print("qout_est: {}".format(qout_est))
print("alpha_est: {}".format(alpha_est))


resps_shift = shift_resps(resps_est, alpha_est)
with open(resp_path, "w+") as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    for i in range(len(names)):
        tsv_writer.writerow([str(names[i]), str(resps_shift[i])])

###########
# Trials
###########

'''

num_trials = 10
alphas = np.zeros(num_trials)
qins = np.zeros(num_trials)
qouts = np.zeros(num_trials)
gen_qin = 3
gen_qout = 1
names_to_expected = tsv_to_dict(expected_path, float_val=True)

for i in range(num_trials):
    qin_est, qout_est, alpha_est = run_trial(names, names_to_expected, gen_qin, gen_qout, i)
    alphas[i] = alpha_est
    qins[i] = qin_est
    qouts[i] = qout_est
    print("trial %d done" % i)

print("%d trials, qin=%d, qout=%d:" % (num_trials, gen_qin, gen_qout))
print("average qin_est={}".format(np.average(qins)))
print("average qout_est={}".format(np.average(qouts)))
print("average alpha_est={}".format(np.average(alphas)))



'''


