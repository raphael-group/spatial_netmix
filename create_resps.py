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

c1 = {'808', '341', '110', '147', '807', '135', '146', '364', '400', '303', '136', '401', '125', '344', '465', '141', '124', '532', '363', '343', '342', '123', '144', '109', '362', '536', '145', '509', '142', '399', '533', '105', '508', '302', '839', '143'}
c2 = {'161', '859', '191', '203', '202', '197', '151', '275', '162', '153', '278', '152', '534', '1055', '276', '196', '163', '204', '199', '871', '180', '195', '771', '1011', '164', '280', '860', '432', '870', '277', '154', '229', '198', '1059'}
c3 = {'850', '993', '991', '319', '46', '321', '322', '42', '987', '849', '118', '47', '122', '49', '48', '116', '320', '992', '835', '523', '986', '836'}
c4 = {'86', '936', '31', '33', '943', '786', '872', '933', '940', '88', '30', '87', '941', '873', '945', '942', '32', '784', '944'}

def generate_scores(names, expected, q_in, q_out, random_seed):
    np.random.seed(random_seed)
    scores = []
    for i in names:
        b = expected[i]
        
        if i in c1 or i in c2 or i in c3 or i in c4:
            score = np.random.poisson(q_in * b)
        else:
            score = np.random.poisson(q_out * b)
        scores.append(score)
    return scores

def run_trial(names, expected, q_in, q_out, random_seed):
    observed = generate_scores(names, names_to_expected, gen_qin, gen_qout, i)

    # C observed, B expected
    qin_est,qout_est,alpha_est,_= em(observed,B)
    return qin_est, qout_est, alpha_est

####################################################################
# MAIN
####################################################################

#num_trials = 10
#alphas = np.zeros(10)
#qins = np.zeros(10)
#qouts = np.zeros(10)
#gen_qin = 2
#gen_qout = 1

expected_path = '../network-anomalies/cancer/data/expected.tsv' 
observed_path = '../network-anomalies/cancer/data/fake_observed.tsv' 
resp_path = './scores_resp.tsv'


#names_to_expected = tsv_to_dict(expected_path, float_val=True)
B = tsv_to_np(expected_path, col=1) 
C = tsv_to_np(observed_path, col=1) 
names = tsv_to_list(expected_path, col=0)

qin_est,qout_est,alpha_est,resps_est = em(C,B)

print("qin_est: {}".format(qin_est))
print("qout_est: {}".format(qout_est))
print("alpha_est {}".format(alpha_est))

#print("%d trials, qin=%d, qout=%d:" % (num_trials, gen_qin, gen_qout))
#print("average qin_est={}".format(np.average(qins)))
#print("average qout_est={}".format(np.average(qouts)))
#print("average alpha_est={}".format(np.average(alphas)))

resps_shift = shift_resps(resps_est, alpha_est)
with open(resp_path, "w+") as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    for i in range(len(names)):
        tsv_writer.writerow([str(names[i]), str(resps_shift[i])])