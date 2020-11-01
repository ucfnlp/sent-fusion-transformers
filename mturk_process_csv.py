import csv
import glob
import os

csv_dir = os.path.join('mturk','raw_output')
processed_file = os.path.join('mturk','processed','valid_pair_indices.txt')
valid_fusions_file = os.path.join('mturk','processed','valid_fusions.csv')
all_fusions_file = os.path.join('mturk','all_pairs_test_and_val.csv')
if not os.path.exists(os.path.dirname(processed_file)):
    os.makedirs(os.path.dirname(processed_file))

workers_to_reject = '''
AJ1Q54P37KT5R
AGA1PIIWM8I8D
A5J3EJX1ADIS
A4V3M708VAAT4
A3ODZJPMUOLZ4K
A3JJ9WEV515N1K
A3AW887GI0NLKF
A30CKD84V82EH2
A2CSV75E3JT58Y
A1UYS657H2WSZW
A1UAKJN1VI68MQ
A1E6XN5C8S0X7W
'''.strip().split('\n')

print(workers_to_reject)

def quality_check_workers(all_lines):
    workers = {}
    for line in all_lines:
        pair_idx = int(line['Input.pair_idx'])
        if pair_idx >= 0:
            continue
        workerid = line['WorkerId']
        if workerid not in workers:
            workers[workerid] = [0,0]       # this list is [NUM_CORRECT, NUM_INCORRECT]
        answer = line['Answer.Faithfulness_0']
        if answer == 'ONE':
            workers[workerid][0] += 1
        else:
            workers[workerid][1] += 1
    workers_failed_quality_check = []
    for workerid, (num_correct, num_incorrect) in workers.items():
        if num_correct < num_incorrect:
            workers_failed_quality_check.append(workerid)
    return workers_failed_quality_check

def both_vs_other_check(all_lines):
    workers = {}
    for line in all_lines:
        pair_idx = int(line['Input.pair_idx'])
        if pair_idx < 0:
            continue
        workerid = line['WorkerId']
        if workerid not in workers:
            workers[workerid] = [0,0]       # this list is [num_both, num_other]
        answer = line['Answer.Faithfulness_0']
        if answer == 'BOTH':
            workers[workerid][0] += 1
        else:
            workers[workerid][1] += 1
        workers_voted_one_or_neither_often = []
    for workerid, (num_both, num_other) in workers.items():
        if num_both < num_other and num_both + num_other >= 5:
            workers_voted_one_or_neither_often.append(workerid)
    return workers_voted_one_or_neither_often

csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

all_lines = []
for csv_file in csv_files:
    with open(csv_file, 'r' ) as f:
        reader = csv.DictReader(f)
        for line in reader:
            all_lines.append(line)

hits = {}
for line in all_lines:
    hitid = line['HITId']
    if hitid not in hits:
        hits[hitid] = []
    hits[hitid].append(line)

workers_failed_quality_check = quality_check_workers(all_lines)
print(workers_failed_quality_check)

workers_voted_one_or_neither_often = both_vs_other_check(all_lines)
print(workers_voted_one_or_neither_often)

valid_fusions = []
invalid_fusions = []
rejected = 0
failed_quality = 0
voted_often = 0
total = 0
for hitid, assigns in hits.items():
    num_both = 0
    num_other = 0
    for assign in assigns:
        total += 1
        if assign['WorkerId'] in workers_to_reject:
            rejected += 1
            continue
        if assign['WorkerId'] in workers_failed_quality_check:
            failed_quality += 1
            continue
        if assign['WorkerId'] in workers_voted_one_or_neither_often:
            voted_often += 1
            continue
        if assign['Answer.Faithfulness_0'] == 'BOTH':
            num_both += 1
        else:
            num_other += 1
    if int(assigns[0]['Input.pair_idx']) < 0:   # if it's a quality check
        a=0
    elif num_both > num_other:      # if workers who voted for BOTH outnumbers the workers who voted for ONE or NEITHER
    # elif num_both >= 2:      # if workers who voted for BOTH outnumbers the workers who voted for ONE or NEITHER
        valid_fusions.append(int(assigns[0]['Input.pair_idx']))
    else:
        invalid_fusions.append(int(assigns[0]['Input.pair_idx']))
        print(assigns[0]['Input.article_content'])
        print("")
        print(assigns[0]['Input.summary'])
        print('----------------------------------------------------')
        # input("")

valid_fusions = sorted(valid_fusions)
with open(processed_file, 'w') as f:
    f.write('\n'.join([str(idx) for idx in valid_fusions]))

print('total assignments', total)
print('rejected assignments', rejected)
print('failed_quality assignments', failed_quality)
print('voted_neither_or_one_too_often assignments', voted_often)
print('')

print('total fusions', len(hits))
print('valid fusions', len(valid_fusions))
print('invalid fusions', len(invalid_fusions))

valid_csv_lines = []
with open(all_fusions_file, 'r' ) as f:
    for line_idx, line in enumerate(f.readlines()):
        if line_idx == 0:
            valid_csv_lines.append(line)
            continue
        if int(line.split(',')[0].replace('"', '')) in valid_fusions:
            valid_csv_lines.append(line)
print(len(valid_csv_lines))
with open(valid_fusions_file, 'w') as f:
    f.write(''.join(valid_csv_lines))