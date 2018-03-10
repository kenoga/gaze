import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('path', help='a path of json file that you want to analyze')
args = parser.parse_args()

with open(args.path, 'r') as fr:
    results = json.load(fr)

total_count = 0
failed_count = 0
for k, v in results['results'].items():
    total_count += 1
    if len(v) == 0:
        failed_count += 1

recall = (total_count - failed_count) / float(total_count)
print('recall: %.4f' % recall)