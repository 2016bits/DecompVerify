# merge two json files into one

import json

in_path1 = 'data/HOVER/plan2/dev_2_decomposed__0_2000.json'
in_path2 = 'data/HOVER/plan2/dev_2_decomposed__2000_4000.json'
out_path = 'data/HOVER/plan2/dev_2_decomposed_without_selfcheck.json'

with open(in_path1, 'r') as f:
    data1 = json.load(f)

with open(in_path2, 'r') as f:
    data2 = json.load(f)

merged_data = data1 + data2
with open(out_path, 'w') as f:
    json.dump(merged_data, f, indent=4)