import json
import argparse

def main(args):
    in_path = args.in_path.replace('[DATA]', args.dataset).replace('[HOP_SOURCE]', args.hop_source)
    with open(in_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    new_data = []
    for data in dataset:
        new_item = {
            "id": data['id'],
            "claim": data['claim'],
            "label": data['label'].lower(),
            "num_hops": data['num_hops'],
            "evidence": "\n".join(data['retrieved_evidence'])
        }
        new_data.append(new_item)
    
    out_path = args.out_path.replace('[DATA]', args.dataset)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./data/[DATA]/NodeFC_retrieved/nodefc_[HOP_SOURCE]_dev_verifying_data.json')
    parser.add_argument('--out_path', type=str, default='./data/[DATA]/NodeFC_retrieved/dev.json')
    parser.add_argument('--dataset', type=str, default='HOVER')
    parser.add_argument('--hop_source', type=str, default='heuristic')
    
    args = parser.parse_args()
    main(args)
