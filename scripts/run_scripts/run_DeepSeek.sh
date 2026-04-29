# python scripts/decompose_atomic_facts.py --plan qc_plan6.4 --max_workers 2 --dataset HOVER_NFC_plan5.2 --end 200 --data_type dev
# python scripts/generate_question.py --plan qc_plan6.4 --max_workers 1 --dataset HOVER_NFC_plan5.2 --data_type dev --end 200
# python scripts/get_answer.py --plan qc_plan6.4 --max_workers 1 --dataset HOVER_NFC_plan5.2 --data_type dev --end 200
python scripts/verify_atomic_facts.py --plan qc_plan6.4 --max_workers 1 --dataset HOVER_NFC_plan5.2 --data_type dev --end 200
python scripts/aggregate_labels.py --plan qc_plan6.4 --dataset HOVER_NFC_plan5.2 --data_type dev --end 200
