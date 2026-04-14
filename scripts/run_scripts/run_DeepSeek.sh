# python scripts/decompose_atomic_facts.py --plan qc_plan6.0 --max_workers 1 --dataset HOVER --end 4000
python scripts/generate_question.py --plan qc_plan6.0 --max_workers 1 --dataset HOVER --end 4000
python scripts/get_answer.py --plan qc_plan6.0 --max_workers 1 --dataset HOVER --end 4000
python scripts/verify_atomic_facts.py --plan qc_plan6.0 --max_workers 1 --dataset HOVER --end 4000
python scripts/aggregate_labels.py --plan qc_plan6.0 --dataset HOVER --end 4000