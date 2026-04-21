python scripts/decompose_atomic_facts.py --plan qc_plan6.2 --max_workers 2 --dataset EXFEVER --end 4071 --data_type test
python scripts/generate_question.py --plan qc_plan6.2 --max_workers 1 --dataset EXFEVER --data_type test --end 4071
python scripts/get_answer.py --plan qc_plan6.2 --max_workers 1 --dataset EXFEVER --data_type test --end 4071
python scripts/verify_atomic_facts.py --plan qc_plan6.2 --max_workers 1 --dataset EXFEVER --data_type test --end 4071
python scripts/aggregate_labels.py --plan qc_plan6.2 --dataset EXFEVER --data_type test --end 4071