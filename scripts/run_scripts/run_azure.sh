python scripts/generate_question.py --plan azure_plan5.1 --max_workers 1 --dataset HOVER_subset
python scripts/get_answer.py --plan azure_plan5.1 --max_workers 1 --dataset HOVER_subset
python scripts/verify_atomic_facts.py --plan azure_plan5.1 --max_workers 1 --dataset HOVER_subset
python scripts/aggregate_labels.py --plan azure_plan5.1 --dataset HOVER_subset