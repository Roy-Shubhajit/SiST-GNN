#!/bin/bash

# Activate the virtual environment
source temp_gnn_venv/bin/activate

echo "========================================"
echo "    Running Fixed-Split Experiments     "
echo "========================================"
FIXED_DATASETS=("bitcoin-otc" "bitcoin-alpha" "uci-message")
for ds in "${FIXED_DATASETS[@]}"; do
    echo "Running fixed-split on $ds..."
    python main.py --eval-method fixed-split --dataset "$ds" --num-epochs 400
done

echo "========================================"
echo "    Running Live-Update Experiments     "
echo "========================================"
LIVE_DATASETS=("bsi-zk" "as-733" "reddit-title" "reddit-body" "bsi-svt" "uci-message" "bitcoin-otc" "bitcoin-alpha")
for ds in "${LIVE_DATASETS[@]}"; do
    echo "Running live-update on $ds..."
    python main.py --eval-method live-update --dataset "$ds"
done

echo "========================================"
echo "      Extracting Results to CSV         "
echo "========================================"
# Short python snippet to parse all generated JSON logs and save them into a CSV
python -c '
import json
import glob
import csv

# Read all run JSONs
json_files = glob.glob("results/exp_*_roland.json")
rows = {}

# Sort files alphabetically so newer timestamp runs overwrite older duplicates
for f in sorted(json_files):
    with open(f, "r") as file:
        try:
            data = json.load(file)
            method = data.get("eval_method", "unknown")
            timestamp = data.get("timestamp", "unknown")
            lp_results = data.get("results", {}).get("lp", {})
            
            for ds, res in lp_results.items():
                if "mrr" in res:
                    mrr_val = res["mrr"]
                    rows[(ds, method)] = {
                        "Dataset": ds, 
                        "Eval_Method": method, 
                        "MRR": f"{mrr_val:.6f}",
                        "Timestamp": timestamp
                    }
        except Exception:
            pass

# Write the latest run values for each config to a summarized CSV
csv_file = "results/experiment_summary.csv"
if rows:
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "Eval_Method", "MRR", "Timestamp"])
        writer.writeheader()
        for key in sorted(rows.keys()):
            writer.writerow(rows[key])
    print(f"✅ Summary of {len(rows)} experiment configurations successfully saved to {csv_file}")
else:
    print("⚠️ No valid MRR results found to convert into CSV.")
'
