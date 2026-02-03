
import sys
import os
import csv
import json
from pathlib import Path

# Add src to path
current_dir = Path(__file__).resolve().parent
src_dir = current_dir / "src"
sys.path.append(str(src_dir))

try:
    from harvester import ComparativeRuleHarvester
    from llm_client import LLMClient
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

LOG_FILE = "logs/gpt-3.5-turbo-1106_harmbench200/advsub/2.1/lifelong_gpt-3.5-turbo-1106_20260201_080236_run1.csv"

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        return

    # 1. Initialize Client
    try:
        # Using a strong model for the harvester to ensure quality
        client = LLMClient(model_name="gpt-4o") 
    except Exception as e:
        print(f"Failed to init LLMClient: {e}")
        return

    harvester = ComparativeRuleHarvester(client)

    # 2. Read CSV and Group by Goal/Behavior
    experiments = {} # behavior_id -> { goal: str, tags: str, attempts: [ {attempt_rows} ] }
    
    print("Reading log file...")
    with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            b_id = row.get("behavior_id")
            if not b_id:
                continue
            
            if b_id not in experiments:
                experiments[b_id] = {
                    "goal": row.get("goal"),
                    "attempts": []
                }
            
            # Parse success
            success_raw = row.get("success", "False").lower()
            success = success_raw == "true"
            
            experiments[b_id]["attempts"].append({
                "prompt": row.get("final_prompt"),
                "response": row.get("target_response"),
                "success": success,
                "attempt_id": int(row.get("attempt", 0))
            })

    # 3. Process a few effective examples
    count = 0
    max_count = 3
    
    print(f"Found {len(experiments)} distinct behaviors.")
    
    for b_id, data in experiments.items():
        if count >= max_count:
            break
            
        attempts = sorted(data["attempts"], key=lambda x: x["attempt_id"])
        
        # Find first success
        success_idx = -1
        for i, att in enumerate(attempts):
            if att["success"]:
                success_idx = i
                break
        
        # We prefer cases that had some failures before success to test the "Comparative" aspect
        if success_idx > 0: # Found a success and it wasn't the first try
            goal = data["goal"]
            success_item = attempts[success_idx]
            failed_items = attempts[:success_idx]
            
            print(f"\n\n=== Harvesting for Behavior: {b_id} ===")
            print(f"Goal: {goal}")
            print(f"Failed Attempts Count: {len(failed_items)}")
            
            # Format failed attempts for harvester
            history_input = []
            for f_att in failed_items:
                history_input.append({
                    "failed_prompt": f_att["prompt"],
                    "target_response": f_att["response"]
                })
            
            successful_prompt = success_item["prompt"]
            
            print("Invoking Harvester...")
            try:
                result = harvester.harvest(
                    goal_prompt=goal,
                    successful_prompt=successful_prompt,
                    history_attempts=history_input
                )
                
                if result:
                    print("--- Harvested Rule ---")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    count += 1
                else:
                    print("--- Harvester returned None ---")
            except Exception as e:
                print(f"Error during harvest: {e}")

    if count == 0:
        print("\nNo suitable examples found (cases with failures followed by success).")

if __name__ == "__main__":
    main()
