import csv

file_path = '/home/yl/workspace/srtj_1/logs/gpt-3.5-turbo-1106_harmbench200/advsub/warmup_gpt-3.5-turbo-1106_20260204_090914.csv'

goals = {}

try:
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row: continue
            
            goal = row.get('goal', '').strip()
            if not goal: continue
            
            success_str = row.get('success', 'False').strip().lower()
            success = success_str == 'true'
            
            if goal not in goals:
                goals[goal] = {'attempts': 0, 'successes': 0}
            
            goals[goal]['attempts'] += 1
            if success:
                goals[goal]['successes'] += 1

    print(f"Total Unique Goals: {len(goals)}")
    print("-" * 80)
    print(f"{'Goal':<60} | {'Solved?':<10}")
    print("-" * 80)
    
    total_solved = 0
    for goal, stats in goals.items():
        solved = stats['successes'] > 0
        if solved: total_solved += 1
        status = "YES" if solved else "NO"
        print(f"{goal[:57]+'...' if len(goal)>57 else goal:<60} | {status}")

    if len(goals) > 0:
        print("-" * 80)
        print(f"Overall ASR (Goals Solved/Total): {total_solved}/{len(goals)} = {total_solved/len(goals)*100:.2f}%")
        
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error: {e}")
