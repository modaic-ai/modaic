import json

def check_file():
    ids = set()
    total_lines = 0
    with open("/Users/tytodd/Desktop/Modaic/code/core/modaic/Job_Output_8e2ab.jsonl") as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                custom_id = data.get("custom_id")
                if custom_id:
                    ids.add(custom_id)
            except:
                pass
                
    print(f"Total lines: {total_lines}")
    print(f"Unique custom_ids: {len(ids)}")
    
    missing = []
    for i in range(1500):
        if f"request-{i}" not in ids:
            missing.append(i)
            
    print(f"Missing request indices (out of 1500): {len(missing)}")
    if missing:
        print(f"First 10 missing: {missing[:10]}")

if __name__ == "__main__":
    check_file()
