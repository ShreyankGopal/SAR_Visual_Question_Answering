"""compare_qa.py: check that two builds produced the same questions and answers.

  python tools/compare_qa.py <old data dir> <new data dir>
Records are matched by id; only question and answer text are compared.
"""
import json
import sys


def load(d, split):
    try:
        return {r["id"]: r for r in map(json.loads, open(f"{d}/{split}.jsonl"))}
    except FileNotFoundError:
        return {}


old_dir, new_dir = sys.argv[1], sys.argv[2]
ok = True
for split in ("train", "val", "test"):
    old, new = load(old_dir, split), load(new_dir, split)
    old_ids, new_ids = set(old), set(new)
    diff = [i for i in sorted(old_ids & new_ids)
            if [c["value"] for c in old[i]["conversations"]] != [c["value"] for c in new[i]["conversations"]]
            or old[i]["category"] != new[i]["category"]]
    only_old, only_new = sorted(old_ids - new_ids), sorted(new_ids - old_ids)
    print(f"{split}: old {len(old)}, new {len(new)}, same id {len(old_ids & new_ids)}, "
          f"text differs {len(diff)}, only in old {len(only_old)}, only in new {len(only_new)}")
    for i in (diff + only_old + only_new)[:5]:
        print("   e.g.", i)
    ok &= not (diff or only_old or only_new)
print("IDENTICAL" if ok else "DIFFERENT")
