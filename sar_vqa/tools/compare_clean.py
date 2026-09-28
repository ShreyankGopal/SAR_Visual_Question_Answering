"""compare_clean.py: check that two clean JSONL files hold the same boxes.

  python tools/compare_clean.py <old clean.jsonl> <new clean.jsonl>
"""
import json
import os
import sys

stem = lambda r: os.path.splitext(os.path.basename(r["image"]))[0]
old = {stem(r): r for r in map(json.loads, open(sys.argv[1]))}
new = {stem(r): r for r in map(json.loads, open(sys.argv[2]))}
diff = [s for s in sorted(set(old) & set(new))
        if [(o["rbox"], o["hbox"], o.get("rbox_src")) for o in old[s]["objects"]]
        != [(o["rbox"], o["hbox"], o.get("rbox_src")) for o in new[s]["objects"]]]
print(f"images: old {len(old)}, new {len(new)}, boxes differ in {len(diff)}, "
      f"missing {len(set(old) - set(new))}, extra {len(set(new) - set(old))}")
print("IDENTICAL" if not diff and set(old) == set(new) else f"DIFFERENT, e.g. {diff[:5]}")
