"""Pick 2 random kept pages of an item for by-eye spot-check: prints page png path + read tab text.
usage: python -I spot.py ID      record:  python -I spot.py ID --record "p3:ok" "p7:1 wrong of 18 (6 read as 8)" """
import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__))
vid = sys.argv[1]; d = os.path.join(HERE, "debug", vid)
if "--record" in sys.argv:
    res = sys.argv[sys.argv.index("--record") + 1:]
    json.dump({"pages": res, "by": "eye (page png vs tab.txt)"}, open(os.path.join(d, "spotcheck.json"), "w"), indent=1)
    sys.exit()
lab = json.load(open(os.path.join(HERE, "labels", vid + ".json")))
kept = sorted({int(n["sec"].split("||")[0].split(" p")[1]) - 1 for n in lab["notes"]})
random.seed(vid)
lines = open(os.path.join(d, "tab.txt"), encoding="utf-8").read().splitlines()
for k in sorted(random.sample(kept, min(2, len(kept)))):
    print(os.path.join(d, f"page{k:02d}.png")); print(lines[k][:700])
