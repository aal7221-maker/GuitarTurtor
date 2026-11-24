import sys,os
sys.path.insert(0,r"D:\sharvin1\Projects\Music\pitch-model")
import transcribe as T, numpy as np
for a in sys.argv[1:]:
    o=T.posteriorgrams(a,os.path.join(os.path.dirname(os.path.abspath(__file__)),"work",os.path.basename(a)+".npz")); print(a,o["note"].shape)
