import pipe
for n in pipe.SONGS:
    a=pipe.run(n)
    for nm,(lo,hi) in dict(full=(0,1),first60=(0,.6),last40=(.6,1)).items():
        print(n,nm,{k:round(v,3) for k,v in pipe.score(n,a,lo,hi).items()})
