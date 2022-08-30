import os
cmd = "./go sift-euclidean-128 sift-euclidean-128 l2 %d 500 24 %d %d 0"
for args in [8, 16, 24, 36, 48, 64, 96]:
    for efs in [10, 20, 40, 80, 120, 200, 400]:
        os.system(cmd % (args, efs, efs))
        if efs - 20 > 0:
            os.system(cmd % (args, efs, 20))
        if efs - 50 > 0:
            os.system(cmd % (args, efs, 50))
