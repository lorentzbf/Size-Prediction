import subprocess
import sys
filelist = "filelist"
if len(sys.argv) >= 2:
    filelist = sys.argv[1]
rf = open(filelist, "r")
filelists = rf.readlines()
wf = open("size_accu.txt", "w")
for mat1 in filelists:
    for mat2 in filelists:
        program = "./estimate_accu " + mat1.rstrip() + " " + mat2.rstrip()
        #print(program)
        sub = subprocess.Popen([program], stdout = subprocess.PIPE, shell = True)
        (out, err) = sub.communicate()
        wf.write(out.decode('utf-8'))
        print(out)
        wf.flush()


