import sys, os
sys.path.insert(0, os.path.expanduser("~/Portfolio/Smart_Traffic_System/datasets"))
import train_fasterrcnn_v3 as t
t.train_fold(int(sys.argv[1]))
