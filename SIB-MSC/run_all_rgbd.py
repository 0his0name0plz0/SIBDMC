
import subprocess
# gamma MLP；alpha DIS；beta KL；rec
cmds = [
        #'python -u main.py --ft True --batch_size 500 --save_name rgbd_checkpoint --cmi 300 --selfExp 0.3 > ./finetune00_cmi300self0.3_test1.log',
        'python -u main.py --save_name rgbd_cmi300_Newtest --cmi 300 > ./pretrain00_cmi300.log',
        'python -u main.py --ft True --batch_size 500 --save_name rgbd_cmi300_Newtest --cmi 300 --selfExp 0.3 > ./finetune00_cmi300self0.3_test2.log',

]

for cmd in cmds:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in iter(p.stdout.readline, b''):
        msg = line.strip().decode('gbk')
        print(msg)

