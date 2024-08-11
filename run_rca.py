import os

for x in range(1, 4):
    if x == 1:
        end = 8
    elif x == 9:
        end = 10
    elif x == 3:
        end = 11

    for i in range(1, end + 1):
        machine = f"{x}-{i}"
        command = f"python3 /content/train.py --dataset 'SMD' --epochs 5 --group {machine}"
        os.system(command)