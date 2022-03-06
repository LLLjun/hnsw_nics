import os

def main():
    dataset = "deep"
    datasize = 1
    stage = "both"

    if stage == "both":
        command = "./main build " + dataset + " && ./main search " + dataset
    else:
        command = "./main " + stage + " " + dataset

    os.system("cd build && make main && " + command)

main()