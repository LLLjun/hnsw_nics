import os

def main():
    dataset = "spacev"
    datasize = 1
    trans = "no_trans"

    command = "./main " + dataset + " " + str(datasize) + " " + trans
    os.system("cd build && make main && " + command)

main()