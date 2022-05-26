import os

def main():
    dataset = "deep"
    datasize = 100
    trans = "trans"

    command = "./main " + dataset + " " + str(datasize) + " " + trans
    os.system("cd build && make main && " + command)

main()