import os

def main():
    stage = "search"
    dataset = "sift"
    datasize = 1

    command = "./main " + stage + " " + dataset + " " + str(datasize)
    os.system("cd build && make main && " + command)

main()