import os

def main():
    stage = "search"
    dataset = "deep"
    datasize = 10

    command = "./main " + stage + " " + dataset + " " + str(datasize)
    os.system("cd build && make main && " + command)

main()