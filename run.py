import os

def main():
    dataset = "deep"
    datasize = 1
    stage = "search"

    command = "./main " + stage + " " + dataset
    os.system("cd build && make main && " + command)

main()