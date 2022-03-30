from distutils import command
import os

def main():
    dataset = "deep"


    stage = "search"

    command = "./main " + stage + " " + dataset

    os.system("cd build && make main && " + command)

main()