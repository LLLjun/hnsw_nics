from distutils import command
import os

def main():
    dataset = "turing"

    stage = "build"

    command = "./main " + stage + " " + dataset

    os.system("cd build && make main && " + command)


    stage = "search"

    command = "./main " + stage + " " + dataset

    os.system("cd build && make main && " + command)

main()