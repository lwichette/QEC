import os
import sys
import shutil


def delete_directories(path, min, max):
    folders = os.listdir(path)

    for f in folders:
        seed = int(f.split("_")[1])

        if seed >= min and seed <= max:
            shutil.rmtree(path + "/" + f)


def main():
    path = sys.argv[1]
    min = int(sys.argv[2])
    max = int(sys.argv[3])

    delete_directories(path, min, max)


if __name__ == "__main__":
    main()
