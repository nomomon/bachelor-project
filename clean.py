# delete all files "Icon\r" in mlruns folder
# these files are created by macos and are not needed

import os

path = "./mlruns"
for root, dirs, files in os.walk(path):
    for file in files:
        if file == "Icon\r" or file.startswith(".DS_Store"):
            os.remove(os.path.join(root, file))
            print("Delete file: " + os.path.join(root, file))
    for dir in dirs:
        if dir.startswith(".tmp"):
            os.rmdir(os.path.join(root, dir))
            print("Delete dir: " + os.path.join(root, dir))