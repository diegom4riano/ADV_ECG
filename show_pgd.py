import os
import numpy as np

def show_pgd_info(folder_path="adv_exp/pgd"):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            data = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            print(f"\nFile: {file_name}")
            print(data)

if __name__ == "__main__":
    show_pgd_info()