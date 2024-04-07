import glob
import os


def clean_folder():
    folder_path = r''
    file_list = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
    for file_path in file_list:
        os.remove(file_path)

if __name__ == '__main__':
    clean_folder()
