import os
import shutil

import gdown

def download():
    gdown.download("https://drive.google.com/uc?export=download&id=1atmaVFyg67MddMevAHyH_cVTxqdHLLT0")
    os.makedirs("./src/best_model_weights", exist_ok=True)
    shutil.move("model_best_ss.pth", "./src/best_model_weights/model_best.pth")


if __name__ == "__main__":
    download()