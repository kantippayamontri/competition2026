## Diabetic Retinopathy Grading (IDRiD 2026) 糖尿病視網膜病變分級

Welcome to the Diabetic Retinopathy Grading competition!  
歡迎來到糖尿病視網膜病變分級競賽！  

This competition focuses on grading the severity of diabetic retinopathy (DR) from retinal fundus images using deep learning techniques.  
此專案旨在使用深度學習技術，從視網膜眼底影像判斷糖尿病視網膜病變的嚴重程度。  

**Table of Contents 目錄**

- [Competition Background](#competition-background-競賽背景)
- [Dataset Description](#dataset-description-資料集描述)
- [Installation Guide](#installation-guide-安裝指南)
- [Usage Instructions](#usage-instructions-使用說明)
- [Competition Structure](#competition-structure-競賽架構)

## Competition Background 競賽背景

Diabetic retinopathy is a leading cause of blindness. Grading it from fundus images is challenging because the lesions (microaneurysms, hemorrhages, exudates, new vessels) are small, and the dataset is small and imbalanced.  
This project aims to build a convolutional neural network (CNN) model capable of classifying fundus images into 5 DR severity grades.  

糖尿病視網膜病變是造成失明的主要原因之一。從眼底影像進行分級具有挑戰性，因為病灶（微血管瘤、出血、滲出物、新生血管）很小，且資料集數量少、類別不平衡。  
本競賽旨在建立一個卷積神經網絡（CNN）模型，能夠將眼底影像分類為 5 個病變嚴重程度等級。  

## Dataset Description 資料集描述

Git does not include the IDRiD (Indian Diabetic Retinopathy Image Dataset) images. On the course cluster, `train.py` reads them from this shared directory:

Git 儲存庫不包含 IDRiD（Indian Diabetic Retinopathy Image Dataset）影像。在課程叢集上，`train.py` 會從以下共用目錄讀取影像：

```
/mnt/groups/aid002-a3463-7ef7d056/competition_2026
```

Outside the cluster, set `data_dir` in `train.py` to the local directory containing both `train/` and `val/`.

若在課程叢集以外訓練，請將 `train.py` 中的 `data_dir` 設為本機 `train/` 與 `val/` 的共同上層目錄。

**Dataset Structure 資料集結構**

The dataset is already split into `train` and `val` folders. Inside each, there is one folder per DR grade, containing the fundus images of that grade.  
資料集已分為 `train` 與 `val` 資料夾，每個資料夾內以病變等級分成子資料夾，其中包含該等級的眼底影像。  

Example directory structure:  
資料夾結構範例：
```
<data_dir>/
├── train/
│   ├── anoDR/
│   │   ├── IDRiD_001.png
│   │   └── ...
│   ├── bmildDR/
│   ├── cmoderateDR/
│   ├── dsevereDR/
│   └── eproDR/
└── val/
    ├── anoDR/
    └── ... (same 5 class folders)
```

| Folder 資料夾 | Grade 等級 | Meaning 意義 | Train | Val |
|---|---|---|---|---|
| anoDR | 0 | No DR 無病變 | 107 | 27 |
| bmildDR | 1 | Mild 輕度 | 16 | 4 |
| cmoderateDR | 2 | Moderate 中度 | 108 | 28 |
| dsevereDR | 3 | Severe 重度 | 59 | 15 |
| eproDR | 4 | Proliferative 增殖性 | 39 | 10 |

- **Total Classes 總類別數**: 5
- **Total Images 總影像數**: 329 (train) + 84 (val)
- **Image Format 圖片格式**: PNG, RGB
- **Image Size 圖片大小**: about 3400x3400 pixels (resized to 512x512 in `train.py`, can be adjusted as needed)
- **Hidden Test Set 隱藏測試集**: kept private and used only for the final evaluation. 不公開，僅用於最終評分。
- **Note 注意**: The classes are imbalanced (`bmildDR` has very few images). 類別不平衡（`bmildDR` 影像很少）。

## Installation Guide 安裝指南

After cloning the repository, open a terminal in the project directory. If you use the course JupyterLab, open its **Terminal** so the kernel is registered on the course server. You need `python3` with the `venv` module.

複製專案後，請在專案根目錄開啟終端機。若使用課程的 JupyterLab，請使用其中的 **Terminal**，以便在課程伺服器上註冊核心。安裝腳本需要具備 `venv` 模組的 `python3`。

```sh
./install.sh
```

The installer creates or reuses `.venv`, installs the dependencies there, and registers the **Python (IDRiD 2026)** kernel for your account.

安裝腳本會建立或沿用 `.venv`，在其中安裝相依套件，並為你的帳號註冊 **Python (IDRiD 2026)** 核心。

## Usage Instructions 使用說明

**Training the Model 訓練模型**

If the class JupyterLab is already running under your account, open `train.ipynb` there. Select **Python (IDRiD 2026)** under **Kernel > Change Kernel**. If the kernel is missing, refresh the page.

若已用自己的帳號開啟課程的 JupyterLab，請直接在其中開啟 `train.ipynb`。在 **Kernel > Change Kernel** 中選擇 **Python (IDRiD 2026)**。若找不到該核心，請重新整理頁面。

To start JupyterLab yourself, run this command from the repository root:

若要自行啟動 JupyterLab，請在專案根目錄執行：

```sh
.venv/bin/jupyter-lab train.ipynb
```

Run the first cell to check the Python interpreter and training device. Once the [dataset](#dataset-description-資料集描述) is ready, run the second cell to start `train.py`. For the same training in a terminal, run `.venv/bin/python train.py`.

執行第一個儲存格，確認 Python 直譯器與訓練裝置。備妥[資料集](#dataset-description-資料集描述)後，執行第二個儲存格即可啟動 `train.py`。若要在終端機執行相同的訓練，請執行 `.venv/bin/python train.py`。

- The train.py script trains a simple CNN model on the `train` set and reports the accuracy on the `val` set after every epoch.
- 此 train.py 腳本將使用 `train` 資料集訓練一個簡單的 CNN 模型，並在每個 epoch 結束後輸出 `val` 資料集的準確率。
- The model from the epoch with the best `val` accuracy is saved as `idrid_model.pth` (TorchScript).
- `val` 準確率最高的 epoch 的模型會儲存為 `idrid_model.pth`（TorchScript 格式）。
- You can modify training parameters like epochs, batch size, learning rate, image size, or the model architecture directly in the train.py script if needed.
- 你可以直接在 train.py 腳本中修改訓練參數，例如 epoch 次數、批次大小、學習率、圖片大小或模型架構。


**Testing the Model 測試模型**

`evaluate.py` reads validation images from `idrid_data/val/` under the repository root. With those images in place, run:

`evaluate.py` 會從專案根目錄下的 `idrid_data/val/` 讀取驗證影像。備妥影像後，請執行：

```sh
.venv/bin/python evaluate.py idrid_model.pth training
```

- The evaluate.py script will run the trained model on the validation dataset and output the accuracy.
- evaluate.py 腳本將使用驗證集評估訓練好的模型並輸出準確率。
- The final score is computed on the hidden test set with `.venv/bin/python evaluate.py idrid_model.pth hidden`.
- 最終成績將以隱藏測試集計算（`.venv/bin/python evaluate.py idrid_model.pth hidden`）。
- If you change the image size or normalization in train.py, make the same change to the `transform` in evaluate.py.
- 若你修改了 train.py 中的圖片大小或正規化設定，請同步修改 evaluate.py 中的 `transform`。


## Competition Structure 競賽架構

```
competition2026/
├── install.sh                              # Environment and kernel setup 環境與核心安裝腳本
├── train.ipynb                             # Training notebook 訓練筆記本
├── train.py                                # Training script 訓練腳本
├── evaluate.py                             # Testing/Evaluation script 測試/評估腳本
├── class_to_idx.txt                        # 每個病變等級類別的對應索引
└── README.md                               # README file 說明文件
```

- **train.py**: Script to train the CNN model. 訓練 CNN 模型的腳本
- **evaluate.py**: Script to test the trained model. 測試訓練模型的腳本
- **class_to_idx.txt**: Mapping from class folder name to class index. 類別資料夾名稱與索引的對應
