**Diabetic Retinopathy Grading (IDRiD 2026) 糖尿病視網膜病變分級**

Welcome to the Diabetic Retinopathy Grading competition!  
歡迎來到糖尿病視網膜病變分級競賽！  

This competition focuses on grading the severity of diabetic retinopathy (DR) from retinal fundus images using deep learning techniques.  
此專案旨在使用深度學習技術，從視網膜眼底影像判斷糖尿病視網膜病變的嚴重程度。  

**Table of Contents 目錄**

- [Competition Background](#Competition-background)
- [Dataset Description](#dataset-description)
- [Usage Instructions](#usage-instructions)
- [Competition Structure](#competition-structure)
- [Contact Information](#contact-information)

**Competition Background 競賽背景**

Diabetic retinopathy is a leading cause of blindness. Grading it from fundus images is challenging because the lesions (microaneurysms, hemorrhages, exudates, new vessels) are small, and the dataset is small and imbalanced.  
This project aims to build a convolutional neural network (CNN) model capable of classifying fundus images into 5 DR severity grades.  

糖尿病視網膜病變是造成失明的主要原因之一。從眼底影像進行分級具有挑戰性，因為病灶（微血管瘤、出血、滲出物、新生血管）很小，且資料集數量少、類別不平衡。  
本競賽旨在建立一個卷積神經網絡（CNN）模型，能夠將眼底影像分類為 5 個病變嚴重程度等級。  

**Dataset Description 資料集描述**

The dataset is based on the IDRiD (Indian Diabetic Retinopathy Image Dataset) and is available in this repository under the directory:  
資料集基於 IDRiD（Indian Diabetic Retinopathy Image Dataset），已包含在本專案的以下路徑中：  
```
idrid_resplit_seed2026/idrid_resplit_seed2026/
```
**Dataset Structure 資料集結構**

The dataset is already split into `train` and `val` folders. Inside each, there is one folder per DR grade, containing the fundus images of that grade.  
資料集已分為 `train` 與 `val` 資料夾，每個資料夾內以病變等級分成子資料夾，其中包含該等級的眼底影像。  

Example directory structure:  
資料夾結構範例：
```
idrid_resplit_seed2026/idrid_resplit_seed2026/
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
| bmildDR | 1 | Mild 輕度 | 15 | 4 |
| cmoderateDR | 2 | Moderate 中度 | 106 | 26 |
| dsevereDR | 3 | Severe 重度 | 58 | 15 |
| eproDR | 4 | Proliferative 增殖性 | 39 | 10 |

- **Total Classes 總類別數**: 5
- **Total Images 總影像數**: 325 (train) + 82 (val)
- **Image Format 圖片格式**: PNG, RGB
- **Image Size 圖片大小**: about 3400x3400 pixels (resized to 224x224 in `train.py`, can be adjusted as needed)
- **Hidden Test Set 隱藏測試集**: 103 images, used only for the final evaluation. 僅用於最終評分。
- **Note 注意**: The classes are imbalanced (`bmildDR` has very few images). 類別不平衡（`bmildDR` 影像很少）。

**Installation Guide 安裝指南**


1. **Clone the Repository**  

```
git clone <repository-url>
cd competition2026
```

2. **Ensure Required Libraries Are Installed 確認已安裝所需的函式庫**

Make sure you have the necessary libraries installed (e.g., PyTorch, torchvision, tqdm, Pillow). It's assumed you have these set up in your environment.  
請確認環境中已安裝所需函式庫（如 PyTorch、torchvision、tqdm、Pillow）。  
假設您已經在環境中完成安裝設定。

3. **Set Up the Dataset 設置資料集**

The dataset should already be in place within the repository. If not, ensure that the dataset directory is correctly placed as per the structure mentioned above, and update `data_dir` in `train.py` if needed.  
資料集應該已包含於此倉庫中。  
如果沒有，請依照上述結構將資料集放置於正確位置，並視需要修改 `train.py` 中的 `data_dir`。


**Usage Instructions使用說明**

**Training the Model 訓練模型**

To train the model, simply run:  
執行以下指令以訓練模型：  

` python train.py `

- The train.py script trains the CNN model on the `train` set and reports the accuracy on the `val` set after every epoch.
- 此 train.py 腳本將使用 `train` 資料集訓練 CNN 模型，並在每個 epoch 結束後輸出 `val` 資料集的準確率。
- The trained model is saved as `idrid_model.pth` (TorchScript).
- 訓練好的模型會儲存為 `idrid_model.pth`（TorchScript 格式）。
- You can modify training parameters like epochs, batch size, learning rate, image size, or the model architecture directly in the train.py script if needed.
- 你可以直接在 train.py 腳本中修改訓練參數，例如 epoch 次數、批次大小、學習率、圖片大小或模型架構。


**Testing the Model 測試模型**

To evaluate the model's performance:  
執行以下指令來評估模型性能：  

` python evaluate.py idrid_model.pth training `

- The evaluate.py script will run the trained model on the validation dataset and output the accuracy.
- evaluate.py 腳本將使用驗證集評估訓練好的模型並輸出準確率。
- The final score is computed on the hidden test set with ` python evaluate.py idrid_model.pth hidden `.
- 最終成績將以隱藏測試集計算（` python evaluate.py idrid_model.pth hidden `）。
- If you change the image size or normalization in train.py, make the same change to the `transform` in evaluate.py.
- 若你修改了 train.py 中的圖片大小或正規化設定，請同步修改 evaluate.py 中的 `transform`。


**Competition Structure 競賽架構**

```
competition2026/
├── train.py                                # Training script 訓練腳本
├── evaluate.py                             # Testing/Evaluation script 測試/評估腳本
├── class_to_idx.txt                        # 每個病變等級類別的對應索引
├── idrid_resplit_seed2026/
│   └── idrid_resplit_seed2026/             # Dataset directory 資料集資料夾
│       ├── train/
│       │   ├── anoDR/
│       │   ├── bmildDR/
│       │   └── ... (5 class folders)
│       └── val/
│           └── ... (5 class folders)
└── README.md                               # README file 說明文件
```

- **train.py**: Script to train the CNN model. 訓練 CNN 模型的腳本
- **evaluate.py**: Script to test the trained model. 測試訓練模型的腳本
- **class_to_idx.txt**: Mapping from class folder name to class index. 類別資料夾名稱與索引的對應
- **idrid_resplit_seed2026/**: Directory containing the dataset. 包含資料集的目錄
