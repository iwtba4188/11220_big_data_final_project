# 11220_big_data_final_project

## Introduction
這是 11220KHCT100700 的期末**分組**作業（另有三位組員），用以訓練能夠分類真實圖片與 AI 圖片的模型，資料集 [來自 Kaggle](https://www.kaggle.com/datasets/sunnykakar/shoes-dataset-real-and-ai-generated-images)。

## Description
- 模型前半部使用 `keras.applications.VGG16()` 預訓練模型，並關閉訓練此部分。在 `VGG16` 輸出後額外加入 `Flatten()` 和兩層 `Dense()` 以供訓練與分類。
- 報告主要著重觀察哪些圖片比較容易被模型分辨成 真實圖片 / AI 圖片，並使用原資料集和自行拍攝之圖片，簡單更動圖片背景和彩度，相互比較這些更動對預測結果的影響。

## Environment
### Windows
```ps
py -m pip install -r requirements.txt
```
如果需使用 `keras.utils.plot_model()`，請額外安裝 [`graphviz`](https://graphviz.org/download/)，並執行：
```ps
py -m pip install -r requirements-plot-model.txt
```

## Note
此份報告內容大量使用 `ChatGPT` 輔助撰寫程式碼，並大量參考 TensorFlow 官方文件教學程式碼，僅自行更動部分參數、結構與資訊輸出供參考與撰寫報告使用。

## Acknowledgements
[Shoes Dataset: Real and AI-Generated Images](https://github.com/sunkakar/dataset-shoes-ai-generated), by [@Sundeep Kakar](https://github.com/sunkakar/) 2024