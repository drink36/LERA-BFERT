# LERA-BFERT: Live Emotional Resonance Application Based on Facial Expression Recognition Technology
> National Taiwan University of Science and Technology & Department of Computer Science and Information Engineering<br>

[English](./README.md) | [中文](./README.zh-TW.md)
## ✨ Overview
本專案旨在透過在觀眾間創造**情緒共鳴**來提升觀影體驗。同時，直播主也能根據觀眾的反應來調整內容。

直播主可以根據觀眾的反應，即時調整直播內容。我們採用了動態臉部情緒辨識 (Dynamic Facial Emotion Recognition, DFER) 的方法，來即時且有效地捕捉觀眾的臉部情緒。

我們結合了微表情辨識模型來改進原有的動態臉部情緒辨識模型，以找出最佳效能的模型。初步結果顯示，我們的模型能**有效**且**即時**地偵測觀眾的臉部情緒。

接著，我們透過 Google Sheets API 匯總所有觀眾的情緒資訊，並將這些資訊處理成易於理解的圖表。最終，我們讓觀眾能夠即時看到他人的情緒反應，藉此提升觀影體驗。

## 🚀 Main Results
### ✨ Interface & Result 
![Interface](pic/Interface.png)

![Result](pic/sample_result.png)


您可以在這裡找到我們的[海報](pic/Project_Poster_zh.png)，並在這裡閱讀我們的[報告](pic/Project_Paper_zh.pdf)。

模型來源：
本專案所開發的核心動態面部情緒辨識模型已於 [MAE-DFER-CA](https://github.com/drink36/MAE-DFER-CA) 開源。
如有興趣了解模型的訓練程式碼與效能測試，請參考該模型專案。
## 🔨 Installation
請依照[requirements.txt](requirements.txt) 中的指示進行安裝。

## ➡️ Preparation
請建立您自己的 Google 帳號，並根據此[文件](https://developers.google.com/sheets/api/guides/concepts)的說明填寫 **.json** 檔案。

## 📍 Model
請從此[連結](https://drive.google.com/file/d/1AySyaGGic-ZrdJp3p3tDpPANE6spaOhx/view?usp=drive_link)下載預訓練模型，並將其放入[此資料夾](model)中。

## ☎️ Contact 
若有任何問題，請隨時透過 `ooo910809@gmail.com` 與我聯繫。

## 👍 Acknowledgements
本專案的模型基於 [VideoMAE](https://github.com/MCG-NJU/VideoMAE), [MAE-DFER](https://github.com/sunlicai/MAE-DFER) 和[MMNET](https://github.com/muse1998/MMNet)。
感謝他們出色的開源專案。