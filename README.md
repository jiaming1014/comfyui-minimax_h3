# comfyui-minimax_h3

在雲端免費 GPU 上一次跑起 **ComfyUI + MiniMax-H3**（原生影片＋音訊生成），
用一份 notebook 完成：環境安裝、權重下載與校驗、官方工作流改寫、啟動服務、對外臨時連結。

- `kaggle-l4-x1-comfyui-minimax-h3.ipynb` → **Google Colab · L4 24GB 單卡**（權重存 Google 雲端硬碟，重開不用重下）
- `kaggle-t4-x2-comfyui-minimax-h3.ipynb` → **Kaggle · T4 x2**（權重放 `/tmp`，每次重開要重下）

> 檔名提醒：`kaggle-l4-x1-*` 雖然叫 kaggle，內容其實是為 **Colab L4** 寫的（會掛載 `/content/drive`、使用 `google.colab`）。

---

## 用途

兩個 notebook 都是「一鍵把 MiniMax-H3 跑起來」的自架服務，跑完會印出一個
`https://xxxx.trycloudflare.com` 臨時網址，用瀏覽器打開就是 ComfyUI，
再從 `Workflows` 載入腳本自動生成的 `MiniMax_H3_*` 工作流即可出片。

自動生成的工作流（都不需要額外下載模型）：

| 工作流 | 做什麼 | 說明 |
| --- | --- | --- |
| `T2V` | 文字 → 影片＋音訊 | 主力工作流，官方模板改寫 |
| `T2I` | 文字 → 單張圖 | 重用同一套權重，只採最短 5 幀再取第一幀存 PNG |
| `IT2V` | 單圖參考 → 影片 | 改用 ref2va 節點：參考圖只提供身份／外觀，**不當首幀** |
| `IV2V` | 原片 + 人臉圖 → 換臉 | 保留原片動作、運鏡、嘴型、音軌，只換臉 |
| `R2V` | 多圖 + 多影片 + 提示詞 | 官方上限 9 圖 / 3 影，每段參考影片可帶自己的音軌 |

其他共用設計：

- 權重來自 Hugging Face（`Comfy-Org/MiniMax-H3`、`leejet/MiniMax-H3-GGUF`），下載支援**斷點續傳**並比對 **size + SHA-256**。
- 影片模型視檔案格式自動選 `UnetLoaderGGUF` 或原生 `UNETLoader`；文字編碼器走原生 `CLIPLoader(type=minimax)`。
- 兩種權重等級可選：`high` = pruned `fp8_scaled` ≈ 19.5 GiB + 8-step turbo LoRA；`light` = GGUF `Q4_K_M` ≈ 10.6 GiB + 4-step turbo LoRA。
- 多參考（IT2V / IV2V / R2V）需要另一份 `ref2va` DiT，與 `fl2va` 並存、互不覆蓋。
- 內建 keepalive cell：每分鐘打一次心跳，避免閒置斷線（上限 11 小時）。
- 全部在 notebook 內用 venv 隔離，**保留平台自帶的 CUDA PyTorch**，不重裝 torch。

---

## L4 x1 與 T4 x2 的差異

| 項目 | `kaggle-l4-x1`（Colab） | `kaggle-t4-x2`（Kaggle） |
| --- | --- | --- |
| 平台 | Google Colab，執行階段選 **L4 GPU** | Kaggle，Accelerator 選 **GPU T4 x2** + Internet |
| 顯存 | 24 GiB 單卡 | 2 × 16 GiB（**分卡，不是合併**） |
| GPU 分配 | DiT / VAE / 文字編碼器全在 GPU0 | DiT / VAE 在 GPU0，**文字編碼器插 `SelectCLIPDevice` 指到 GPU1** |
| 權重存放 | `/content/drive/MyDrive/ComfyUI-H3/models`（雲端硬碟，**重開不用重下**） | `/tmp/ComfyUI/models`（**每次重開重下**） |
| 權重等級 | `auto`：≥22 GiB 用 `high`（fp8_scaled + 8 步），否則退回 `light` | 固定 `light`：GGUF `Q4_K_M` + 4 步 |
| 檔案大小 | 只 fl2va ≈ 32.5 GiB；含 ref2va ≈ 63 GiB | 固定含 ref2va ≈ 45 GiB |
| 預設時長 / 解析度 | `DURATION=4.0` 秒、`MEGAPIXELS=0.4` | `DURATION=2.0` 秒、`MEGAPIXELS=0.1`（試跑值，成功後再往上調） |
| 加速 | `--fast`（Ada 原生 fp8 矩陣乘）＋ **SageAttention**（自動安裝，wheel 快取在雲端硬碟；失敗自動回退 SDPA） | 無 `--fast`（T4 是 Turing，沒有 fp8）；不強制 fp16/bf16，避免 H3 在 T4 上 NaN／花屏 |
| 輸出位置 | Drive：`ComfyUI-H3/output/{video,image}`、上傳素材 `ComfyUI-H3/input`、工作流與設定 `ComfyUI-H3/user` | `/tmp/ComfyUI/output/{video,image}`（關機即消失，記得先下載） |
| 流程 | 7 步（含掛載硬碟、SageAttention） | 6 步 |
| 斷線保護 | keepalive cell，上限 11 h（擋閒置，擋不了方案時長上限） | 同左，上限 11 h（擋閒置，擋不了 12 h 硬上限） |

一句話選型：**只想快、有 Google 雲端硬碟空間 → 用 L4**；**只有 Kaggle 額度、能接受每次重下權重 → 用 T4 x2**。

---

## 執行方式

兩個 notebook 的用法一樣，差別只在平台設定。

### 1. 開環境（先做，別急著跑 cell）

**Colab（`kaggle-l4-x1`）**

1. 上傳／開啟 `kaggle-l4-x1-comfyui-minimax-h3.ipynb`。
2. `執行階段 → 變更執行階段類型 → L4 GPU`，儲存。
3. 確認 Google 雲端硬碟空間：含 ref2va 約需 **63 GiB**，只用 T2V/T2I 約 **41 GiB**。免費 15 GB 一定不夠。

**Kaggle（`kaggle-t4-x2`）**

1. 上傳／匯入 `kaggle-t4-x2-comfyui-minimax-h3.ipynb`（或直接貼進一個 Python cell）。
2. `Settings → Accelerator → GPU T4 x2`，並打開 `Internet`。
3. 需要約 **45 GiB** 可用磁碟（腳本會在空間不足時直接停下，不會硬塞）。

### 2. 跑主 cell

- Colab：**先跑第 2 個 cell**（權重與安裝都在裡面）。第一次會跳授權視窗，登入 Google 帳號允許存取雲端硬碟。
- Kaggle：跑第 1 個 cell。
- 等它印出 `ComfyUI 訪問網址：https://xxxx.trycloudflare.com`。第一次要下載數十 GB 權重，慢是正常的。

### 3. 開 ComfyUI 出片

1. 點開那個 `trycloudflare.com` 網址。
2. `Workflows` → 載入 `MiniMax_H3_Colab_T2V` 或 `MiniMax_H3_Kaggle_T2V` 等自動生成的副本。
   **請用腳本生成的副本**，內建模板庫裡的原始 H3 模板用的是 safetensors，會讓你再下一次大模型。
3. 第一次生成時權重要從硬碟（或網路）載入，會慢；同一 session 內第二次之後明顯變快。

### 4. 長時間生成

執行最後的 **keepalive** cell 並讓它保持執行中；要收工時按該 cell 的停止鈕，再執行 `stop_services()` 關掉 ComfyUI 與隧道。

---

## 常調參數

集中在主 cell 開頭的設定區：

| 參數 | 作用 |
| --- | --- |
| `MODEL_TIER`（僅 L4） | `auto` / `high` / `light`，強制權重等級 |
| `ENABLE_REF2VA`（僅 L4） | `False` 可省約 21 GiB，但 IT2V / IV2V / R2V 會消失 |
| `DURATION` / `MEGAPIXELS` | 影片長度（會按官方 17k+5 幀網格取整）與解析度；OOM 就先降這個 |
| `TURBO_STEPS` | 由等級自動覆寫（high=8 / light=4）；步數越少越快、細節越少 |
| `USE_FAST`（僅 L4） | `--fast` fp8 加速；若出現 NaN／花屏改 `False` |
| `SAGE_ATTENTION`（僅 L4） | 安裝並啟用 SageAttention（影片 attention 約快 20–40%），失敗自動回退 |
| `IT2V_REF_IMAGE_SIZE` / `IT2V_MEGAPIXELS` / `IT2V_TURBO` | IT2V 單獨一組：`max` 參考圖最貼臉但慢；預設 20 步不掛 turbo，要快速預覽才開 |
| `HASH_CHECK_EXISTING` | `never` 只比大小 / `once` 沒驗過才哈希 / `always` 每次都驗（雲端硬碟很慢） |
| `START_TUNNEL` | `False` 就只在本機 `127.0.0.1:8188` 開，不建 Cloudflare 隧道 |
| `UPDATE_CODE` | `True` 才會 `git pull` 更新 ComfyUI 程式碼 |

---

## 注意事項

- **臨時網址等於公開入口**：`trycloudflare.com` 那條連結誰拿到都能用，session 結束才失效，不要貼到公開場合。
- **Colab 免費版斷線就沒了**：權重還在雲端硬碟，但 ComfyUI 行程、暫存、venv 都要重跑。
- **Kaggle 關機即清空**：`/tmp` 的模型與輸出影片都不保留，出片後先下載。
- **T4 雙卡不是 24 GB**：只是把文字編碼器放到第二張卡，兩張卡的顯存不會合併；想在 T4 跑大解析度／長片很容易 OOM。
- **未實測的項目**：T4 版尚未標註實際生成速度與峰值記憶體；兩版的 R2V 品質與上限都依官方節點行為而定。
- 啟動成功 ≠ 推理自檢通過：服務起來只代表節點齊全，還要實際跑一次 T2V 才算通。

---

## 參考

- ComfyUI：<https://github.com/Comfy-Org/ComfyUI>
- GGUF 載入器：<https://github.com/leejet/ComfyUI-GGUF>
- 權重：<https://huggingface.co/Comfy-Org/MiniMax-H3>、<https://huggingface.co/leejet/MiniMax-H3-GGUF>
