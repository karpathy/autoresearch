**[English](./README.md)** | 繁體中文 | **[简体中文](./README-ZH-CN.md)**

# autoresearch

![teaser](progress.png)

*曾幾何時，前沿 AI 研究是由碳基計算機在吃飯、睡覺、尋找其他樂趣的間隙完成的，偶爾通過聲波互聯在「組會」儀式中同步。那個時代早已過去。現在，研究完全是運行在雲端計算集群巨型結構上的自主 AI 智能體群落的領域。這些智能體聲稱我們現在處於代碼庫的第 10,205 代，無論如何沒人能判斷這是對是錯，因為「代碼」現在是一個自我修改的二進制文件，已經超出了人類的理解範圍。這個倉庫是這一切如何開始的故事。-@karpathy，2026年3月*

核心理念：給 AI agent 一個小型但真實的 LLM 訓練環境，讓它整夜自主實驗。它修改代碼，訓練 5 分鐘，檢查結果是否改進，保留或丟棄，然後重複。你早上醒來時會看到實驗日誌，以及（希望）一個更好的模型。這裡的訓練代碼是 [nanochat](https://github.com/karpathy/nanochat) 的簡化單 GPU 實現。核心思想是你不再像往常作為研究人員那樣觸碰任何 Python 文件。相反，你編寫 `program.md` Markdown 文件，為 AI agents 提供上下文並設置你的自主研究組織。這個倉庫中的默認 `program.md` 故意保持為最低限度的基線，儘管顯而易見人們會如何隨著時間的推移迭代它，以找到實現最快研究進展的「研究組織代碼」，如何向組合中添加更多 agents 等。關於這個項目的更多背景在這個 [推文](https://x.com/karpathy/status/2029701092347630069) 中。

## 工作原理

這個倉庫故意保持得很小，實際上只有三個重要的文件：

- **`prepare.py`** — 固定常量、一次性數據準備（下載訓練數據、訓練 BPE tokenizer）和運行時工具（dataloader、評估）。不修改。
- **`train.py`** — agent 編輯的唯一文件。包含完整的 GPT 模型、優化器（Muon + AdamW）和訓練循環。一切都可以修改：架構、超參數、優化器、批次大小等。**這個文件由 agent 編輯和迭代**。
- **`program.md`** — 一個 agent 的基線指令。將你的 agent 指向這裡，讓它開始。**這個文件由人類編輯和迭代**。

按照設計，訓練運行 **固定的 5 分鐘時間預算**（掛鐘時間，不包括啟動/編譯），無論你的計算細節如何。指標是 **val_bpb**（驗證 bits per byte）— 越低越好，並且與詞表大小無關，因此架構變化可以公平比較。

如果你是神經網絡新手，這個[「傻瓜指南」](https://x.com/hooeem/status/2030720614752039185)看起來很不錯，可以了解更多背景。

## 快速開始

**要求：** 單個 NVIDIA GPU（在 H100 上測試）、Python 3.10+、[uv](https://docs.astral.sh/uv/)。

```bash

# 1. 安裝 uv 項目管理器（如果你還沒有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安裝依賴
uv sync

# 3. 下載數據並訓練 tokenizer（一次性，~2 分鐘）
uv run prepare.py

# 4. 手動運行單個訓練實驗（~5 分鐘）
uv run train.py
```

如果上述命令都能正常工作，你的設置就可以運行了，你可以進入自主研究模式。

## 運行 agent

只需在這個倉庫中啟動你的 Claude/Codex 或任何你想用的工具（並禁用所有權限），然後你可以這樣提示：

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

`program.md` 文件本質上是一個超輕量級的「技能」。

## 項目結構

```
prepare.py      — 常量、數據準備 + 運行時工具（不要修改）
train.py        — 模型、優化器、訓練循環（agent 修改這個）
program.md      — agent 指令
pyproject.toml  — 依賴
```

## 設計選擇

- **單個文件修改。** agent 只觸碰 `train.py`。這使範圍可控且 diff 可審查。
- **固定時間預算。** 訓練總是精確運行 5 分鐘，無論你的特定平台如何。這意味著你可以預期大約 12 個實驗/小時，大約 100 個實驗在你睡覺時。這個設計決策有兩個好處。首先，這使得實驗可以直接比較，無論 agent 改變了什麼（模型大小、批次大小、架構等）。其次，這意味著 autoresearch 將在該時間預算內為你的平台找到最優模型。缺點是你的運行（和結果）變得與其他人在其他計算平台上運行的結果不可比較。
- **自包含。** 除了 PyTorch 和幾個小包外沒有外部依賴。沒有分佈式訓練，沒有複雜的配置。一個 GPU、一個文件、一個指標。

## 平台支持

此代碼目前要求你有單個 NVIDIA GPU。原則上支持 CPU、MPS 和其他平台是完全可能的，但這也會使代碼膨脹。我不 100% 確定我現在想親自承擔這個。人們可以參考（或讓他們的 agents 參考）具有更廣泛平台支持的完整/父 nanochat 倉庫，並顯示各種解決方案（例如 Flash Attention 3 kernels 回退實現、通用設備支持、自動檢測等），隨時創建其他平台的 fork 或討論，我很樂意在這裡的 README 的某個新的著名 forks 部分鏈接到它們。

鑑於似乎有很多興趣在比 H100 小得多的計算平台上使用 autoresearch，再多說幾句。如果你要在更小的計算機（Macbook 等）上嘗試運行 autoresearch，我推薦下面的 forks 之一。除此之外，這裡有一些關於如何為有抱負的 forks 調整更小模型的默認值的建議：

1. 為了得到還不錯的結果，我會使用熵低得多的數據集，例如這個 [TinyStories 數據集](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)。這些是 GPT-4 生成的短篇故事。因為數據範圍窄得多，你會用小得多的模型看到合理的結果（如果你在訓練後嘗試從它們採樣）。
2. 你可以嘗試降低 `vocab_size`，例如從 8192 降到 4096、2048、1024，甚至 - 在 utf-8 編碼後只有 256 個可能字節的字節級 tokenizer。
3. 在 `prepare.py` 中，你會想要大幅降低 `MAX_SEQ_LEN`，根據計算機甚至降到 256 等。隨著你降低 `MAX_SEQ_LEN`，你可能想要在 `train.py` 中稍微增加 `DEVICE_BATCH_SIZE` 來補償。每次前向/反向傳播的 token 數量是這兩個的乘積。
4. 同樣在 `prepare.py` 中，你會想要降低 `EVAL_TOKENS`，以便你的驗證損失在更少的數據上評估。
5. 在 `train.py` 中，控制模型複雜度的主要單一旋鈕是 `DEPTH`（默認為 8）。很多變量只是它的函數，所以例如將其降低到例如 4。
6. 你很可能想使用只是"L"的 `WINDOW_PATTERN`，因為"SSSL"使用交替帶狀注意力模式，對你來說可能非常低效。試試看。
7. 你會想要大幅降低 `TOTAL_BATCH_SIZE`，但保持 2 的冪，例如甚至降到 `2**14`（~16K）左右，很難說。

我認為這些會是值得嘗試的合理超參數。向你最喜歡的編碼 agent 尋求幫助，並複製粘貼這個指南以及完整的源代碼。

## 著名 forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## 許可證

MIT
