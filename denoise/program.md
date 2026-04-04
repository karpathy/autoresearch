# autoresearch — 画像ノイズ除去

ガウシアンノイズ（σ=25）を加えた画像からクリーン画像を復元するモデルを、自律的に改善する実験。改善の様子が `samples/latest.png` に視覚的に残る。

## セットアップ

1. **ランタグを決定**: 日付ベース（例 `apr4`）。`denoise/<tag>` ブランチが未使用であること。
2. **ブランチ作成**: `git checkout -b denoise/<tag>`
3. **ファイル読み込み**:
   - `prepare.py` — データ・評価・可視化（**変更禁止**）
   - `train.py` — モデル・学習ループ（**これだけ編集する**）
4. **データ準備**: `uv run prepare.py`（初回のみ。Kodak画像24枚を `~/.cache/autoresearch-denoise/` にDL）
5. **results.tsv 初期化**: ヘッダ行のみ作成
6. **確認して開始**

## ルール

- **編集できるファイル**: `train.py` のみ
- **変更禁止**: `prepare.py`
- **依存追加禁止**: `pyproject.toml` にあるもの + Python標準ライブラリのみ

## 目標

**val_psnr を最大化する**（dB, 高いほど良い）。

ノイズ画像のベースライン PSNR は約 20 dB。何もしないモデルでも 20 dB。まともなデノイザーなら 28–30 dB、良いデノイザーなら 30 dB 以上。

## 出力形式

`uv run train.py` を実行すると以下を出力:

```
---
val_psnr:         28.4321
training_seconds: 300.1
total_seconds:    310.5
num_steps:        5000
num_epochs:       12
num_params:       300000
```

結果抽出:
```
grep "^val_psnr:" run.log
```

**視覚出力**: 毎回 `samples/latest.png` に比較画像（ノイズ入力 | 復元結果 | 正解）が保存される。`samples/<commit>.png` にも履歴として残る。

## ログ形式

`results.tsv`（タブ区切り）:

```
commit	val_psnr	params	status	description
```

1. git commit hash（7文字）
2. val_psnr（例: 28.4321）— クラッシュ時は 0.0000
3. パラメータ数（例: 300K）
4. status: `keep`, `discard`, `crash`
5. 試行内容の説明

例:
```
commit	val_psnr	params	status	description
a1b2c3d	26.1234	295K	keep	baseline (plain conv stack)
b2c3d4e	27.8901	295K	keep	add residual learning (predict noise)
c3d4e5f	27.5000	1.2M	discard	U-Net but too big, slower convergence
d4e5f6g	28.4321	590K	keep	lightweight U-Net with skip connections
```

## 実験ループ

LOOP FOREVER:

1. git状態を確認
2. `train.py` を編集（モデル構造、ハイパラ、学習手法を変更）
3. git commit
4. 学習実行: `uv run train.py > run.log 2>&1`
5. 結果確認: `grep "^val_psnr:" run.log`
6. 空ならクラッシュ → `tail -n 50 run.log` でデバッグ
7. results.tsv に記録
8. val_psnr 改善 → コミット保持
9. 同等以下 → **results.tsv を退避してから** `git reset --hard HEAD^` し、退避した results.tsv を書き戻す

**重要: results.tsvの退避手順**（discardのとき）:
```bash
cp results.tsv /tmp/results.tsv.bak
git reset --hard HEAD^
cp /tmp/results.tsv.bak results.tsv
```
`git reset --hard` は results.tsv も巻き戻してしまうため、必ずこの手順を踏むこと。

## 改善戦略ガイド

ベースラインの `train.py` は**意図的に素朴**に作られている。以下の順序で改善できる:

### Phase 1: 低コストで大きな改善（+2–4 dB）
- **残差学習**: モデルがノイズを予測し、入力から引く（`return x - self.net(x)`）
- **スキップ接続**: 入力を出力に直結（最も単純な残差）
- **Loss関数**: MSE → L1（`F.l1_loss`）に変更

### Phase 2: アーキテクチャ改善（+1–2 dB）
- **U-Net構造**: ダウンサンプル → ボトルネック → アップサンプル + skip connections
- **チャンネル数増加**: 64 → 96 or 128（メモリと相談）
- **Batch Normalization** or **Group Normalization**

### Phase 3: 学習テクニック（+0.3–1 dB）
- **学習率スケジューリング**: CosineAnnealing, warmup
- **データ拡張の強化**: multi-scale patches
- **混合損失**: L1 + SSIM loss
- **Optimizer**: AdamW, weight decay

### Phase 4: 高度な手法（+0.3–0.5 dB）
- **Attention機構**: Channel attention (SE block) or self-attention
- **Multi-scale processing**: 複数解像度で処理して統合
- **Progressive training**: 徐々にノイズレベルを上げる

## タイムアウト

各実験は5分間の学習 + 数秒の評価。10分超えたら kill して crash 扱い。

## 視覚的進捗

`samples/` ディレクトリに各実験の比較画像が残る。改善の歴史が目で見える:
- `samples/latest.png` — 最新の結果
- `samples/<commit>.png` — 各実験の結果

## NEVER STOP

実験ループ開始後、確認を求めない。手動停止まで無限ループ。
