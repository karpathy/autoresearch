# codesolver — コード自己改善実験

Claude Codeが自律的にアルゴリズム問題の解法を書き、テストで評価し、改善を繰り返す実験。

## セットアップ

1. **ランタグを決定**: 日付ベースのタグ（例: `apr4`）。ブランチ `codesolver/<tag>` が未使用であること。
2. **ブランチ作成**: `git checkout -b codesolver/<tag>`
3. **ファイル読み込み**: 以下を読んで全体像を把握:
   - `prepare.py` — 問題定義 + 評価ハーネス（**変更禁止**）
   - `solve.py` — 解法実装（**これだけ編集する**）
4. **results.tsv 初期化**: ヘッダ行のみ作成
5. **確認して開始**

## ルール

**編集できるファイル**: `solve.py` のみ
**編集禁止**: `prepare.py`（問題定義とテストケースは固定）
**依存追加禁止**: Python標準ライブラリのみ使用可能（import math, collections, itertools, etc. はOK）

## 目標

**pass_rate を最大化する。**

30問のアルゴリズム問題（Easy 10 / Medium 10 / Hard 10）に対して、全テストケースを通過する問題数の割合。

## 出力形式

`uv run prepare.py` を実行すると以下の形式で出力される:

```
  PASS  [easy  ] two_sum
  FAIL  [medium] three_sum: test 2: got [-1, 0, 1], expected [[-1, -1, 2], [-1, 0, 1]]
  SKIP  [hard  ] lcs_length: not implemented

---
pass_rate:    0.133333
solved:       4/30
easy:         3/10
medium:       1/10
hard:         0/10
elapsed_sec:  0.2
```

結果の抽出:
```
grep "^pass_rate:" run.log
```

## ログ形式

`results.tsv`（タブ区切り）:

```
commit	pass_rate	solved	status	description
```

1. git commit ハッシュ（短縮7文字）
2. pass_rate（例: 0.433333）
3. solved（例: 13/30）
4. status: `keep`, `discard`, `crash`
5. 試行内容の短い説明

例:
```
commit	pass_rate	solved	status	description
a1b2c3d	0.000000	0/30	keep	baseline (all stubs)
b2c3d4e	0.333333	10/30	keep	implement all easy problems
c3d4e5f	0.333333	10/30	discard	attempt three_sum but wrong output format
d4e5f6g	0.433333	13/30	keep	fix three_sum, add coin_change and binary_search
```

## 実験ループ

LOOP FOREVER:

1. 現在のgit状態を確認
2. `solve.py` を編集して問題の解法を実装・改善
3. git commit
4. 評価実行: `uv run prepare.py > run.log 2>&1`
5. 結果確認: `grep "^pass_rate:\|^solved:" run.log`
6. grep出力が空ならクラッシュ。`tail -n 30 run.log` でエラーを確認して修正
7. results.tsv に記録
8. pass_rate が改善 → コミットを保持（ブランチ前進）
9. pass_rate が同じか悪化 → `git reset --hard HEAD^` で巻き戻し

## 戦略ガイド

### 進め方の推奨順序

1. **まずEasy問題を全て解く** — 確実にpass_rateを稼ぐ
2. **Medium問題に着手** — 一問ずつ慎重に実装
3. **Hard問題に挑戦** — DP, スタック, 二分探索などのテクニックを駆使
4. **失敗した問題のデバッグ** — `uv run prepare.py --quick` で最初の失敗だけ表示

### 重要なヒント

- **テストケースをよく読む**: 戻り値の型（list vs tuple）、ソート順序、エッジケースに注意
- **一度に多くの問題を変更しない**: 2-3問ずつ実装してテストし、壊れた場合の切り分けを容易にする
- **エッジケースに注意**: 空リスト、空文字列、負数、ゼロ、単一要素
- **効率も重要**: 一部のテストケースは大きな入力を含む。O(n²) では TLE する問題がある

### 問題一覧

**Easy**: two_sum, is_palindrome, fizzbuzz, fibonacci, max_profit, valid_parentheses, merge_sorted, remove_duplicates, count_chars, reverse_words

**Medium**: max_subarray, group_anagrams, longest_common_prefix, binary_search, three_sum, container_water, coin_change, product_except_self, longest_substring_no_repeat, rotate_matrix

**Hard**: lcs_length, edit_distance, longest_increasing_subsequence, knapsack, spiral_order, word_break, decode_ways, largest_rectangle_histogram, median_sorted_arrays, serialize_deserialize_tree (serialize + deserialize)

## タイムアウト

各評価は通常数秒で完了する（GPU学習と違い計算量が小さい）。1分以上かかる場合はアルゴリズムに問題がある。

## NEVER STOP

実験ループ開始後、人間に確認を求めない。pass_rate が 1.0 になっても、コードの簡略化や別解の試行を続ける。手動で停止されるまでループを継続する。
