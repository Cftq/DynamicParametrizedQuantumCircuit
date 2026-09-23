# Reset-DPQCの学習・解析・描画

プロジェクトのルートから実行してください。学習済みデータがある場合、学習を再実行する必要はありません。

計算プログラムは`--device auto/cpu/gpu`に対応しています。このPCでは既定の`auto`で学習はRTX 3080、QFIM・HessianはCPUを使用します。学習する例は`python src/dpqc/DPQC_overparam_reset_compute.py --stage vqe --h-param 0.1`です。解析だけなら`--stage vqe`を省略します。`--stage all`も学習GPU・解析CPUの順で実行し、`--device cpu/gpu`の明示指定は選択した全工程に適用します。[GPU実行・環境設定・速度比較](README_gpu.md)も参照してください。

| ファイル | 実行する処理 |
| --- | --- |
| `DPQC_overparam_reset_vqe.py` | 変分量子回路の学習とパラメータ履歴の保存 |
| `DPQC_overparam_reset_qfim.py` | ランダムなパラメータ点におけるQFIM計算 |
| `DPQC_overparam_reset_hessian.py` | ランダムなパラメータ点におけるHessian計算 |
| `DPQC_overparam_reset_compute.py` | QFIMとHessianの計算を別プロセスで順に実行 |
| `DPQC_overparam_reset_visualize.py` | 保存済みデータから図を描画 |
| `dpqc_reset_model.py` | 共通の回路・パラメータ定義・保存先・モデル情報 |

学習が必要な場合だけ実行します。

```powershell
python src/dpqc/DPQC_overparam_reset_vqe.py --h-param 0.1
```

QFIMとHessianはそれぞれ独立して計算できます。現在のランダム点解析は学習済みパラメータを使わないため、VQEの保存データも不要です。

```powershell
python src/dpqc/DPQC_overparam_reset_qfim.py --h-param 0.1
python src/dpqc/DPQC_overparam_reset_hessian.py --h-param 0.1
```

両方を計算する場合は次のコマンドでも実行できます。**従来と異なり、既定の処理は `analysis` であり、学習は実行しません。**

```powershell
python src/dpqc/DPQC_overparam_reset_compute.py --h-param 0.1
```

学習・QFIM・Hessianの保存データが揃っていれば、次のコマンドで再計算せずに描画できます。

```powershell
python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1
```

Hessianの保存データだけを描画する場合は次を使います。

```powershell
python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1 --hessian-only --reuse-hessian-results
```

`--hessian-only` や `--with-hessian` を付けた描画では、`--reuse-hessian-results` も付けるとHessianを再計算しません。どの描画モードもVQEを実行しません。

各層の量子ビット対 `(1,3)`, `(2,3)`, `(0,2)`, `(0,4)` に、次の順でゲートを適用します。

1. 両量子ビットそれぞれに `RY → RZ → RY`（各3パラメータ）。
2. その対に `RXX → RYY → RZZ`（3パラメータ）。
3. 両量子ビットそれぞれに `RY → RZ → RY`（各3パラメータ）。

量子ビット間・ブロック間・層間・前後でパラメータは共有しません。1ブロック15個、1層60個、L層で `60 * L` 個です。パラメータ配列は層、ブロックの順で並べ、各ブロック内は前段の第1量子ビット3個、第2量子ビット3個、`RXX,RYY,RZZ` の3個、後段の第1量子ビット3個、第2量子ビット3個の順です。`U3` はここでは上記 `RY → RZ → RY` を意味します。各層の末尾にあるreset処理 `CX(center→fresh) → CRX(fresh→center,π)` と、数値計算における等価な `trace_center(ρ) ⊗ |0⟩⟨0|` は従来どおりです。

修正版の保存先は `figs/dpqc_reset/u3_cartan/h_0.1/numerical_results/` の `energy`、`qfim`、`hessian` です。学習、QFIM、Hessian、可視化はすべてこのモデル専用の保存先を使用します。旧回路の `12 * L` パラメータの結果とは互換性がないため、旧保存先 `figs/dpqc_reset/h_0.1/` はそのまま保持します。旧結果の退避や削除は不要です。

モデルIDは `dpqc_reset_u3_cartan_fixed_rx_pi`、モデル情報のschemaは3です。以前の版で出ていた `schema_version: 2 != 3` は、新旧モデルが同じ保存先を使用したことによる不一致でした。修正版では上記の別ディレクトリに新しく計算し、可視化も同じ場所を参照します。schemaの数字だけを書き換えて旧データを流用することはできません。

QFIMの層数・サンプル数・乱数シードは `src/common/config_overparam.py` を参照します。Hessianも同じ設定を既定値とし、専用ファイルでは `--layers`、`--num-samples`、`--seed-base` で変更できます。

従来の `--stage vqe`、`--stage qfim`、`--stage hessian` も利用できます。学習から全工程を実行する場合だけ `--stage all` を明示してください。既存の学習データに対応する `reset_model_metadata.json` が失われている場合は、そのモデル情報を復元してください。解析処理が学習を自動実行して作り直すことはありません。

## ギャップで規格化した最適化性能

保存済みの各試行の最終エネルギーを使って、横軸を層数、縦軸を
`(E_final - E0) / Delta` としたbeeswarm plotと、規格化誤差の成功確率の図を作成します。
beeswarmでは各試行の縦軸の値を保持して横方向へ配置し、層ごとの中央値を横線で示します。
`Delta = E1 - E0` は同じHamiltonianのスペクトルギャップです。
基底状態が縮退している場合、`E1` は基底エネルギーより高い最初の異なる固有値を指します。
成功判定は `(E_final - E0) / Delta < epsilon` で、
`epsilon = 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10` のそれぞれに対して初期値試行中の成功割合を求めます。

これらの図だけを追加する場合は、プロジェクトのルートで次を実行します。
学習・QFIM・Hessianは再計算せず、QFIM・Hessianの保存済み解析結果も必要ありません。
入力には保存済み学習結果とresetモデルのメタデータを使います。

```powershell
python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1 --gap-normalized-only
```

出力先は `figs/dpqc_reset/u3_cartan/h_<h>/` の下です。

- `energy_figures/final_gap_normalized_energy_error.pdf`
- `energy_figures/success_probability_multiple_tolerances_gap_normalized.pdf`
- `energy_figures/gap_normalized_success_probability.pdf`
- `numerical_results/energy/gap_normalized_energy_statistics.npz`

成功率の2つのPDFは同じ内容で、従来のファイル名も維持しています。平均・SEM・最小最大・中央値は数値ファイルに保存します。
通常の可視化コマンドにもこれらの出力を追加しています。
規格化前の誤差に基づく `success_probability_multiple_tolerances.pdf` は `delta = 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10` の8閾値を使い、従来どおり `<= delta` で成功を判定します。
`--gap-normalized-only` は `--hessian-only` や `--with-hessian` と同時には指定できません。

## ランダム点QFIMの平均log-determinant

保存済みのQFIM固有値から、層数に対する `mean_theta[log det(I + kappa F)]` を描画します。既定値は `kappa = 1` で、`--qfim-logdet-kappa 10` のように有限の正の値で変更できます。通常の可視化にも出力を追加しています。この指標だけを出力するコマンドは次です。

```powershell
python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1 --qfim-logdet-only
```

専用モードはQFIM・学習・Hessianを再計算しません。VQE履歴も必要ありません。`figs/dpqc_reset/u3_cartan/h_<h>/numerical_results/qfim/` の `qfim_random_points_keep0123.npz` と `qfim_random_points_keep01234.npz`、現在のresetモデルを識別する `reset_model_metadata.json` を使用します。

各ランダム点で、保存された全固有値による `sum_i log1p(kappa * lambda_i)` を計算してから平均します。自然対数を使い、ランクの閾値やパラメータ数による規格化を加えません。既存のサンプル生成分布 `p(theta)` は各角度が独立な `[-pi, pi)` の一様分布です。

出力先は `figs/dpqc_reset/u3_cartan/h_<h>/figures/qfim/logdet/` で、`keep0123`・`keep01234` ごとの `qfim_logdet_random_points_<keep_key>_kappa_1.pdf` と `.npz` を保存します。図は平均±SEMを示し、数値ファイルには各点の値、最小・最大を含む統計量、κ、サンプル数、入力情報を保存します。κを変えると別名で保存されます。`--qfim-logdet-only` は `--gap-normalized-only`、`--hessian-only`、`--with-hessian` と同時指定できません。
