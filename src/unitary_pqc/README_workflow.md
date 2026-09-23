# Unitary-PQC：末尾の Rz → Rx → Rz を除いた Cartan 回路

プロジェクトのルートから実行してください。以下の `!python` は Jupyter Notebook 用です。ターミナルでは先頭の `!` を外してください。

系の量子ビットは q0–q3、補助量子ビットは q4 です。各層で `(1,3)`、`(2,3)`、`(0,2)`、`(0,4)` の順に4個の U3-Cartan ブロックを適用します。各ブロックは、両量子ビット上の独立な `Ry → Rz → Ry`、`Rxx → Ryy → Rzz`、両量子ビット上の独立な `Ry → Rz → Ry` で構成されます。ここで U3 はこの Euler 回転を指します。

**1層のパラメータ数は `4 × 15 = 60`、L層では `60L` です。** q4 の各層末尾にあった `Rz(varphi) → Rx(2*phi) → Rz(varphi)` を削除しています。5量子ビット全体をユニタリに発展させ、測定・reset・測定結果に応じた操作は行いません。エネルギーは系の4量子ビットを対象とし、QFIM・HS は系の縮約状態と5量子ビット全体の純粋状態を解析します。

## 最適化から解析・描画まで

初めてこの回路を実行する場合は、次のセルを順に実行します。例は `h=0.1` です。

```python
!python src/unitary_pqc/unitary_pqc_overparam_compute.py --h-param 0.1 --stage all --device auto
!python src/unitary_pqc/unitary_pqc_overparam_visualize.py --h-param 0.1
!python src/unitary_pqc/unitary_pqc_overparam_draw_circuits.py --h-param 0.1
```

`--stage all` は VQE、QFIM・HS、Hessian をそれぞれ別の Python プロセスで順に実行します。描画は保存済み結果を使用します。最適化・可視化・回路描画で同じ `--h-param` を指定してください。回路描画は指定した `h` の保存先を使用します。個別のファイルや出力先を指定する場合は `--input` と `--output-dir` で上書きできます。

## 工程を個別に実行する

| コマンドの指定 | 処理 |
| --- | --- |
| `compute.py --stage all` | 最適化、その後 QFIM・HS と Hessian |
| `compute.py --stage vqe` または `_vqe.py` | 最適化と学習結果の保存 |
| `compute.py --stage analysis` または stage の省略 | QFIM・HS と Hessian |
| `compute.py --stage qfim` または `_qfim.py` | ランダム点と最適化経路の QFIM・HS |
| `compute.py --stage hessian` または `_hessian.py` | ランダム点の Hessian |

**既定の stage は `analysis` です。最適化を行う場合は `--stage all` または `--stage vqe` を指定してください。** 通常の QFIM は保存済み VQE パラメータ履歴を読み込みます。同じ60パラメータ回路の VQE 結果があれば、学習を繰り返さず解析できます。

```python
# 最適化だけ
!python src/unitary_pqc/unitary_pqc_overparam_vqe.py --h-param 0.1

# 保存済み学習結果を使って QFIM・HS と Hessian を計算
!python src/unitary_pqc/unitary_pqc_overparam_compute.py --h-param 0.1

# 解析工程をそれぞれ実行する場合
!python src/unitary_pqc/unitary_pqc_overparam_qfim.py --h-param 0.1
!python src/unitary_pqc/unitary_pqc_overparam_hessian.py --h-param 0.1
```

ランダム点だけの QFIM・HS には `--random-only` を指定します。この場合、VQE の保存結果は不要です。Hessian は常に VQE・QFIM の保存結果から独立して計算します。同じ設定なら、ランダム点 QFIM と Hessian は同じ乱数シード・パラメータ点を使用します。

```python
# ランダム点 QFIM・HS だけ
!python src/unitary_pqc/unitary_pqc_overparam_qfim.py --h-param 0.1 --random-only

# ランダム点 QFIM・HS と Hessian
!python src/unitary_pqc/unitary_pqc_overparam_compute.py --h-param 0.1 --random-only
```

全体の可視化は、VQE、ランダム点と最適化経路の QFIM・HS、Hessian の保存結果を使用します。`--random-only` で計算した結果だけを描画するときは、以下の専用モードを使用してください。

## 保存済み結果の可視化

```python
# 全体の図：エネルギー、収束、QFIM、HS、Hessian など
!python src/unitary_pqc/unitary_pqc_overparam_visualize.py --h-param 0.1

# エネルギー誤差をスペクトルギャップで規格化した図と成功確率
!python src/unitary_pqc/unitary_pqc_overparam_visualize.py --h-param 0.1 --gap-normalized-only

# Hessian のランク・条件数・固有値など
!python src/unitary_pqc/unitary_pqc_overparam_visualize.py --h-param 0.1 --hessian-only

# QFIM ランクの平均・SEM・最小・最大
!python src/unitary_pqc/unitary_pqc_overparam_visualize.py --h-param 0.1 --qfim-rank-only --qfim-rank-threshold 1e-12

# QFIM の平均 log det(I + kappa F)
!python src/unitary_pqc/unitary_pqc_overparam_visualize.py --h-param 0.1 --qfim-logdet-only --qfim-logdet-kappa 1
```

これら4つの `--*-only` モードは同時指定せず、必要な保存結果だけを読み込みます。最適化や行列解析を再実行せず、JAX・Optax・TensorCircuit の読み込みも不要です。QFIM のランクと log-determinant は `qfim_random_points_keep0123.npz` と `qfim_random_points_keep01234.npz` を使用します。前者が4量子ビットの縮約状態、後者が5量子ビット全体の純粋状態です。

QFIM ランクは `lambda_i >= threshold` の個数です。log-determinant は保存された全固有値から各サンプルの値を計算して平均します。ギャップ規格化の最終エネルギーは保存済みエネルギー履歴の最後の値を用います。現在の VQE 履歴は各更新前の値なので、最終更新後のパラメータのエネルギーとは異なる場合があります。

収束判定の許容誤差は通常描画に `--convergence-tolerance 1e-6` などを指定できます。複数指定すると各許容誤差の図を作成します。

## 最適化済み回路の描画

1層分だけ描く場合は `--layers 1` を追加します。描画にはその層の保存済み VQE 結果が必要です。

```python
!python src/unitary_pqc/unitary_pqc_overparam_draw_circuits.py --h-param 0.1 --layers 1
```

出力は `optimized_circuit_L1.png` などです。角度の数値を表示するには `--show-params`、行の折り返しには `--fold 20` などを指定します。回路内部の局所回転は保持され、削除対象だった q4 の層末尾の追加回転は含まれません。

## GPU・WSL と計算設定

デバイス選択は measured_1・DPQC と同じです。`--device auto` では VQE は利用可能な GPU を使用し、QFIM・HS と Hessian は CPU を使用します。`--device cpu` は選択した全工程を CPU に固定し、`--device gpu` は GPU を必須にします。Windows では既存の `.dpqc-gpu-wsl.json` に従って WSL 環境へ引き継ぎます。環境構築は [DPQC の GPU 実行手順](../dpqc/README_gpu.md) を参照してください。

`--device` の省略時は環境変数 `DPQC_DEVICE` が優先されるため、工程別の自動選択には `--device auto` を明示できます。実際のデバイスは起動時の `[DPQC] JAX backend: gpu/cpu` に表示されます。計算精度は float64 / complex128 です。

層数、試行数、最適化ステップ数、乱数シードなどは [config_overparam.py](../common/config_overparam.py) の設定を使用します。`--vqe-batch-size` は学習用、`--analysis-batch-size` は QFIM・HS・Hessian 用です。

```python
!python src/unitary_pqc/unitary_pqc_overparam_compute.py --h-param 0.1 --stage all --device auto --vqe-batch-size 5 --analysis-batch-size 1
```

Python からの `run_unitary_pqc_overparam()` は、CPU の同一プロセス内で学習から解析まで行う互換 API です。GPU 学習と CPU 解析を組み合わせる場合は上記 CLI を使用してください。工程ごとの API は `run_unitary_pqc_vqe_stage()`、`run_unitary_pqc_qfim_stage()`、`run_unitary_pqc_hessian_stage()` です。

## 保存先とモデルの区別

新しい結果の保存先は `figs/unitary_pqc/u3_cartan/h_<h>/` です。

| ディレクトリ | 保存内容 |
| --- | --- |
| `numerical_results/energy/` | VQE のパラメータ・エネルギー履歴など |
| `numerical_results/qfim/` | QFIM のランダム点・最適化経路解析 |
| `numerical_results/hs/` | Hilbert–Schmidt metric の解析 |
| `numerical_results/hessian/` | Hessian の行列と集計結果 |
| `figures/` | 各解析の図 |
| `optimized_circuits/` | 最適化済み回路図 |

この回路のアーカイブは `ansatz="unitary_pqc"`、`num_params_per_layer=60` を持ち、`measurement_outcome` は保存しません。旧12パラメータ版の `figs/unitary_pqc/h_<h>/` と、62パラメータ版の `figs/unitary_pqc_measured_1/u3_cartan/h_<h>/` は別のモデルです。それらのパラメータ・解析結果を新しいモデルの結果として直接読み込むことはできません。新しい60パラメータ回路で学習・解析してください。
