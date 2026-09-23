# 測定結果1のUnitary-PQC：学習・解析・描画

プロジェクトのルートから実行してください。同じ `u3_cartan` 回路の学習データがある場合、学習の再実行は不要です。旧 `Rz–Rz–Rxx` 回路の結果は使用できないため、新しい回路では再計算してください。

各層は `(1,3)`, `(2,3)`, `(0,2)`, `(0,4)` の4ブロックで構成されます。各ブロックは reset DPQC と同じ `u3_cartan` で、両量子ビットに独立した `Ry → Rz → Ry` を適用し、`Rxx → Ryy → Rzz`、再び両量子ビットの独立した `Ry → Rz → Ry` が続きます。ここでの U3 はこの Euler 回転を指し、ライブラリの U3 ゲートではありません。

各ブロック15パラメータに加え、各層末尾の中心 ancilla（q4）に従来どおり `Rz(varphi) → Rx(2*phi) → Rz(varphi)` を適用します。合計は **1層あたり `4 × 15 + 2 = 62` パラメータ**です。5量子ビットのユニタリ回路を維持し、reset 操作は追加しません。

| ファイル | 処理 |
| --- | --- |
| `unitary_pqc_measured_1_overparam_vqe.py` | 学習し、パラメータ履歴・エネルギーなどを保存 |
| `unitary_pqc_measured_1_overparam_qfim.py` | ランダム点と保存済み学習経路のQFIM・HSを計算 |
| `unitary_pqc_measured_1_overparam_hessian.py` | ランダム点のHessianを計算 |
| `unitary_pqc_measured_1_overparam_compute.py` | QFIM・HS、Hessianを別プロセスで順に実行 |
| `unitary_pqc_measured_1_overparam_visualize.py` | 保存結果を読み込み、図を作成 |
| `unitary_pqc_measured_1_overparam_common.py` | 共通の回路定義・状態・保存形式・描画補助 |
| `unitary_pqc_measured_1_model.py` | 軽量な回路定数・U3-Cartanブロック定義 |
| `unitary_pqc_measured_1_overparam_draw_circuits.py` | 保存済み学習結果からU3-Cartan回路を描画 |
| `unitary_pqc_measured_1_overparam_cli.py` | 数値計算を開始しない共通の引数処理 |

学習が必要な場合だけ実行します。

```powershell
python src/unitary_pqc/unitary_pqc_measured_1_overparam_vqe.py --h-param 0.1
```

QFIM・HSとHessianは個別に計算できます。QFIMの学習経路解析は、保存済みの`energy/vqe_optimization_results.npz`からパラメータ履歴を読み込みます。学習は実行しません。

```powershell
python src/unitary_pqc/unitary_pqc_measured_1_overparam_qfim.py --h-param 0.1
python src/unitary_pqc/unitary_pqc_measured_1_overparam_hessian.py --h-param 0.1
```

ランダム点のQFIM・HSだけ必要な場合は、`--random-only`を付けます。この場合は学習データも不要です。Hessianは常に学習・QFIMの保存データから独立しており、同じ設定ならQFIMと同一の乱数シード・パラメータ点を使用します。

```powershell
python src/unitary_pqc/unitary_pqc_measured_1_overparam_qfim.py --h-param 0.1 --random-only
```

両方の解析を実行するには次のコマンドを使用します。**既定の処理を`all`から`analysis`に変更したため、このコマンドでは学習しません。** 学習データがない状態でランダム点解析を行う場合は、ここにも`--random-only`を追加できます。

```powershell
python src/unitary_pqc/unitary_pqc_measured_1_overparam_compute.py --h-param 0.1
```

保存データが揃っていれば、再学習・再計算せずに図を作成できます。

```powershell
python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1
# Hessianの保存データだけを描画する場合
python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1 --hessian-only
```

保存先は `figs/unitary_pqc_measured_1/u3_cartan/h_0.1/numerical_results/` の `energy`、`qfim`、`hs`、`hessian` です。可視化・回路描画も同じ `u3_cartan` 保存先を使用します。旧14パラメータ回路の `figs/unitary_pqc_measured_1/h_0.1/` はそのまま保持し、読み込む際のパラメータ数チェックでも旧結果を拒否します。全体の描画には学習経路のQFIM・HSも必要なので、`--random-only`のみで作成した結果だけでは全体の図は揃いません。

`--vqe-batch-size`は学習専用、`--analysis-batch-size`はQFIM・HSとHessian用です。層数・サンプル数・乱数シードなどは`src/common/config_overparam.py`を参照します。

従来の明示的な`--stage vqe`と`--stage qfim`に加え、`--stage hessian`が使えます。`--stage qfim`はQFIM・HSだけを計算します。以前この処理に含まれていたHessianは専用ファイルまたは`--stage analysis`で計算してください。学習から全工程を実行する場合だけ`--stage all`を指定します。

既存のPython API `run_unitary_pqc_overparam()`は、同一プロセス内で学習から全工程を実行する互換用のCPU関数として維持しています。GPU学習とCPU解析を組み合わせる場合は、`compute.py --stage all`を使用してください。解析のみの場合は、専用ファイルの`run_unitary_pqc_qfim_stage()`と`run_unitary_pqc_hessian_stage()`を使用できます。各ステージ関数の`device`引数でもデバイスを選択できますが、異なるデバイスの工程は別のPythonプロセスで実行してください。

旧computeファイル経由での数値関数呼出しと結果参照は引き続き利用できます。独自スクリプトでモジュールの共有状態へ直接代入する場合は、`unitary_pqc_measured_1_overparam_common`をimportして変更してください。設定値の変更には引き続き`config_overparam.py`を使用します。

## GPU学習とCPU解析

DPQC・Reset DPQCと同じデバイス選択を使用します。既定の`--device auto`では、VQEは利用可能なGPUを使い、QFIM・HS・HessianはCPUで計算します。`implement.ipynb`の既存セルも引数を追加せずにこの設定で動作します。

```powershell
# 学習のみ：既定でGPUを利用
python src/unitary_pqc/unitary_pqc_measured_1_overparam_compute.py --h-param 0.1 --stage vqe
# 保存済み学習結果を使ってQFIM・HSとHessianをCPUで計算
python src/unitary_pqc/unitary_pqc_measured_1_overparam_compute.py --h-param 0.1
# 学習から解析まで：工程ごとに別プロセスで実行
python src/unitary_pqc/unitary_pqc_measured_1_overparam_compute.py --h-param 0.1 --stage all
```

個別の`_vqe.py`、`_qfim.py`、`_hessian.py`でも同じ既定値です。WindowsではDPQCで設定済みの`.dpqc-gpu-wsl.json`を読み、学習・解析とも同じWSL環境を使用します。設定方法は[DPQCのGPU実行手順](../dpqc/README_gpu.md)を参照してください。

`--device cpu`は選択した全工程をCPUに固定し、`--device gpu`は全工程でGPUを必須にします。GPUを明示指定して利用できない場合は、計算開始前にエラーになります。`--device`を省略した場合は環境変数`DPQC_DEVICE`が優先されるため、工程別の既定選択を明示するには`--device auto`を指定してください。WSL/LinuxのVQEの`auto`はJAXの既定デバイスと外部のJAX設定に従います。

実際のデバイスは起動時の`[DPQC] JAX backend: gpu/cpu`ログに表示されます。CPU/GPUとも同じ `u3_cartan` 回路を使い、float64/complex128精度と学習アルゴリズムを維持します。

## ギャップで規格化した最適化性能

可視化プログラムは、層数を横軸とした `(E_final - E0) / Delta` のbeeswarm plotと、
`epsilon = 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10` の成功確率の図も作成します。
誤差図では各試行の縦軸の値を保持して横方向へ配置し、層ごとの中央値を横線で示します。
規格化誤差の成功判定は厳密な `< epsilon` です。
全試行を分母に含めます。`Delta` は同じ `h` のHamiltonianについて、
基底状態部分空間より上の最初の励起エネルギーとの差です。
補助量子ビットなどによる基底状態の縮退にも対応します。
規格化誤差は基底状態部分空間の外にある重みの上界であり、その重み自体の測定値ではありません。

これらの図だけを作成するには次を実行します。
保存済みの `vqe_optimization_results.npz` のエネルギー履歴だけを使い、
学習・QFIM・HS・Hessianは再実行せず、それらの解析結果ファイルも必要ありません。
このモードはJAX・Optax・TensorCircuitを読み込みません。

```powershell
python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1 --gap-normalized-only
```

既定の入力は `figs/unitary_pqc_measured_1/u3_cartan/h_<h>/numerical_results/energy/vqe_optimization_results.npz` です。
出力は同じ `h_<h>/` の下に保存します。

- `figures/energy/final_gap_normalized_energy_error.pdf`
- `figures/energy/success_probability_multiple_tolerances_gap_normalized.pdf`
- `figures/energy/gap_normalized_success_probability.pdf`
- `numerical_results/energy/gap_normalized_energy_statistics.npz`

成功率の2つのPDFは同じ内容で、従来のファイル名も維持しています。
規格化前の誤差に基づく `success_probability_multiple_tolerances.pdf` も同じ6閾値を使い、従来どおり `<= delta` で成功を判定します。
数値ファイルには各試行の保存済み最終エネルギー、規格化誤差、ギャップ、
閾値、試行数、成功確率、平均・SEM・最小最大・中央値、入力ファイルの情報を保存します。
入力の学習データは変更しません。丸め誤差による微小な負の誤差だけを0に補正し、
誤差図では0を表示できる対称対数軸を使います。
保存済み履歴の最後の列を `E_final` として扱います。
現行のMeasured 1の学習履歴は各更新前のエネルギーを記録するため、
最終更新後のパラメータに対するエネルギーを新たに評価する処理は行いません。

保存場所を指定する場合は `--gap-results-dir <入力ディレクトリ>` と
`--gap-figures-dir <図の出力ディレクトリ>` を使えます。
これらは `--gap-normalized-only` と併用してください。
`--hessian-only` とは同時に指定できません。
モデル情報・測定結果・`h`・パラメータ数を検証するため、旧14パラメータ回路のファイルを
新しい `u3_cartan` の結果として読み込むことはできません。

## QFIMの平均log-determinant

通常の可視化では、層数に対する次の指標も描画します。既定値は `kappa = 1` です。

```text
Lbar_kappa(L) = (1/N) sum_s sum_i log(1 + kappa * lambda_i(F_L(theta_s)))
```

各ランダム点で `log det(I + kappa F)` を計算してから平均し、平均と標準誤差（SEM）を表示します。
対数は自然対数です。全固有値を使い、ランク閾値による切り捨てやパラメータ数による規格化はしません。
現行のQFIMプログラムでは各パラメータを独立な `Uniform[-pi, pi)` から標本化しています。
この指標は保存済み標本に対する期待値の推定であり、最適化経路の標本は混ぜません。

この図だけを生成する場合は、次を実行してください。

```powershell
python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1 --qfim-logdet-only
# kappaを変更する例（通常の可視化でも同じオプションを使用可能）
python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1 --qfim-logdet-only --qfim-logdet-kappa 10
```

入力は `figs/unitary_pqc_measured_1/u3_cartan/h_<h>/numerical_results/qfim/` にある
`qfim_random_points_keep0123.npz` と `qfim_random_points_keep01234.npz` です。
前者は4量子ビットの縮約状態、後者は測定結果1に条件づけた5量子ビット純粋状態のQFIMです。
測定結果・モデル・パラメータ数（1層62個）などを検証し、旧回路のデータや閾値で切り捨て済みの固有値は拒否します。
層数・標本数・固有値はアーカイブから読み込み、現在のランク閾値の設定には依存しません。

出力は同じ `h_<h>/figures/qfim/logdet/` 内の
`qfim_logdet_random_points_<keep_key>_kappa_1.pdf` と同名の `.npz` です。
NPZには標本ごとのlog-determinant、平均・標準偏差・SEM、最小最大、標本数・パラメータ数、
kappa、入力のパスとメタデータを保存します。kappaを変えるとファイル名も変わります。

専用モードはJAX・TensorCircuit・Optaxを読み込まず、学習・QFIM・Hessianを再実行しません。
保存済みQFIMがない場合はエラーで停止し、計算を自動起動しません。
`--qfim-logdet-results-dir` で上記2ファイルのあるディレクトリ、
`--qfim-logdet-figures-dir` でPDFとNPZの出力先を変更できます。
これらのディレクトリ指定は `--qfim-logdet-only` 専用です。
`--gap-normalized-only` や `--hessian-only` とは同時に指定できません。

## ランダム点QFIMのランク

DPQC・reset DPQCと同様に、層数を横軸としてQFIMランクの平均±SEMと最小・最大を表示します。
各ランダム点のランクは、保存済みの生の固有値について `lambda_i >= threshold` を満たす個数です。
閾値に等しい固有値も含めます。既定値は `config_overparam.py` の
`QFIM_EFFECTIVE_RANK_THRESHOLD`（現在 `1e-12`）で、次のように変更できます。

```powershell
python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1 --qfim-rank-only
python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1 --qfim-rank-only --qfim-rank-threshold 1e-10
```

通常の可視化にも同じ出力を追加しています。`--qfim-rank-threshold` は通常の可視化でも使用できます。
層数とサンプル数は入力アーカイブから読み込みます。各標本でランクを数えてから平均し、
SEMは標本標準偏差をサンプル数の平方根で割って求めます。
アーカイブに保存された旧閾値のランクではなく、指定閾値と生の固有値から集計します。

入力は同じ `h_<h>/numerical_results/qfim/` 内の
`qfim_random_points_keep0123.npz` と `qfim_random_points_keep01234.npz` です。
4量子ビット縮約状態と、測定結果1に条件づけた5量子ビット純粋状態について、別々の図を出力します。
モデル・測定結果・1層62パラメータなどを検証し、旧モデルや閾値で切り捨て済みの固有値は拒否します。
両ファイルを検証してから出力します。

出力先は `figs/unitary_pqc_measured_1/u3_cartan/h_<h>/figures/qfim/rank/random_points/` です。
既定閾値では `qfim_rank_mean_sem_min_max_random_points_ge_1e-12_<keep_key>.pdf` と同名の `.npz` を保存します。
NPZには標本ごとのランク、平均・標準偏差・SEM、最小最大、層数、サンプル数、パラメータ数、
閾値と入力情報を保存します。閾値を変えるとファイル名も変わります。

`--qfim-rank-only` は量子計算ライブラリを読み込まず、学習・QFIM・HS・Hessianを再実行しません。
VQE・HS・Hessianや旧形式の統合アーカイブも不要です。
`--qfim-rank-results-dir` で上記2ファイルを含むディレクトリ、
`--qfim-rank-figures-dir` でPDFとNPZの出力先を指定できます。これらは専用モードでのみ使用できます。
`--qfim-rank-only` は `--qfim-logdet-only`、`--gap-normalized-only`、`--hessian-only` と同時指定できません。
