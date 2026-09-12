# 測定結果1のUnitary-PQC：学習・解析・描画

プロジェクトのルートから実行してください。既存の学習データがある場合、学習の再実行は不要です。

| ファイル | 処理 |
| --- | --- |
| `unitary_pqc_measured_1_overparam_vqe.py` | 学習し、パラメータ履歴・エネルギーなどを保存 |
| `unitary_pqc_measured_1_overparam_qfim.py` | ランダム点と保存済み学習経路のQFIM・HSを計算 |
| `unitary_pqc_measured_1_overparam_hessian.py` | ランダム点のHessianを計算 |
| `unitary_pqc_measured_1_overparam_compute.py` | QFIM・HS、Hessianを別プロセスで順に実行 |
| `unitary_pqc_measured_1_overparam_visualize.py` | 保存結果を読み込み、図を作成 |
| `unitary_pqc_measured_1_overparam_common.py` | 共通の回路定義・状態・保存形式・描画補助 |
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

保存先は引き続き`figs/unitary_pqc_measured_1/h_0.1/numerical_results/`の`energy`、`qfim`、`hs`、`hessian`です。回路と保存形式は維持しています。全体の描画には学習経路のQFIM・HSも必要なので、`--random-only`のみで作成した結果だけでは全体の図は揃いません。

`--vqe-batch-size`は学習専用、`--analysis-batch-size`はQFIM・HSとHessian用です。層数・サンプル数・乱数シードなどは`src/common/config_overparam.py`を参照します。

従来の明示的な`--stage vqe`と`--stage qfim`に加え、`--stage hessian`が使えます。`--stage qfim`はQFIM・HSだけを計算します。以前この処理に含まれていたHessianは専用ファイルまたは`--stage analysis`で計算してください。学習から全工程を実行する場合だけ`--stage all`を指定します。

既存のPython API `run_unitary_pqc_overparam()`は、明示的に学習から全工程を実行する関数として維持しています。解析のみの場合は、専用ファイルの`run_unitary_pqc_qfim_stage()`と`run_unitary_pqc_hessian_stage()`を使用してください。

旧computeファイル経由での数値関数呼出しと結果参照は引き続き利用できます。独自スクリプトでモジュールの共有状態へ直接代入する場合は、`unitary_pqc_measured_1_overparam_common`をimportして変更してください。設定値の変更には引き続き`config_overparam.py`を使用します。
