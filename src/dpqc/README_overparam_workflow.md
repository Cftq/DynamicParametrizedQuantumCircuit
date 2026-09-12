# DPQCの学習・解析・描画

プロジェクトのルートから実行してください。学習済みのデータがあれば、QFIMやHessianの計算、図の描画のために学習を再実行する必要はありません。

| ファイル | 実行する処理 |
| --- | --- |
| `DPQC_overparam_vqe.py` | 変分量子回路の学習とパラメータ履歴の保存 |
| `DPQC_overparam_qfim.py` | ランダム点と保存された学習経路上のQFIM計算 |
| `DPQC_overparam_hessian.py` | ランダムなパラメータ点におけるHessian計算 |
| `DPQC_overparam_compute.py` | QFIMとHessianの計算を別プロセスで順に実行 |
| `DPQC_overparam_visualize.py` | 保存済みの学習・QFIM・Hessianデータから図を描画 |

学習が必要な場合だけ実行します。

```powershell
python src/dpqc/DPQC_overparam_vqe.py --h-param 0.1
```

QFIMとHessianはそれぞれの専用ファイルで計算できます。QFIMは学習経路上の統計量も計算するため、同じHamiltonianパラメータの保存済みVQE履歴を読み込みます。保存データが不足していても学習を自動で実行しません。Hessianのランダム点計算にはVQE履歴やQFIMの計算結果は不要です。

```powershell
python src/dpqc/DPQC_overparam_qfim.py --h-param 0.1
python src/dpqc/DPQC_overparam_hessian.py --h-param 0.1 --output-family dpqc
```

両方の解析を実行する場合は次のコマンドも使えます。**既定の処理を従来の `all` から `analysis` に変更したため、学習は実行しません。**

```powershell
python src/dpqc/DPQC_overparam_compute.py --h-param 0.1
```

必要な保存結果が揃っていれば、次のコマンドで再計算せずに描画できます。

```powershell
python src/dpqc/DPQC_overparam_visualize.py --h-param 0.1
```

保存されたHessianだけを描画する場合は次を使います。

```powershell
python src/dpqc/DPQC_overparam_visualize.py --h-param 0.1 --hessian-only --reuse-hessian-results
```

`--hessian-only` や `--with-hessian` を付けた描画では、`--reuse-hessian-results` も付けるとHessianを再計算しません。どの描画モードもVQEを実行しません。

既存の保存形式と保存先 `figs/dpqc/h_0.1/numerical_results/` 内の `energy`、`qfim`、`hessian` は維持しています。Hessianは1層あたり14個の学習パラメータに対応します。

従来の `--stage vqe`、`--stage qfim`、`--stage hessian` も利用できます。学習から全工程を実行する場合だけ `--stage all` を明示してください。`--vqe-batch-size` は学習を選択した場合だけ使用します。
