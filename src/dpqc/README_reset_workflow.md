# Reset-DPQCの学習・解析・描画

プロジェクトのルートから実行してください。学習済みデータがある場合、学習を再実行する必要はありません。

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

保存先と既存アーカイブの形式は変更していません。`figs/dpqc_reset/h_0.1/numerical_results/` の `energy`、`qfim`、`hessian` を引き続き使用します。QFIMの層数・サンプル数・乱数シードは `src/common/config_overparam.py` を参照します。Hessianも同じ設定を既定値とし、専用ファイルでは `--layers`、`--num-samples`、`--seed-base` で変更できます。

従来の `--stage vqe`、`--stage qfim`、`--stage hessian` も利用できます。学習から全工程を実行する場合だけ `--stage all` を明示してください。既存の学習データに対応する `reset_model_metadata.json` が失われている場合は、そのモデル情報を復元してください。解析処理が学習を自動実行して作り直すことはありません。
