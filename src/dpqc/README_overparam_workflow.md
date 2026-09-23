# DPQCの学習・解析・描画

プロジェクトのルートから実行してください。学習済みのデータがあれば、QFIMやHessianの計算、図の描画のために学習を再実行する必要はありません。

計算プログラムは`--device auto/cpu/gpu`に対応しています。このPCでは既定の`auto`で学習はRTX 3080、QFIM・HessianはCPUを使用します。学習する例は`python src/dpqc/DPQC_overparam_compute.py --stage vqe --h-param 0.1`です。解析だけなら`--stage vqe`を省略します。`--stage all`も学習GPU・解析CPUの順で実行し、`--device cpu/gpu`の明示指定は選択した全工程に適用します。[GPU実行・環境設定・速度比較](README_gpu.md)も参照してください。

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

## ギャップで規格化した最適化性能

通常の可視化では、層数を横軸とする `(E_final - E0) / Delta` のbeeswarm plotと、`epsilon = 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10` の成功確率も出力します。各試行の値を変えずに横方向へ配置し、層ごとの中央値を横線で示します。規格化誤差の成功判定は厳密な `< epsilon` です。全試行を分母に含め、値を1で打ち切りません。

これらの図だけを作る場合は、次のコマンドを使います。必要なのは既存の `vqe_optimization_histories.npz` だけで、QFIM・Hessianのファイルも量子計算ライブラリも不要です。

```powershell
python src/dpqc/DPQC_overparam_visualize.py --h-param 0.1 --gap-normalized-only
python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1 --gap-normalized-only
```

入力は各試行の保存済みエネルギー履歴の最終値です。初期状態を含む `steps + 1` 点の履歴と、旧形式の `steps` 点の履歴に対応します。`E0` と `Delta` は同じ `h` の4量子ビットHamiltonianの小さな行列から求めます。基底状態が縮退する場合、`Delta` は基底状態部分空間より上の最初の励起エネルギーとの差です。補助量子ビットの縮退をゼロギャップと誤認しません。

出力は `figs/dpqc/h_<h>/energy_figures/` の `final_gap_normalized_energy_error.pdf` と `success_probability_multiple_tolerances_gap_normalized.pdf` です。成功率の図は従来名 `gap_normalized_success_probability.pdf` にも同じ内容を保存します。数値は `numerical_results/energy/gap_normalized_energy_statistics.npz` です。reset DPQC の保存先は `figs/dpqc_reset/u3_cartan/h_<h>/` です。数値ファイルにはギャップ、閾値、各試行の規格化誤差、試行数、成功確率、平均・SEM・最小最大・中央値を保存します。丸め誤差による微小な負値だけを0に補正し、誤差図は0を表示できる対称対数軸（最小閾値の1/10以下は線形）を使用します。密集した点は隣の層に重ならない範囲へ横幅を圧縮しますが、試行を間引いたり縦軸の値を変更したりしません。学習結果やQFIMの入力ファイルは変更しません。

`success_probability_multiple_tolerances.pdf` は `delta = 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10` の8閾値を使います。この図は規格化前の最終エネルギー誤差を `<= delta` で判定し、規格化誤差の成功率とは別に保存します。

## ランダム点QFIMの平均log-determinant

保存済みのランダム点QFIM固有値から、横軸を層数 `L`、縦軸を `mean_theta[log det(I + kappa F)]` とした図を出力します。既定値は `kappa = 1` で、通常の可視化にも追加されています。この指標だけを作成する場合は次を実行します。

```powershell
python src/dpqc/DPQC_overparam_visualize.py --h-param 0.1 --qfim-logdet-only
python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1 --qfim-logdet-only
```

`--qfim-logdet-kappa 10` のように有限の正の値を指定して変更できます。入力は `numerical_results/qfim/qfim_random_points_keep0123.npz` と `qfim_random_points_keep01234.npz` の保存済み固有値です。専用モードはQFIM、学習、Hessianを再実行せず、VQE履歴や量子計算ライブラリを必要としません。resetでは現在のモデルを示す `reset_model_metadata.json` も検証します。

各ランダム点で `sum_i log1p(kappa * lambda_i)` を計算し、その後に点間の算術平均を取ります。自然対数を用い、QFIMランクの閾値やパラメータ数による規格化は適用しません。微小な正の固有値も含めます。`p(theta)` は保存したサンプルの分布であり、既存の生成プログラムでは各角度が独立な `[-pi, pi)` の一様分布です。

出力先は `figs/dpqc/h_<h>/figures/qfim/logdet/`（resetは `figs/dpqc_reset/u3_cartan/h_<h>/figures/qfim/logdet/`）です。各部分系について `qfim_logdet_random_points_keep0123_kappa_1.pdf` と同名の `.npz`、`keep01234` の組を保存します。図は平均±SEMを示し、数値ファイルには各点の値、最小・最大を含む統計量、層数、サンプル数、パラメータ数、κ、入力ファイルの情報を保存します。κを変えると別名で保存されます。`--qfim-logdet-only` は `--gap-normalized-only`、`--hessian-only`、`--with-hessian` と同時指定できません。
