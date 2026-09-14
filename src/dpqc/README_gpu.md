# DPQC・Reset-DPQCのGPU実行

学習は搭載されているNVIDIA GeForce RTX 3080（10GB）、QFIM・Hessian解析はCPUで実行するのが既定の動作です。どちらもWSL2の`Ubuntu-24.04`内の同じJAX環境を使用します。このPCでは専用環境の作成と、GPU上のfloat64演算の確認が完了しています。以下はプロジェクトのルートで実行してください。

## 学習と解析

学習を実行する場合は、`--stage vqe`を明示します。回路の評価、勾配計算、Adamの更新をJAXの`jit`・`vmap`・`lax.scan`でまとめ、選択したデバイス上で実行します。

```powershell
python src/dpqc/DPQC_overparam_compute.py --stage vqe --h-param 0.1
python src/dpqc/DPQC_overparam_reset_compute.py --stage vqe --h-param 0.1
```

保存済みの学習結果を使ってQFIM・Hessianを計算する場合は、次を実行します。**既定の`--stage analysis`は学習を実行しません。** Reset-DPQCのランダム点解析と両モデルのHessian計算には、学習結果自体も不要です。

```powershell
python src/dpqc/DPQC_overparam_compute.py --h-param 0.1
python src/dpqc/DPQC_overparam_reset_compute.py --h-param 0.1
```

学習から解析まで実行する場合だけ`--stage all`を指定します。`--device`を省略すれば学習だけGPU、後続のQFIM・HessianはCPUになります。各工程を別プロセスで順に実行するため、学習プロセス終了時にGPUメモリが解放されます。専用ファイルを直接実行する場合も同じ既定値です。

```powershell
python src/dpqc/DPQC_overparam_compute.py --stage all --h-param 0.1
python src/dpqc/DPQC_overparam_reset_compute.py --stage all --h-param 0.1
```

| 処理 | DPQC | Reset-DPQC |
| --- | --- | --- |
| 学習 | `DPQC_overparam_vqe.py` | `DPQC_overparam_reset_vqe.py` |
| QFIM | `DPQC_overparam_qfim.py` | `DPQC_overparam_reset_qfim.py` |
| Hessian | `DPQC_overparam_hessian.py` | `DPQC_overparam_reset_hessian.py` |
| 工程の選択 | `DPQC_overparam_compute.py` | `DPQC_overparam_reset_compute.py` |

## デバイスの選択

| 指定 | このPCの設定での動作 |
| --- | --- |
| `--device auto`（既定） | 設定済みWSL環境で学習はGPU、QFIM・HessianはCPU |
| `--device gpu` | 選択したすべての工程でGPUを必須とし、使用できなければ計算開始前にエラー |
| `--device cpu` | 選択したすべての工程を同じWSL環境のCPUで実行 |

実行時の`[DPQC] JAX backend: ...; devices: ...`に、実際に選択されたデバイスが表示されます。`gpu`を明示した処理がCPUに切り替わることはありません。

`--device`を省略したときは環境変数`DPQC_DEVICE`も利用できます。`DPQC_DEVICE=gpu`が設定されている場合は解析もGPUになるため、工程ごとの自動選択を使うには変数を削除するか`--device auto`を明示してください。WSL/Linux上の学習の`auto`はJAXが選ぶ既定デバイスと外部のJAXプラットフォーム設定に従います。解析の`auto`はCPUを指定します。

CPU解析ではCUDAカーネルを生成しないため、QFIM・HessianでのGPU共有メモリ上限やVRAM不足によるエラーを避けられます。解析時間はCPU性能に依存し、計算規模が大きい場合のシステムRAM不足まで防ぐ設定ではありません。

WindowsのPythonを直接使用する場合は、PowerShellで次のようにWSLへの引き継ぎを無効にできます。そのPython環境には数値計算の依存パッケージが必要です。

```powershell
$env:DPQC_USE_WSL = "0"
python src/dpqc/DPQC_overparam_compute.py --stage hessian --h-param 0.1 --device cpu
Remove-Item Env:DPQC_USE_WSL
```

## 数値精度・バッチ・保存先

実数はfloat64、複素数はcomplex128を維持しています。回路、初期パラメータの乱数キー、学習率、最適化手法などは既存の`src/common/config_overparam.py`に従います。GPUの選択によってAdam以外の設定をAdamへ変更することはありません。CPUとGPUでは浮動小数点演算の順序による小さな差が生じるため、比較には許容誤差を用います。

GPU上の学習では、RzとRxxによる密度行列の更新に専用の演算を自動で使用します。Rzは各行・列の位相の積、RxxはXORによる基底インデックスの置換と要素ごとの演算で計算します。どちらも密度行列全体に対する元の`U rho U†`と同じ演算をcomplex128で実装し、小さな行列積を繰り返す処理を減らしています。学習アルゴリズムや数値精度の変更はありません。CPUでは既定で従来の行列演算を使用します。

演算方式の比較・再現が必要な場合は、環境変数`DPQC_DENSITY_KERNEL`に`generic`（従来の行列演算）または`elementwise`（専用の演算）を指定できます。Windowsで指定した値はWSLへ引き継がれます。変数を削除するか`auto`を指定すると、GPUでは`elementwise`、CPUでは`generic`を選択します。実際の方式は`[DPQC] VQE density kernel: ...`に表示されます。

`--vqe-batch-size`は同時に計算する試行数です。指定値が総試行数`NUM_RUNS`より大きい場合は、実効バッチサイズを`NUM_RUNS`に制限して余分な試行を計算しません。末尾の不完全なバッチには従来どおりパディングを行い、有効な試行だけ保存します。学習アーカイブには`vqe_batch_size`と`requested_vqe_batch_size`を追加して記録します。GPUメモリは、利用者が別のJAX設定を指定していなければ必要に応じて確保します。

WSLは`/mnt/c/`を通じて同じプロジェクトを使用します。保存先は従来どおり`figs/dpqc/h_<h>/numerical_results/`と`figs/dpqc_reset/h_<h>/numerical_results/`です。保存済み結果の描画に学習の再実行は必要ありません。

## 深い回路のQFIM計算

QFIMの微分方向は、`config_overparam.py`の`RED_JVP_CHUNK`（既定16）ごとに`jax.lax.map`で処理します。以前のPythonループをJITで展開する方式では、Reset-DPQCの44層でGPUカーネルの共有メモリが上限を超え、`ptxas ... uses too much shared data`で停止しました。現在は同じ微分処理を繰り返す構造にして、このコンパイルエラーを解消しています。SLD-QFIMの定義、倍精度、サンプル数、乱数シードは維持しています。この修正は通常DPQCとReset-DPQC、ランダム点と保存済み学習経路のQFIMに適用されます。

RTX 3080で両モデルの1層・44層を各2サンプル計算し、既存形式での保存まで確認しました。Reset-DPQCの44層では、別の固定パラメータ点で従来CPU実装とのQFIM行列要素の最大絶対差が`3.75e-16`以下でした。

上記のエラーで停止したReset-DPQCのQFIMは、次のコマンドで再実行できます。VQE学習は実行しません。QFIMの途中再開機能はないため、設定された層・サンプルを最初から計算し、完了時に同じ保存先へ書き出します。

```powershell
python src/dpqc/DPQC_overparam_reset_compute.py --stage qfim --h-param 0.1
```

`lax.map`のコンパイルとメモリ使用量については[JAX公式ドキュメント](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html)を参照してください。

## 環境の再作成

WindowsからWSLへ引き継ぐ設定は、Gitの管理対象外の`.dpqc-gpu-wsl.json`に保存されます。専用Python環境はWSL内の`~/.venvs/dpqc-gpu`です。通常の実行時に再セットアップする必要はありません。

別の環境に再作成する場合は、WSL内でプロジェクトのルートに移動し、次を実行します。

```bash
python3 src/common/setup_dpqc_gpu_wsl.py
```

セットアップは`src/common/requirements-dpqc-gpu.txt`から専用環境に依存パッケージをインストールし、GPUとfloat64演算を検証してからWindows用設定を書き込みます。既にインストール済みの環境を検証・登録する場合は`--skip-install`を指定できます。

## 実際のAdam処理の速度比較

`benchmark_dpqc_vqe.py`は本番と同じ学習ループを使い、同じfloat64初期値でCPU・GPUを比較します。コンパイル、コンパイル後の実行、CPUへの結果転送をそれぞれ同期して計測します。結果は指定ディレクトリにJSONとNPZで保存し、本番の学習結果は上書きしません。

```powershell
python src/dpqc/benchmark_dpqc_vqe.py --family dpqc --device cpu --layers 4 8 --steps 100 --trials 10 --repeats 3 --output-dir tmp/dpqc_gpu_benchmark
python src/dpqc/benchmark_dpqc_vqe.py --family dpqc --device gpu --layers 4 8 --steps 100 --trials 10 --repeats 3 --output-dir tmp/dpqc_gpu_benchmark
```

Reset-DPQCの比較は`--family dpqc_reset`を指定します。上の100ステップ・10試行などはベンチマークのプロセス内だけで使う値であり、本番の設定ファイルや現在の30試行という設定は変更しません。高速化の程度は層数・試行数・精度などに依存するため、JSON内の`median_seconds`を同条件で比較してください。この値はコンパイル後の実行時間です。初回実行を含めた所要時間を評価する場合は、別項目の`compile_seconds`も確認します。

RTX 3080での実測結果と測定条件は[GPUベンチマーク結果](GPU_BENCHMARK.md)にまとめています。

学習・解析・描画の詳細は[DPQCの手順](README_overparam_workflow.md)と[Reset-DPQCの手順](README_reset_workflow.md)を参照してください。
