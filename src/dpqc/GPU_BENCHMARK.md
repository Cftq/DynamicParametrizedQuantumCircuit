# RTX 3080でのDPQC学習速度の検証

測定環境はWSL2 Ubuntu 24.04、NVIDIA GeForce RTX 3080（10GB）、JAX 0.5.3、Optax 0.2.4です。CPUとGPUには同一のPython環境、Hamiltonian h=0.1、初期パラメータ、Adam（学習率0.001）を使用しました。計算精度はfloat64/complex128のままです。

各条件は100学習ステップ・10独立試行・バッチサイズ10です。初回コンパイルと1回のウォームアップを除外し、デバイス同期を行った2回の実行時間の中央値を比較しています。CPU側は従来の行列積によるゲート演算、GPU側は今回実装した等価なRz/Rxxの要素演算を使用します。ホストへの結果転送時間も実行時間と分けて保存しています。

| 回路 | 層数 | CPU（秒） | 最適化GPU（秒） | 高速化倍率 | GPU初回コンパイル（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPQC | 4 | 0.845 | 0.218 | 3.87倍 | 4.57 |
| DPQC | 8 | 1.848 | 0.415 | 4.46倍 | 4.65 |
| Reset-DPQC | 4 | 0.872 | 0.209 | 4.18倍 | 4.13 |
| Reset-DPQC | 8 | 1.740 | 0.393 | 4.43倍 | 4.04 |

CPUとGPUの最大絶対差は、エネルギー履歴が3.997e-15、勾配ノルム履歴が5.329e-15、最終パラメータが2.388e-10でした。初期パラメータは完全一致しています。GPUでの演算順序が異なるため、ビット単位の一致を要求する比較ではありません。

最適化前に単純にGPUへ移したDPQC（4層）の実行時間は3.433秒でした。小さな行列積を置き換える今回の最適化が、GPUでの高速化に必要でした。単精度化や学習ステップの削減は行っていません。

この倍率は上記の測定条件に対する結果です。現在のconfig_overparam.pyの試行数・層数・学習ステップは変更していません。全ての層数やバッチサイズで同じ倍率になることを保証するものではありません。

再測定はプロジェクトのルートから行えます。Windowsでは設定済みWSL環境へ転送されます。

```powershell
python src/dpqc/benchmark_dpqc_vqe.py --family dpqc --device cpu --layers 4 8 --steps 100 --trials 10 --repeats 2 --output-dir figs/benchmarks/rtx3080/recheck_cpu
python src/dpqc/benchmark_dpqc_vqe.py --family dpqc --device gpu --layers 4 8 --steps 100 --trials 10 --repeats 2 --output-dir figs/benchmarks/rtx3080/recheck_gpu
```

Reset-DPQCは`--family dpqc_reset`に変更します。各実行のJSON、初期値と学習結果のNPZ、および今回の比較結果comparison.jsonは`figs/benchmarks/rtx3080/`に保存しています。ベンチマークは本番の学習アーカイブを上書きしません。

回帰テストは計89件が成功しました。デバイス選択・WSL引き継ぎ・既定の解析分離・バッチ制限に加え、ゲートの独立した行列演算との比較、状態・微分・Adam更新の一致を確認しています。

実機では両モデルを1層・2試行・3ステップで学習し、指定バッチ20が実効バッチ2になることと通常のアーカイブ保存を確認しました。両モデルのQFIM（1層・1サンプル）とreset版HessianもGPUで計算し、既存ローダーでHessianを読めることを確認しています。これらの保存テストは隔離した一時フォルダで行いました。

実際の学習・解析の実行手順は[README_gpu.md](README_gpu.md)を参照してください。
