# Booler

元のクワッドポリゴン構造（エッジフロー）を極力温存したままブーリアン接続し、Catmull-Clark Subdivision（SubD）を適用しても破綻しない Blender アドオン（Blender 4.2+ / 5.x 対応）。

## 主な特徴
- **元トポロジ保持**: 交差部周辺（1〜2リング）のみを局所再構築し、それ以外の元クワッドを無変更のまま保持。
- **完全オールクワッド & 水密**: 2パス分割数マッチング＋大域最適 DP による 3-to-1 SubD リダイレクト＋Coons グリッドキャップにより、非クワッド・非多様体・交差線上の異常ポールをゼロに抑制。
- **サポート帯（Collar）& $C^1$ フィレット**: シャープ接続用のサポートループ自動生成と、`Fillet Segments` による滑らかなクワッドフィレット切替に対応。
- **Catmull-Clark Limit Surface Inverse Fitting**: Halstead et al. (1993) のリミットポイント・ステンシルにより、SubD 適用時の「痩せ・段差」を局所逆フィッティングで補正。
- **非破壊ワークフロー**: 元オブジェクトを隠しバックアップとして保持し、SubD モディファイアやマテリアルを保ったまま N-Panel からライブ編集・ソース移動後の `Rebuild`・`Apply / Bake` が可能。

## インストール方法 (Blender 4.2+ / 5.x)
1. `Booler` フォルダを ZIP 圧縮（`Booler.zip`）します。
2. Blender を開き、`Edit > Preferences > Add-ons`（または `Get Extensions` 右上メニュー）から **Install from Disk...** を選択し、`Booler.zip` を指定します。
3. 3D ビューポート右側の N-Panel（`N` キー）に **Booler** タブが表示されます。

## 基本的な使い方
1. 2 つのメッシュオブジェクトを選択します（**アクティブオブジェクト = Target (A)**、もう一方 = Cutter (B)）。
2. N-Panel の **Booler** タブから `Union` / `Diff` / `Intersect` をクリックします。
3. 生成された結果オブジェクトを選択した状態で、**Live Booler Settings** から以下をリアルタイム調整できます：
   - `Operation` / `Solver` (`EXACT`, `MANIFOLD`, `FLOAT`)
   - `Collar Width` / `Collar Rings`
   - `Fillet Segments`（`0` = シャープ接続、`1〜6` = $C^1$ フィレット）
   - `SubD Limit Fit`（極限曲面フィッティングの ON/OFF）
   - `Show Sources (Wire)` で元メッシュをワイヤーフレーム表示して位置調整し、`Rebuild` で再計算、`Apply / Bake` で確定。

## ヘッドレス・自動ベンチマーク実行
```powershell
& "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe" --background --factory-startup --python tests/run_benchmark.py
```

視覚比較用 `.blend` の再生成：
```powershell
& "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe" --background --factory-startup --python tests/export_showcase.py
```
