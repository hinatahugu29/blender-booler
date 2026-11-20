# Booler Research Notes

各開発フェーズにおける Research Gate（論文・既存技術・Blender API・GitHub OSS 調査）の検証ログ。

---

## Research Gate 0: Blender 環境 & Boolean 属性引き継ぎ検証 (2026-11-20)

### 検証環境
- **Blender**: `5.3.0` (Steam版: `C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe`)
- **Python**: `3.13.13`
- **標準同梱パッケージ**: `numpy 2.3.4`, `mathutils.geometry.delaunay_2d_cdt` 利用可能
- **未同梱**: `scipy`（Pure Python + NumPy ベースを基本とし、必要時のみ検討）

### Boolean API & 属性引き継ぎ実験結果
- `bmesh.ops` に直接の boolean 関数は存在しないため、**Boolean Modifier + `depsgraph` 評価**（非破壊的に元メッシュを汚染せず評価可能）を採用。
- 利用可能ソルバー: `FLOAT`, `EXACT`, `MANIFOLD`（デフォルト推奨: `EXACT`、高速化オプションとして `MANIFOLD`）。
- 入力メッシュ A (id=1), B (id=2) の `Mesh.attributes`（`INT`, `FACE` ドメイン）に `booler_orig_obj` および `booler_orig_face` を設定して評価した結果、**全ソルバーで完全に属性が引き継がれる**ことを確認。
- 交差エッジは `len(e.link_faces) == 2 and e.link_faces[0][orig_obj] != e.link_faces[1][orig_obj]` で 100% 確実に抽出できる。
- また、元メッシュ A/B の各フェイス ID (`booler_orig_face`) も残るため、「元のクワッドが分断されずにそのまま残っているか」および「元メッシュのどのフェイスが交差の影響を受けたか」を正確に追跡できる。

---

## Research Gate 1: オフセット帯（Collar）とパリティ保証の理論 (2026-11-20)

### 数学的発見：オールクワッド元メッシュの境界パリティ定理
- 閉じたオールクワッド多様体メッシュから任意のクワッド部分集合 $D$ を削除したとき、残る境界エッジの総数は $4|D| - 2|E_{\text{int}}|$ となり**常に偶数**になる。
- したがって、交差ループのリサンプリング分割数 $m$ を**偶数に制約するだけ**で、Object A 側・Object B 側それぞれの隙間領域（外周境界＋内周境界）の総辺数が **100% 確実に偶数**となり、オールクワッド充填の必要条件を満たす。

### オフセット曲線生成の比較
- **簡易法（Phase 1 採用）**: 交差線上の各点でブーリアン結果の残存側フェイス方向へ接平面オフセットし、元メッシュの `BVHTree.find_nearest` で表面再投影＋接平面緩和を行う。Union/Difference/Intersect すべてで法線・向きが自動整合し高速。
- **拡張法（必要時 Phase 3 で検討）**: 曲率が極端に急で自己交差が起きるケースには、Crane et al. (2013) の **Heat Method** または Fast Marching による測地距離場の等高線（Isoline）へ差し替え可能なインターフェースとする。

### 局所クワッド化 OSS 調査（Phase 2 向け先取り）
- **Blossom-Quad (Remacle 2012)**: 双対グラフの最小重み完全マッチング（Edmonds' Blossom）。`gmsh` で標準採用。Python では Joris van Rantwijk の純 Python Blossom アルゴリズム（`networkx` 採用アルゴリズム）が外部依存ゼロで実装可能。

---

## Research Gate 2: 局所クワッド充填（3-to-1 Reduction vs Blossom-Quad）(2026-11-20)

### 1. 単一リング（アニュラス）ギャップに対する 3-to-1 U字リダイレクト理論
- 内周ループ $I$（辺数 $m$、偶数）と外周ループ $O$（辺数 $n$、偶数）の差は常に偶数 $|n - m| = 2k$。
- SubD モデリングの王道リダイレクトである **3-to-1 U字ブロック**（多い側の連続 3 辺を、内部 2 頂点・4 枚のクワッドを介して少ない側の 1 辺へ縮約）を $k$ 箇所に配置することで：
  - **100% オールクワッド（Tri=0, N-gon=0）・100% 多様体（Non-Manifold=0）** が数学的に保証される。
  - 価数 3 のポールは隙間内部に、価数 5 のポールはサポート帯外縁（交差線から 1 ループ外側の安全地帯）に配置され、**交差線上は全頂点 価数 4 を厳守**。
  - 密度比が 3:1 を超える場合も、中間リング（長さ $m' = \text{round\_even}((m+n)/2)$）を自動挿入して多段化すれば任意比率に対応可能。
  - 開始位相（ねじれ角）は 3D 距離二乗和最小化で一意に決定でき、3-to-1 の配置位置は「多い側の局所エッジ長が短い（密な）場所」に優先配分することでアスペクト比とエッジフローを最適化できる。

### 2. Blossom-Quad (Remacle et al., 2012) の一般領域への位置づけ
- 単一リング以外の **多穴領域（複数交差ループの切り抜きが合体した場合）** や非定型パッチに対しては、`delaunay_2d_cdt` ＋ Blossom-Quad（双対グラフの最小重み完全マッチング＋高コスト Extra Edge によるフォールバック）を適用するハイブリッド戦略とする。

---

## Research Gate 3: トポロジ最適化・$C^1$ フィレット・Catmull-Clark 極限曲面フィッティング (2026-11-20)

### 1. クワッド歪み（蝶ネクタイ・極小内角）の根本原因と対策
- 実測診断の結果、Collar（サポート帯）側は全クワッドが良好（$\ge 25^\circ$）であり、極小内角は 100% Bridge 側で発生していた。
- 原因は 2 つ：(a) 2つの対角三角形法線比較だけでは平面上の蝶ネクタイ反転を見逃すケースがあること（辺ベクトル進行方向の内積 `(d1 - d0) · (s1 - s0) > 0` と最小内角コストの追加で完全解消）、(b) 外周グリッド境界の凸凹角に対してサポート帯外縁が固定されすぎていたこと（`topology_opt.py` の接平面緩和で解消）。

### 2. Catmull-Clark Limit Surface Inverse Fitting (Halstead et al., SIGGRAPH 1993)
- 価数 $k$ の頂点 $v$ の Catmull-Clark 極限位置（Limit Point Stencil）は：
  $$v^\infty = \frac{k^2 v + 4 \sum_{i=1}^k e_i + \sum_{i=1}^k f_i}{k(k+5)}$$
- 局所生成頂点のみを対象に $v^{(t+1)} = v^{(t)} + \omega (p_{\text{target}}(v^{\infty,(t)}) - v^{\infty,(t)})$ を反復することで、外部疎行列ソルバーなしで SubD (Lv.2+) 適用時の「痩せ・段差」を補正できる。

### 3. $C^1$ 連続フィレット生成（Cubic Hermite Cross-Section Blending）
- 交差ループの各点において、Side A 外縁 $(p_A, \mathbf{t}_A)$ と Side B 外縁 $(p_B, \mathbf{t}_B)$ を 3 次エルミート曲線で補間することで、両側の元曲面と厳密に $C^1$ 連続なフィレット帯を任意のセグメント数で生成可能。

---

## Research Gate 4: 非破壊ワークフロー & エッジケース対応設計 (2026-11-20)

### 1. Blender Python における非破壊ワークフロー設計
- 結果オブジェクトに `PropertyGroup` (`obj.booler`) を紐づけ、`source_a`, `source_b`（`PointerProperty(type=bpy.types.Object)`）と全パラメータを保持する。
- パラメータ変更時は結果オブジェクト自体を削除・再作成するのではなく、**`bm.to_mesh(result_obj.data)` でメッシュデータのみをインプレース更新**する。これにより、ユーザーが結果オブジェクトに追加した **Subdivision Surface モディファイア・マテリアル・トランスフォーム・親子関係が完全に保持**される。
- 元オブジェクトは非表示（または Wireframe プレビュー切替可能）で保持し、位置調整後の `Rebuild` や、リンク解除する `Bake / Apply` オペレーターを提供する。

### 2. 実務エッジケースへの堅牢化
- **非オールクワッド入力（奇数パリティ）へのフォールバック**: 元メッシュに三角形が混ざり外周辺数が奇数になった場合でも、穴（Non-Manifold）を残さず最小 1 枚の三角形（2-to-1 の片側三角キャップ）で安全に閉じて水密性を保証する。
- **非均一スケール・回転・連続ブーリアン**: `matrix_world` の相対変換と既存 `booler_*` 属性の上書きクリーンアップを徹底し、Booler 結果に対してさらに別メッシュを連続ブーリアンできるようにする。
