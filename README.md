# myDeZero

## Progress
7/23 
 - step01: Variable classの実装
 - step02: Function, Square classの実装
 - step03: Exp classの実装
 - step04: 中心差分近似による数値微分の実装
 - step05: バックプロパゲーションのテキスト説明
 - step06: バックプロパゲーションの実装
 - step07: リンク付きノードの実装によるバックプロパゲーションの自動化
 - step08: バックプロパゲーションを再帰処理からループ処理へ変更
 - step09: Functionの関数化、backwardメソッドの簡略化、ndarray型のみ利用できるよう厳格化
 - step10: 勾配確認に関するユニットテストの作成

7/24
 - step11: 可変長の引数に対応するためにFunctionクラスの修正
 - step12: 可変長の引数に対応するためにFunctionクラスの更なる改善
 - step13: バックプロパゲーション処理の可変長引数への対応
 - step14: 同一の変数を利用したバックプロパゲーションへの対応
 - step15: 誤差逆伝播法の処理優先度に関する現状の問題についてテキスト説明
 - step16: 複雑な計算グラフに対応するためにGenerationの追加とBP処理への適用
 - step17: 参照カウント方式の理解とweakrefを使った循環参照問題の解決
 - step18: 勾配データの保持についてモード切り替えによって制御する（順伝播の場合は保持しない）
 - step19: Variableにndarrayのインスタンス変数を設定＆len(),print()への対応
 - step20: 演算子のオーバーロード1（Variableインスタンス同士の演算に対応）
 - step21: 演算子のオーバーロード2（int,floatとの混合演算、左項ndarrayのケースへの対応）
 - step22: 演算子のオーバーロード3（減算、除算、乗算、負数に対応）
 - step23: dezeroのパッケージ化（__init__.py, core_simple.pyを作成）
 - step24: dezeroを使った複雑な関数の微分を計算

7/27
 - step25: graghVizのインストールと使い方についてテキスト説明

7/28
 - step26: 計算をDOT言語に変換して計算グラフの画像出力までの自動化