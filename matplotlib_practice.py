import pandas as pd
import matplotlib.pyplot as plt # Matplotlibをpltというニックネームで呼び出す

# 前回のレッスンと同じデータ
data = {
    '患者ID': ['ICU-001', 'ICU-002', 'ICU-003'],
    '治療スコア': [85, 92, 78]
}

# Pandasのデータフレームを作成
df = pd.DataFrame(data)

# Matplotlibを使って、データフレームから棒グラフを作成
plt.bar(df['患者ID'], df['治療スコア'])

# グラフにタイトルを追加
plt.title("Patient Scores")

# 作成したグラフを表示する
plt.show()