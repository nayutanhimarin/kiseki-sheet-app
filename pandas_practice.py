import pandas as pd

# 辞書（dictionary）を使ってデータを作成
data = {
    '患者ID': ['ICU-001', 'ICU-002', 'ICU-003'],
    '治療スコア': [85, 92, 78]
}

# Pandasのデータフレーム（データ表）を作成
df = pd.DataFrame(data)

# データフレームを表示
print(df)