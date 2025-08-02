# 点数リストを受け取って、平均点を計算して返す関数
def calculate_average(score_list):
    total = sum(score_list) # sum()はリストの合計を計算する便利な組み込み関数
    average = total / len(score_list)  # len()はリストの個数を数える便利な組み込み関数
    return average

# ---ここからが実際の処理---

# 2つの患者のグループリスト
group_a_scores = [85, 92, 78]
group_b_scores = [100, 64, 88, 95]

# 上で定義した関数を呼び出して、平均点を計算する
group_a_average = calculate_average(group_a_scores)
group_b_average = calculate_average(group_b_scores)


print("グループAの平均点:", group_a_average)
print("グループBの平均点:", group_b_average)
