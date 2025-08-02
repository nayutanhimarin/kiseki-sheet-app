import streamlit as st
import pandas as pd
import datetime
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib
import numpy as np

# --- 定数と設定 ---
DATA_FILE_PREFIX = "patient_data_"
MASTER_ID = "master_admin"
MASTER_PASSWORD = "master_password_123"
PASSWORDS = {
    "hospital_a": "pass123",
    "test": "test123",
    "hospital_b": "Kiseki-sheet"
}
DISEASE_OPTIONS = ["敗血症性ショック", "心原性ショック", "心臓・大血管術後", "その他（自由記載）"]
PHASE_LABELS = ["超急性期", "維持期", "回復期", "転棟期"]
# ★★★要因タグの選択肢を定義★★★
FACTOR_TAGS = ["#循環", "#呼吸", "#意識/鎮静", "#腎/体液", "#活動/リハ", "#栄養/消化管", "#感染/炎症"]
# ★★★ イベントフラグの定義 ★★★
EVENT_FLAGS = {
    # イベント名: { カテゴリ, 色, マーカー }
    "抜管":      {"category": "#呼吸", "color": "green", "marker": "^"},
    "再挿管":    {"category": "#呼吸", "color": "red", "marker": "v"},
    "気管切開":  {"category": "#呼吸", "color": "blue", "marker": "v"},
    "SBT成功":   {"category": "#呼吸", "color": "green", "marker": "s"},
    "SBT失敗":   {"category": "#呼吸", "color": "green", "marker": "s"},
    "昇圧薬変更": {"category": "#循環", "color": "orange", "marker": "o"},
    "新規不整脈": {"category": "#循環", "color": "red", "marker": "o"},
    "出血イベント": {"category": "#循環", "color": "darkred", "marker": "o"},
    "腎代替療法開始": {"category": "#腎/体液", "color": "purple", "marker": "D"},
    "腎代替療法終了": {"category": "#腎/体液", "color": "purple", "marker": "D"},
    "せん妄":      {"category": "#意識/鎮静", "color": "magenta", "marker": "*"},
    "SAT成功":      {"category": "#意識/鎮静", "color": "magenta", "marker": "*"},
    "SAT失敗":      {"category": "#意識/鎮静", "color": "magenta", "marker": "*"},
    "新規感染症":  {"category": "#感染/炎症", "color": "brown", "marker": "X"},
    "端坐位":    {"category": "#活動/リハ", "color": "cyan", "marker": "P"},
    "立位":    {"category": "#活動/リハ", "color": "cyan", "marker": "P"},
    "歩行":    {"category": "#活動/リハ", "color": "cyan", "marker": "P"},
    "経管栄養開始": {"category": "#栄養/消化管", "color": "lime", "marker": "+"},
    "経口摂取開始": {"category": "#栄養/消化管", "color": "lime", "marker": "+"}
}
# --- 関数 ---
def load_data(filename):
    """CSVファイルを読み込む"""
    # ★★★列名に「要因タグ」を追加★★★
    columns = ["アプリ用患者ID", "日付", "時間帯", "スコア", "イベント", "ステータス", "疾患群", "要因タグ"]
    if os.path.exists(filename):
        df = pd.read_csv(filename, dtype={'イベント': str, '疾患群': str, '要因タグ': str})
        for col in columns:
            if col not in df.columns:
                df[col] = None # 古いデータファイルに新しい列を追加
        return df
    else:
        return pd.DataFrame(columns=columns)

def calculate_derived_columns(df):
    """フェーズと経過日数を計算して列を追加する関数"""
    if df.empty or 'スコア' not in df.columns or '日付' not in df.columns:
        return df.assign(フェーズ=None, 経過日数=None)
    
    df_copy = df.copy()
    df_copy['日付'] = pd.to_datetime(df_copy['日付'])
    
    bins = [-1, 20, 60, 80, 100]
    labels = PHASE_LABELS
    df_copy['フェーズ'] = pd.cut(df_copy['スコア'], bins=bins, labels=labels, right=True)

    df_copy['入室日'] = df_copy.groupby('アプリ用患者ID')['日付'].transform('min')
    df_copy['経過日数'] = (df_copy['日付'] - df_copy['入室日']).dt.days + 1
    df_copy = df_copy.drop(columns=['入室日'])
    
    return df_copy

def run_app():
    """アプリのメイン処理"""
    st.set_page_config(layout="wide")
    st.title("軌跡シート")

    # --- セッションステートの初期化 ---
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # --- ログイン管理 ---
    if not st.session_state.get('logged_in'):
        st.header("ログイン")
        facility_id_input = st.text_input("施設ID")
        password = st.text_input("パスワード", type="password")
        if st.button("ログイン"):
            if facility_id_input == MASTER_ID and password == MASTER_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.facility_id = MASTER_ID
                st.rerun()
            elif facility_id_input in PASSWORDS and PASSWORDS[facility_id_input] == password:
                st.session_state.logged_in = True
                st.session_state.facility_id = facility_id_input
                st.rerun()
            else:
                st.error("施設IDまたはパスワードが間違っています。")
    else:
        # --- ログイン後の処理 ---
        facility_id = st.session_state.facility_id
        patient_id_to_use = None
    
        # --- サイドバー (全ユーザー共通) ---
        with st.sidebar:
            st.header(f"施設ID: {facility_id}")
            
            if facility_id == MASTER_ID:
                st.subheader("管理者メニュー")
            else: # 通常ユーザー
                DATA_FILE = f"patient_data_{facility_id}.csv"
                if 'df' not in st.session_state or st.session_state.get('current_facility') != facility_id:
                    st.session_state.df = load_data(DATA_FILE)
                    st.session_state.df['ステータス'] = st.session_state.df['ステータス'].fillna('在室中')
                    st.session_state.current_facility = facility_id
                
                st.subheader("患者選択")
                active_patients = sorted(st.session_state.df[st.session_state.df['ステータス'] == '在室中']['アプリ用患者ID'].unique()) if not st.session_state.df.empty else []
                NEW_PATIENT_OPTION = "新しい患者を登録..."
                
                selected_patient = st.selectbox("表示・記録する患者IDを選択", options=[NEW_PATIENT_OPTION] + active_patients)
                
                if selected_patient == NEW_PATIENT_OPTION:
                    patient_id_to_use = st.text_input("新しいアプリ用患者IDを入力してください")
                else:
                    patient_id_to_use = selected_patient

                if patient_id_to_use:
                    st.subheader("データ入力・修正")
                    st.write(f"**対象患者:** {patient_id_to_use}")
                    disease_group_select = st.selectbox("疾患群を選択", options=DISEASE_OPTIONS)
                    disease_group = disease_group_select
                    if disease_group_select == "その他（自由記載）":
                        disease_group = st.text_input("疾患群を自由記載")
                    record_date = st.date_input("日付", datetime.date.today())
                    time_of_day = st.selectbox("時間帯", options=["朝", "夕"])
                    # 1. スコア入力UI
                    st.write("**治療スコア**")
                    if 'score_value' not in st.session_state:
                        st.session_state.score_value = 10
                    # 10点刻みのスライダーで、大まかな値を設定
                    slider_score = st.slider("大まかに調整", 0, 100, st.session_state.score_value, step=10)
                    
                    # 1点刻みの数値入力で、細かい値を調整
                    number_score = st.number_input("細かく調整", 0, 100, slider_score, step=1)
                    
                    # 最終的なスコアをセッションステートに保存
                    st.session_state.score_value = number_score
                    score = number_score

                    # 2. イベント入力UI
                    # --- ★★★ここからが今回の修正点(1)★★★ ---
                    st.write("**イベント**")
                    major_event_options = [""] + list(EVENT_FLAGS.keys())
                    selected_event = st.selectbox("主要イベントを選択（任意）", options=major_event_options)
                    
                    # 選択された主要イベントを、自由記述欄の初期値にする
                    event_text = st.text_input("イベント（自由記述も可）", value=selected_event)
                   
                    # 3. 要因タグ入力UI
                    st.write("**スコア判断の要因タグ**")
                    selected_tags = st.multiselect(
                        "スコア判断の要因タグ（複数選択可）",
                        options=FACTOR_TAGS,
                        label_visibility="collapsed"
                    )
                    
                    if st.button("記録・修正する"):
                        tags_str = ", ".join(selected_tags)
                        new_data = pd.DataFrame([{
                            "アプリ用患者ID": patient_id_to_use, 
                            "日付": str(record_date), 
                            "時間帯": time_of_day, 
                            "スコア": int(score), 
                            "イベント": event_text, 
                            "ステータス": "在室中", 
                            "疾患群": disease_group,
                            "要因タグ": tags_str
                        }])
                        st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True)
                        st.session_state.df = st.session_state.df.drop_duplicates(subset=['アプリ用患者ID', '日付', '時間帯'], keep='last').sort_values(by=["アプリ用患者ID", "日付", "時間帯"])
                        st.session_state.df.to_csv(DATA_FILE, index=False)
                        st.success("データを記録しました！")
                        st.rerun()
            
            st.write("---")
            if st.button("ログアウト"):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()

        # --- メイン画面 ---
        if facility_id == MASTER_ID:
            st.header("マスター管理者モード")
            with st.sidebar:
                st.subheader("全施設のデータ管理")
                facility_files = glob.glob(f"{DATA_FILE_PREFIX}*.csv")
                facility_ids = [os.path.basename(f).replace(DATA_FILE_PREFIX, '').replace('.csv', '') for f in facility_files]
                selected_facility = st.selectbox("施設を選択", options=facility_ids)
                if selected_facility:
                    st.session_state.current_facility = selected_facility
                    st.session_state.df = load_data(f"{DATA_FILE_PREFIX}{selected_facility}.csv")
                    st.write(f"#### {selected_facility} のデータ")
                    st.dataframe(st.session_state.df)
                

            st.subheader("全施設のアーカイブデータ")
            all_files = glob.glob(f"{DATA_FILE_PREFIX}*.csv")
            all_archived_dfs = []
            for f in all_files:
                df_temp = load_data(f)
                archived = df_temp[df_temp['ステータス'] == '退室済']
                if not archived.empty:
                    archived.insert(0, '施設ID', os.path.basename(f).replace(DATA_FILE_PREFIX, '').replace('.csv', ''))
                    all_archived_dfs.append(archived)
            
            if all_archived_dfs:
                master_df = pd.concat(all_archived_dfs, ignore_index=True)
                st.dataframe(master_df)
                csv_master = master_df.to_csv(index=False).encode('utf-8')
                st.download_button("全アーカイブデータをCSVでダウンロード", csv_master, 'master_archived_data.csv', 'text/csv')
            else:
                st.info("アーカイブされたデータを持つ施設はありません。")

        else: # 通常ユーザー
            st.header("フェーズスコア")
            if patient_id_to_use:
                display_df = st.session_state.df[st.session_state.df['アプリ用患者ID'] == patient_id_to_use].copy()
                if not display_df.empty:
                    display_df = calculate_derived_columns(display_df)
                    st.write(f"#### {patient_id_to_use} のデータ")
                    st.dataframe(display_df.sort_values(by="日付")) # ★★★データ表に要因タグが表示される★★★
                    
 # --- グラフ作成コード　★★★ここからが今回の修正点★★★ ---
                    df_graph = display_df.copy()
                    
                    # 1. 「日付」列をdatetimeオブジェクトに変換
                    df_graph['日付'] = pd.to_datetime(df_graph['日付'])
                    
                    # 2. 時間帯に応じてプロット用の日時データを作成
                    # 朝は8時、夕は20時としてプロットする
                    def create_plot_datetime(row):
                        if row['時間帯'] == '夕':
                            return row['日付'].replace(hour=20)
                        else: # 朝 または時間帯が未入力の場合
                            return row['日付'].replace(hour=8)
                    
                    df_graph['プロット用日時'] = df_graph.apply(create_plot_datetime, axis=1)
                    df_graph = df_graph.sort_values(by='プロット用日時')

                    fig, ax = plt.subplots(figsize=(12, 7))            
                    ax.plot(df_graph['プロット用日時'], df_graph['スコア'], marker='o', linestyle='-', label=patient_id_to_use)
                    
                            # ★★★ ここに、新しいイベント・フラグのコードを貼り付ける ★★★
                    events_to_plot = df_graph.dropna(subset=['イベント'])
                    if not events_to_plot.empty:
                        for idx, row in events_to_plot.iterrows():
                            event_name = row['イベント']
                            plot_time = row['プロット用日時']
                            
                            if event_name in EVENT_FLAGS:
                                # 主要イベントの場合：特別なフラグを描画
                                flag = EVENT_FLAGS[event_name]
                                ax.scatter(plot_time, row['スコア'], 
                                           color=flag['color'], 
                                           marker=flag['marker'], 
                                           s=200, 
                                           zorder=12,
                                           label=event_name)
                                ax.annotate(event_name, 
                                            xy=(plot_time, row['スコア']),
                                            xytext=(0, 15), textcoords='offset points',
                                            ha='center', va='bottom', fontsize=10,
                                            bbox=dict(boxstyle='round,pad=0.2', fc=flag['color'], alpha=0.7))
                            else:
                                # 自由記述イベントの場合：これまで通りの縦線とテキスト
                                ax.axvline(x=plot_time, color='gray', linestyle='--', linewidth=1, zorder=1)
                                ax.annotate(event_name, 
                                            xy=(plot_time, 95), 
                                            xytext=(0, 10), textcoords='offset points',
                                            ha='center', va='bottom', fontsize=12,
                                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7), zorder=11)
    
                    # --- ★★★ここからが今回の修正点(2)★★★ ---
                    # グラフの整形
                    ax.set_ylim(0, 100)
                    ax.set_title("軌跡シート", fontsize=16, pad=20) # タイトルの位置を少し上げる
                    ax.set_ylabel("フェーズスコア", fontsize=16)
                    ax.set_xlabel("日付", fontsize=16)
                    ax.tick_params(axis='both', labelsize=16)
                    plt.xticks(rotation=30, ha="right")
                    

                    
                    bbox_style = dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8)
                    ax.axhspan(0, 20, color='pink', alpha=0.3)
                    ax.axhspan(20, 60, color='bisque', alpha=0.3)
                    ax.axhspan(60, 80, color='yellow', alpha=0.3)
                    ax.axhspan(80, 100, color='lightgreen', alpha=0.3)
                    ax.text(0.02, 10, "超急性期", fontsize=24, transform=ax.get_yaxis_transform(), ha="left", va="center", bbox=bbox_style, zorder=10)
                    ax.text(0.02, 40, "維持期", fontsize=24, transform=ax.get_yaxis_transform(), ha="left", va="center", bbox=bbox_style, zorder=10)
                    ax.text(0.02, 70, "回復期", fontsize=24, transform=ax.get_yaxis_transform(), ha="left", va="center", bbox=bbox_style, zorder=10)
                    ax.text(0.02, 90, "転棟期", fontsize=24, transform=ax.get_yaxis_transform(), ha="left", va="center", bbox=bbox_style, zorder=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info(f"「{patient_id_to_use}」さんのデータはまだありません。")
            else:
                st.info("サイドバーで患者を選択または新規登録してください。")
            # --- アーカイブ管理 ---
            st.write("---")
            st.header("管理")
            show_archive = st.checkbox("アーカイブされた患者を表示")
            if show_archive:
                archived_df = st.session_state.df[st.session_state.df['ステータス'] == '退室済']
                st.write("#### 退室済（アーカイブ）患者一覧")
                st.dataframe(archived_df)
                
                # アーカイブデータのエクスポートボタン
                csv_archived = archived_df.to_csv(index=False).encode('utf-8')
                st.download_button("アーカイブ全データをCSVでダウンロード", csv_archived, f'archived_scores_{facility_id}.csv', 'text/csv')
                
                # 「在室中に戻す」ボタンのループ
                for patient_id in sorted(archived_df['アプリ用患者ID'].unique()):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(patient_id)
                    with col2:
                        if st.button("在室中に戻す", key=f"reactivate_{patient_id}"):
                            st.session_state.df.loc[st.session_state.df['アプリ用患者ID'] == patient_id, 'ステータス'] = '在室中'
                            st.session_state.df.to_csv(DATA_FILE, index=False)
                            st.success(f"{patient_id}さんを在室中に戻しました。")
                            st.rerun()

# --- プログラムの実行開始点 ---
if __name__ == "__main__":
    run_app()