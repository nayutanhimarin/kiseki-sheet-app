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

# 新しいデータ構造に対応する列名を定義
# これまでの「要因タグ」も互換性のために残しておきます
NEW_COLUMN_NAMES = [
    "アプリ用患者ID", "日付", "時間帯", 
    "総合スコア",  # 「スコア」から「総合スコア」に名称変更
    "循環スコア", "呼吸スコア", "意識_鎮静スコア", "腎_体液スコア", 
    "活動_リハスコア", "栄養_消化管スコア", "感染_炎症スコア",
    "イベント", "ステータス", "疾患群", "要因タグ"
]
# ★★★ イベントフラグの定義 ★★★
EVENT_FLAGS = {
    # イベント名: { カテゴリ, 色, マーカー }
    "入室":      {"category": "#その他", "color": "red", "marker": "s"},
    "再手術":    {"category": "#その他", "color": "darkred", "marker": "X"},
    "転棟":      {"category": "#その他", "color": "blue", "marker": "s"},
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
    if not os.path.exists(filename):
        return pd.DataFrame(columns=NEW_COLUMN_NAMES)
    df = pd.read_csv(filename, dtype={'イベント': str, '疾患群': str, '要因タグ': str})
    if 'スコア' in df.columns:
        df = df.rename(columns={'スコア': '総合スコア'})
    for col in NEW_COLUMN_NAMES:
        if col not in df.columns:
            df[col] = None
    return df

# ★★★ ステップ1の変更点(3) ★★★
def calculate_derived_columns(df):
    if df.empty or '総合スコア' not in df.columns or '日付' not in df.columns:
        return df.assign(フェーズ=None, 経過日数=None)
    df_copy = df.copy()
    df_copy['日付'] = pd.to_datetime(df_copy['日付'])
    bins = [-1, 20, 60, 80, 100]
    labels = PHASE_LABELS
    df_copy['フェーズ'] = pd.cut(df_copy['総合スコア'], bins=bins, labels=labels, right=True)
    df_copy['入室日'] = df_copy.groupby('アプリ用患者ID')['日付'].transform('min')
    df_copy['経過日数'] = (df_copy['日付'] - df_copy['入室日']).dt.days + 1
    df_copy = df_copy.drop(columns=['入室日'])
    return df_copy

def run_app():
    st.set_page_config(layout="wide")
    st.title("軌跡シート")

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False

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
        facility_id = st.session_state.facility_id
        patient_id_to_use = None
    
        with st.sidebar:
            st.header(f"施設ID: {facility_id}")
            
            if facility_id != MASTER_ID:
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
                    
                    # ★★★ ステップ2の変更点(1) ★★★
                    # 入力インターフェースの刷新
                    st.write("---")
                    st.write("**多職種スコア入力**")

                    # スコアリングガイドラインの定義
                    guidelines = {
                        "循環スコア": """
                            - **0-20:** 昇圧薬(高用量) or 補助循環(ECMO/Impella)導入 or 致死的不整脈
                            - **21-40:** 昇圧薬(中等量) or 補助循環化に安定 
                            - **41-60:** 昇圧薬(少量) or 補助循環weaning
                            - **61-80:** 昇圧薬離脱 or 補助循環終了
                            - **81-100:** 循環動態が安定
                        """,
                        "呼吸スコア": """
                            - **0-20:** 挿管中、高い呼吸器設定
                            - **21-40:** 自発呼吸モード、低い呼吸器設定、非挿管だが頻呼吸
                            - **41-60:** SBT成功～抜管
                            - **61-80:** 抜管～HFNC/NPPV離脱
                            - **81-100:** 経鼻酸素～酸素なしで安定
                        """,
                        "意識_鎮静スコア": """
                            - **0-20:** 深い鎮静(RASS-4~-5) or 意識障害
                            - **21-40:** 浅い鎮静(RASS-1~-3) or せん妄
                            - **41-60:** SAT成功
                            - **61-80:** 会話可能 or 良好な筆談
                            - **81-100:** 意識清明、良好な睡眠
                        """,
                        "腎_体液スコア": """
                            - **0-20:** 大量輸液・輸血が必要
                            - **21-40:** 大量輸液は不要だが除水はできず
                            - **41-60:** バランス±0～-500mL/dayほどの緩徐なマイナスバランス
                            - **61-80:** refilling、積極的な除水
                            - **81-100:** 適正体重への除水達成
                        """,
                        "活動_リハスコア": """
                            - **0-20:** 体位変換にも制限、ROM訓練のみ
                            - **21-40:** ベッド上安静（ギャッジアップなど）
                            - **41-60:** 端座位達成
                            - **61-80:** 立位達成
                            - **81-100:** 室内歩行開始
                        """,
                        "栄養_消化管スコア": """
                            - **0-20:** 絶食、消化管トラブルあり
                            - **21-40:** 経腸栄養(少量)開始
                            - **41-60:** 経腸栄養を増量中
                            - **61-80:** 目標カロリー達成、経口摂取開始
                            - **81-100:** 経口摂取が自立
                        """,
                        "感染_炎症スコア": """
                            - **0-20:** 敗血症性ショック
                            - **21-40:** マーカー高値だがIL-6、PCT peak out
                            - **41-60:** 解熱、CRPもpeak out
                            - **61-80:** 抗菌薬のDe-escalation済み、CRP<10mg/dL
                            - **81-100:** 抗菌薬終了、炎症反応正常化
                        """
                    }
                    
                    # 各要因スコアのスライダーとポップオーバーを作成
                    factor_scores = {}
                    for score_name, guideline_text in guidelines.items():
                        col1, col2 = st.columns([0.8, 0.2])
                        with col1:
                            factor_scores[score_name] = st.slider(score_name, 0, 100, 50)
                        with col2:
                            st.popover("?", use_container_width=True).markdown(guideline_text)

                    # 総合スコアの入力
                    st.write("---")
                    st.write("**ICU医師 最終判断**")
                    total_score = st.slider("総合スコア", 0, 100, 50)
                    
                    # イベント入力UI
                    st.write("---")
                    st.write("**イベント**")
                    major_event_options = [""] + list(EVENT_FLAGS.keys())
                    selected_event = st.selectbox("主要イベントを選択（任意）", options=major_event_options)
                    event_text = st.text_input("イベント（自由記述も可）", value=selected_event)
                   
                    if st.button("記録・修正する"):
                        # ★★★ ステップ2の変更点(2) ★★★
                        # 新しいデータ構造でDataFrameを作成
                        new_data_dict = {
                            "アプリ用患者ID": patient_id_to_use, 
                            "日付": str(record_date), 
                            "時間帯": time_of_day, 
                            "総合スコア": int(total_score),
                            "イベント": event_text, 
                            "ステータス": "在室中", 
                            "疾患群": disease_group,
                            "要因タグ": "" # 将来的に削除予定
                        }
                        # 各要因スコアを辞書に追加
                        for score_name, score_value in factor_scores.items():
                            new_data_dict[score_name] = int(score_value)

                        new_data = pd.DataFrame([new_data_dict])
                        
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
            if patient_id_to_use:
                display_df = st.session_state.df[st.session_state.df['アプリ用患者ID'] == patient_id_to_use].copy()
                if not display_df.empty:
                    st.header(f"患者: {patient_id_to_use}")
                    display_df_with_calc = calculate_derived_columns(display_df)
                    st.dataframe(display_df_with_calc)
                    
                    df_graph = display_df.copy()
                    df_graph['日付'] = pd.to_datetime(df_graph['日付'])
                    
                    def create_plot_datetime(row):
                        if row['時間帯'] == '夕': return row['日付'].replace(hour=20)
                        else: return row['日付'].replace(hour=8)
                    
                    df_graph['プロット用日時'] = df_graph.apply(create_plot_datetime, axis=1)
                    df_graph = df_graph.sort_values(by='プロット用日時')

                    fig, ax = plt.subplots(figsize=(12, 7))            
                    ax.plot(df_graph['プロット用日時'], df_graph['総合スコア'], marker='o', linestyle='-', label=patient_id_to_use)
                    
                    events_to_plot = df_graph.dropna(subset=['イベント']).drop_duplicates(subset=['プロット用日時', 'イベント'])
                    if not events_to_plot.empty:
                        for idx, row in events_to_plot.iterrows():
                            event_name = row['イベント']
                            plot_time = row['プロット用日時']
                            plot_score = row['総合スコア']
                            if pd.isna(plot_score): continue # スコアがNaNの場合はプロットしない
                            
                            flag = EVENT_FLAGS.get(event_name)
                            if flag:
                                ax.scatter(plot_time, plot_score, color=flag['color'], marker=flag['marker'], s=200, zorder=12, label=event_name)
                                ax.annotate(event_name, xy=(plot_time, plot_score), xytext=(0, 15), textcoords='offset points', ha='center', va='bottom', fontsize=10, bbox=dict(boxstyle='round,pad=0.2', fc=flag['color'], alpha=0.7))
                    
                    # ★★★ 問題点修正(1) X軸のフォーマットを修正 ★★★
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # 1日ごとに目盛り
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d')) # 月/日の形式で表示
                    fig.autofmt_xdate(rotation=45) # ラベルを斜めにして見やすくする

                    ax.set_ylim(-5, 105) # 少し余裕を持たせる
                    ax.set_title("軌跡シート", fontsize=16, pad=20)
                    ax.set_ylabel("フェーズスコア", fontsize=12)
                    ax.set_xlabel("日付", fontsize=12)
                    
                    bbox_style = dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8)
                    ax.axhspan(0, 20, color='pink', alpha=0.3)
                    ax.axhspan(20, 60, color='bisque', alpha=0.3)
                    ax.axhspan(60, 80, color='yellow', alpha=0.3)
                    ax.axhspan(80, 100, color='lightgreen', alpha=0.3)
                    y_text_pos = 0.02
                    ax.text(y_text_pos, 0.1, "超急性期", fontsize=14, transform=ax.transAxes, ha="left", va="center", bbox=bbox_style, zorder=10)
                    ax.text(y_text_pos, 0.4, "維持期", fontsize=14, transform=ax.transAxes, ha="left", va="center", bbox=bbox_style, zorder=10)
                    ax.text(y_text_pos, 0.7, "回復期", fontsize=14, transform=ax.transAxes, ha="left", va="center", bbox=bbox_style, zorder=10)
                    ax.text(y_text_pos, 0.9, "転棟期", fontsize=14, transform=ax.transAxes, ha="left", va="center", bbox=bbox_style, zorder=10)
                    
                    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
                    plt.tight_layout(pad=3.0)
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