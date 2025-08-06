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

DISEASE_OPTIONS = ["敗血症性ショック", "心原性ショック", "心臓・大血管術後", "その他（自由記載）"]
PHASE_LABELS = ["超急性期", "維持期", "回復期", "転棟期"]
PHASE_COLORS = {
    "超急性期": "#ffc0cb", # Pink
    "維持期": "#ffe4c4",   # Bisque
    "回復期": "#ffd700",   # Gold
    "転棟期": "#90ee90"    # LightGreen
}
FACTOR_SCORE_NAMES = ["循環スコア", "呼吸スコア", "意識_鎮静スコア", "腎_体液スコア", "活動_リハスコア", "栄養_消化管スコア", "感染_炎症スコア"]
ALL_COLUMN_NAMES = ["アプリ用患者ID", "日付", "時間帯", "総合スコア"] + FACTOR_SCORE_NAMES + ["イベント", "ステータス", "疾患群", "要因タグ"]

# ★★★ イベントフラグの定義 ★★★
EVENT_FLAGS = {
    # イベント名: { カテゴリ, 色, マーカー }
    "入室":      {"category": "#その他", "color": "red", "marker": "s"},
    "再手術":    {"category": "#その他", "color": "darkred", "marker": "X"},
    "転棟":      {"category": "#その他", "color": "blue", "marker": "s"},
    "抜管":      {"category": "#呼吸", "color": "green", "marker": "^"},
    "再挿管":    {"category": "#呼吸", "color": "red", "marker": "v"},
    "気管切開":  {"category": "#呼吸", "color": "blue", "marker": "v"},
    "SBT成功":   {"category": "#呼吸", "color": "lightgreen", "marker": "s"},
    "SBT失敗":   {"category": "#呼吸", "color": "darkgreen", "marker": "s"},
    "昇圧薬増量": {"category": "#循環", "color": "darkorange", "marker": "P"},
    "昇圧薬減量": {"category": "#循環", "color": "orange", "marker": "P"},
    "昇圧薬離脱": {"category": "#循環", "color": "gold", "marker": "P"},
    "補助循環開始": {"category": "#循環", "color": "deeppink", "marker": "h"},
    "補助循環weaning": {"category": "#循環", "color": "hotpink", "marker": "h"},
    "補助循環離脱": {"category": "#循環", "color": "lightpink", "marker": "h"},
    "新規不整脈": {"category": "#循環", "color": "red", "marker": "o"},
    "出血イベント": {"category": "#循環", "color": "darkred", "marker": "o"},
    "腎代替療法開始": {"category": "#腎/体液", "color": "purple", "marker": "D"},
    "腎代替療法終了": {"category": "#腎/体液", "color": "purple", "marker": "D"},
    "せん妄":      {"category": "#意識/鎮静", "color": "magenta", "marker": "*"},
    "SAT成功":      {"category": "#意識/鎮静", "color": "lightpink", "marker": "*"},
    "SAT失敗":      {"category": "#意識/鎮静", "color": "deeppink", "marker": "*"},
    "新規感染症":  {"category": "#感染/炎症", "color": "brown", "marker": "X"},
    "端坐位":    {"category": "#活動/リハ", "color": "cyan", "marker": "P"},
    "立位":    {"category": "#活動/リハ", "color": "darkcyan", "marker": "P"},
    "歩行":    {"category": "#活動/リハ", "color": "blue", "marker": "P"},
    "経管栄養開始": {"category": "#栄養/消化管", "color": "greenyellow", "marker": "+"},
    "経口摂取開始": {"category": "#栄養/消化管", "color": "lime", "marker": "+"}
}
# --- 関数 ---
def load_data(filename):
    if not os.path.exists(filename):
        return pd.DataFrame(columns=ALL_COLUMN_NAMES)
    df = pd.read_csv(filename, dtype={'イベント': str, '疾患群': str, '要因タグ': str})
    if 'スコア' in df.columns:
        df = df.rename(columns={'スコア': '総合スコア'})
    for col in ALL_COLUMN_NAMES:
        if col not in df.columns:
            df[col] = pd.NA
    return df

def calculate_derived_columns(df):
    if df.empty or '総合スコア' not in df.columns or '日付' not in df.columns:
        return df.assign(フェーズ=None, 経過日数=None)
    df_copy = df.copy()
    df_copy['日付'] = pd.to_datetime(df_copy['日付'])
    bins = [-1, 20, 60, 80, 100]
    labels = PHASE_LABELS
    df_copy['フェーズ'] = pd.cut(pd.to_numeric(df_copy['総合スコア'], errors='coerce'), bins=bins, labels=labels, right=True)
    df_copy['入室日'] = df_copy.groupby('アプリ用患者ID')['日付'].transform('min')
    df_copy['経過日数'] = (df_copy['日付'] - df_copy['入室日']).dt.days + 1
    df_copy = df_copy.drop(columns=['入室日'])
    return df_copy

# ★★★ ステップ4で変更 ★★★
def create_radar_chart(labels, current_data, previous_data=None, current_label='最新', previous_label='前回', current_color='blue', previous_color='red', current_style='-', previous_style='--'):
    """2つのデータを比較するレーダーチャートを作成する関数"""
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    # 背景にフェーズごとの色分けを追加
    ax.bar(x=0, height=20, width=2*np.pi, bottom=80, color='lightgreen', alpha=0.3, zorder=0)
    ax.bar(x=0, height=20, width=2*np.pi, bottom=60, color='yellow', alpha=0.3, zorder=0)
    ax.bar(x=0, height=40, width=2*np.pi, bottom=20, color='bisque', alpha=0.3, zorder=0)
    ax.bar(x=0, height=20, width=2*np.pi, bottom=0, color='pink', alpha=0.3, zorder=0)
    
    # 比較対象(前回)のデータをプロット
    if previous_data:
        prev_values = [previous_data.get(label, 0) for label in labels]
        prev_values = [v if pd.notna(v) else 0 for v in prev_values]
        prev_values += prev_values[:1]
        ax.plot(angles, prev_values, color=previous_color, linestyle=previous_style, linewidth=2, label=previous_label)
        ax.fill(angles, prev_values, color=previous_color, alpha=0.1)

    # 現在のデータをプロット
    curr_values = [current_data.get(label, 0) for label in labels]
    curr_values = [v if pd.notna(v) else 0 for v in curr_values]
    curr_values += curr_values[:1]
    ax.plot(angles, curr_values, color=current_color, linestyle=current_style, linewidth=2.5, label=current_label)
    ax.fill(angles, curr_values, color=current_color, alpha=0.25)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_rlim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return fig

def create_score_input(label, default_value, key_prefix):
    slider_val = st.slider(f"{label} (大まか)", 0, 100, int(default_value), step=5, key=f"{key_prefix}_slider")
    number_val = st.number_input(f"{label} (細かく)", 0, 100, slider_val, step=1, key=f"{key_prefix}_number")
    return number_val

def run_app():
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        .title-box {
            background-color: #e8f0f7; /* 薄い青色 */
            padding: 12px;
            border-radius: 10px;
            border: 2px solid #a3c1de; /* 少し濃い青色の枠線 */
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .title-box h1 {
            text-align: center;
            color: #1f497d; /* 濃い青色の文字 */
            margin: 0;
        }
        </style>
        <div class="title-box">
            <h1>軌跡シートアプリ 🏥</h1>
        </div>
    """, unsafe_allow_html=True)
    st.write("") # スペース

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    
    if not st.session_state.get('logged_in'):
        st.header("ログイン")
        # (ログイン画面は変更なし)
        facility_id_input = st.text_input("施設ID")
        password = st.text_input("パスワード", type="password")
        if st.button("ログイン"):
            # ★★★ ここからログイン処理を変更 ★★★
            try:
                # マスター管理者でのログインを試みる
                master_id_secret = st.secrets["master_credentials"]["id"]
                master_pw_secret = st.secrets["master_credentials"]["password"]
                
                # 通常施設のパスワード辞書を取得
                passwords_secret = st.secrets["passwords"]

                if facility_id_input == master_id_secret and password == master_pw_secret:
                    st.session_state.logged_in = True
                    st.session_state.facility_id = master_id_secret
                    st.rerun()
                
                # 通常ユーザーでのログインを試みる
                elif facility_id_input in passwords_secret and password == passwords_secret[facility_id_input]:
                    st.session_state.logged_in = True
                    st.session_state.facility_id = facility_id_input
                    st.rerun()
                
                else:
                    st.error("施設IDまたはパスワードが間違っています。")

            except Exception as e:
                st.error(f"認証中にエラーが発生しました。Secretsが正しく設定されているか確認してください。: {e}")
           # ★★★ ここまでログイン処理を変更 ★★★
    else:
        facility_id = st.session_state.facility_id
        patient_id_to_use = None
    
        with st.sidebar:
            st.header(f"施設ID: {facility_id}")
            
            if facility_id != st.secrets.get("master_credentials", {}).get("id", "master_admin_fallback"):
                DATA_FILE = f"patient_data_{facility_id}.csv"
                if 'df' not in st.session_state or st.session_state.get('current_facility') != facility_id:
                    st.session_state.df = load_data(DATA_FILE)
                    st.session_state.df['ステータス'] = st.session_state.df['ステータス'].fillna('在室中')
                    st.session_state.current_facility = facility_id
                
                st.subheader("患者選択")
                active_patients = sorted(st.session_state.df[st.session_state.df['ステータス'] == '在室中']['アプリ用患者ID'].unique()) if not st.session_state.df.empty else []
                selected_patient = st.selectbox("表示・記録する患者IDを選択", options=["新しい患者を登録..."] + active_patients)
                
                patient_id_to_use = st.text_input("新しいアプリ用患者IDを入力してください") if selected_patient == "新しい患者を登録..." else selected_patient

                if patient_id_to_use:
                    st.subheader("データ入力・修正")
                    st.write(f"**対象患者:** {patient_id_to_use}")
                    
                    record_date = st.date_input("日付", datetime.date.today())
                    time_of_day = st.selectbox("時間帯", options=["朝", "夕"])
                    
                    # 1. 絶対的なデフォルト値を「10」に設定
                    default_values = {name: 10 for name in FACTOR_SCORE_NAMES}
                    default_values["総合スコア"] = 10
                    default_values["イベント"] = ""
                    
                    # 患者の最新の疾患群をデフォルトとして取得
                    patient_df = st.session_state.df[st.session_state.df['アプリ用患者ID'] == patient_id_to_use]
                    if not patient_df.empty:
                        latest_disease_group = patient_df.sort_values(by="日付", ascending=False).iloc[0]['疾患群']
                        default_values["疾患群"] = latest_disease_group if pd.notna(latest_disease_group) else DISEASE_OPTIONS[0]
                    else:
                        default_values["疾患群"] = DISEASE_OPTIONS[0]

                    # 2. 修正対象のデータを検索
                    existing_data = patient_df[
                        (patient_df['日付'] == str(record_date)) & 
                        (patient_df['時間帯'] == time_of_day)
                    ]

                    if not existing_data.empty:
                        # Case 1: 選択した日時のデータが存在する場合（修正モード）
                        # そのデータをデフォルト値として読み込む
                        record = existing_data.iloc[0].to_dict()
                        for col, val in record.items():
                            if pd.notna(val) and col in default_values:
                                default_values[col] = val
                    else:
                        # Case 2: 選択した日時のデータが存在しない場合（新規入力モード）
                        # 直前の勤務帯のデータを検索してデフォルト値として読み込む
                        patient_df_copy = patient_df.copy()
                        if not patient_df_copy.empty:
                            patient_df_copy['日付'] = pd.to_datetime(patient_df_copy['日付'])
                            patient_df_copy['プロット用日時'] = patient_df_copy.apply(lambda row: row['日付'].replace(hour=8 if row['時間帯'] == '朝' else 20), axis=1)
                            
                            current_selection_dt = pd.to_datetime(str(record_date)).replace(hour=8 if time_of_day == '朝' else 20)
                            previous_records = patient_df_copy[patient_df_copy['プロット用日時'] < current_selection_dt]

                            if not previous_records.empty:
                                last_record = previous_records.sort_values(by='プロット用日時').iloc[-1].to_dict()
                                for col, val in last_record.items():
                                    if pd.notna(val) and col in default_values:
                                        if col != 'イベント': # イベント内容は引き継がない
                                            default_values[col] = val
                    
                    # 3. UIウィジェットを生成（default_valuesが使用される）
                    disease_group_index = DISEASE_OPTIONS.index(default_values["疾患群"]) if default_values["疾患群"] in DISEASE_OPTIONS else 3
                    disease_group_select = st.selectbox("疾患群を選択", options=DISEASE_OPTIONS, index=disease_group_index)
                    disease_group = st.text_input("疾患群を自由記載", value=default_values["疾患群"]) if disease_group_select == "その他（自由記載）" else disease_group_select

                    st.write("---")
                    st.write("**多職種スコア入力**")


                    # ★★★ 要望1を反映 ★★★
                    guidelines = {
                        "循環スコア": "- **0-20:** 昇圧薬(高用量) or 補助循環(ECMO/Impella)導入 or 致死的不整脈\n- **21-40:** 昇圧薬(中等量) or 補助循環化に安定\n- **41-60:** 昇圧薬(少量) or 補助循環weaning\n- **61-80:** 昇圧薬離脱 or 補助循環終了\n- **81-100:** 循環動態が安定",
                        "呼吸スコア": "- **0-20:** 高い呼吸器設定、筋弛緩使用\n- **21-40:** 自発呼吸モード、低い呼吸器設定、非挿管だが頻呼吸\n- **41-60:** SBT成功～抜管\n- **61-80:** 抜管～HFNC/NPPV離脱\n- **81-100:** 経鼻酸素～酸素なしで安定",
                        "意識_鎮静スコア": "- **0-20:** 深い鎮静(RASS-4~-5) or 意識障害\n- **21-40:** 浅い鎮静(RASS-1~-3) or せん妄\n- **41-60:** SAT成功\n- **61-80:** 会話可能 or 良好な筆談\n- **81-100:** 意識清明、良好な睡眠",
                        "腎_体液スコア": "- **0-20:** 大量輸液・輸血が必要\n- **21-40:** 大量輸液は不要だが除水はできず\n- **41-60:** バランス±0～-500mL/dayほどの緩徐なマイナスバランス\n- **61-80:** refilling、積極的な除水\n- **81-100:** 適正体重への除水達成",
                        "活動_リハスコア": "- **0-20:** 体位変換にも制限、ROM訓練のみ\n- **21-40:** ベッド上安静（ギャッジアップなど）\n- **41-60:** 端座位達成\n- **61-80:** 立位達成\n- **81-100:** 室内歩行開始",
                        "栄養_消化管スコア": "- **0-20:** 絶食、消化管トラブルあり\n- **21-40:** 経腸栄養(少量)開始\n- **41-60:** 経腸栄養を増量中\n- **61-80:** 目標カロリー達成、経口摂取開始\n- **81-100:** 経口摂取が自立",
                        "感染_炎症スコア": "- **0-20:** 敗血症性ショック\n- **21-40:** マーカー高値だがIL-6、PCT peak out\n- **41-60:** 解熱、CRPもpeak out\n- **61-80:** 抗菌薬のDe-escalation済み、CRP<10mg/dL\n- **81-100:** 抗菌薬終了、炎症反応正常化"
                    }
                    
                    factor_scores = {}
                    for score_name in FACTOR_SCORE_NAMES:
                        col1, col2 = st.columns([0.85, 0.15])
                        with col1:
                            # ★★★ 要望3を反映 ★★★
                            factor_scores[score_name] = create_score_input(score_name, default_values[score_name], score_name)
                        with col2:
                            st.popover("❓", help="スコアリングの目安").markdown(guidelines[score_name])
                     
                    st.write("---")
                    st.write("**ICU医師 最終判断**")
                    total_score = create_score_input("総合スコア", default_values["総合スコア"], "total_score")
                    
                     # ★★★ 要望3を反映 ★★★
                    st.write("---")
                    st.write("**イベント**")
                    major_event_options = [""] + list(EVENT_FLAGS.keys())
                    default_event_text = default_values["イベント"]
                    try:
                        event_index = major_event_options.index(default_event_text)
                    except ValueError:
                        event_index = 0
                    selected_event = st.selectbox("主要イベントを選択（任意）", options=major_event_options, index=event_index)
                    event_text = st.text_input("イベント（自由記述も可）", value=selected_event if selected_event and selected_event != default_event_text else default_event_text)
  
                    if st.button("記録・修正する"):
                        new_data_dict = {
                            "アプリ用患者ID": patient_id_to_use, "日付": str(record_date), "時間帯": time_of_day, 
                            "総合スコア": total_score, "イベント": event_text, "ステータス": "在室中", 
                            "疾患群": disease_group, "要因タグ": ""
                        }
                        new_data_dict.update(factor_scores)
                        
                        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_data_dict])], ignore_index=True)
                        st.session_state.df = st.session_state.df.drop_duplicates(subset=['アプリ用患者ID', '日付', '時間帯'], keep='last').sort_values(by=["アプリ用患者ID", "日付", "時間帯"])
                        st.session_state.df.to_csv(DATA_FILE, index=False)
                        st.success("データを記録しました！")
                        st.rerun()
            
            st.write("---")
            if st.button("ログアウト"):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()

        # --- メイン画面 ---
        if facility_id == st.secrets.get("master_credentials", {}).get("id", "master_admin_fallback"):
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
                display_df = calculate_derived_columns(display_df) # ★★★ この行を追加 ★★★

                if not display_df.empty:
                    st.header(f"患者: {patient_id_to_use}")
                    latest_record = display_df.sort_values(by="日付", ascending=False).iloc[0]
                    st.markdown(f"#### **疾患群:** {latest_record['疾患群']}")

                    # ★★★ 要望4を反映 ★★★
                    st.write("---")
                                   # --- 日付・時間帯選択 ---
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        available_dates = sorted(pd.to_datetime(display_df['日付']).dt.date.unique(), reverse=True)
                        selected_date = st.selectbox("日付を選択", options=available_dates, format_func=lambda d: d.strftime('%Y-%m-%d'))
                    with col2:
                        times_on_date = display_df[pd.to_datetime(display_df['日付']).dt.date == selected_date]['時間帯'].unique()
                        index_val = 1 if "夕" in times_on_date and len(times_on_date) > 1 else 0
                        selected_time = st.radio("時間帯を選択", ["朝", "夕"], horizontal=True, index=index_val)
                     # --- データ準備 ---
                    df_sorted = display_df.copy()
                    df_sorted['日付'] = pd.to_datetime(df_sorted['日付'])
                    df_sorted['プロット用日時'] = df_sorted.apply(lambda row: row['日付'].replace(hour=8 if row['時間帯'] == '朝' else 20), axis=1)
                    df_sorted = df_sorted.sort_values(by='プロット用日時').reset_index(drop=True)
                    current_index = df_sorted.index[(df_sorted['日付'].dt.date == selected_date) & (df_sorted['時間帯'] == selected_time)].tolist()
                    
                    if current_index:
                        current_idx = current_index[0]
                        current_record = df_sorted.iloc[current_idx]
                        previous_record = df_sorted.iloc[current_idx - 1] if current_idx > 0 else None
                       
                        # --- ★★★ 要望2, 3を反映 ★★★ ---
                        # スコアサマリーを先に表示
                        st.subheader("スコアサマリー")
                        cols_metric = st.columns(2)
                        with cols_metric[0]: # 前回データ
                            if previous_record is not None:
                                phase_color = PHASE_COLORS.get(previous_record['フェーズ'], '#888')
                                st.markdown(f"""
                                <div class="metric-container">
                                    <div style="font-size: 14px; color: #888;">前回 ({previous_record['日付'].strftime('%m/%d')} {previous_record['時間帯']})</div>
                                    <div style="font-size: 32px; font-weight: bold; color: #333;">{int(previous_record['総合スコア'])}</div>
                                    <div style="font-size: 18px; font-weight: bold; color: {phase_color};">{previous_record['フェーズ']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("比較対象の前回データがありません。")
                        
                        with cols_metric[1]: # 今回データ
                            phase_color = PHASE_COLORS.get(current_record['フェーズ'], '#888')
                            st.markdown(f"""
                                <div class="metric-container">
                                    <div style="font-size: 14px; color: #888;">今回 ({current_record['日付'].strftime('%m/%d')} {current_record['時間帯']})</div>
                                    <div style="font-size: 32px; font-weight: bold; color: #1f497d;">{int(current_record['総合スコア'])}</div>
                                    <div style="font-size: 18px; font-weight: bold; color: {phase_color};">{current_record['フェーズ']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        st.write("---")
                        # コンディションサマリー（レーダーチャート）
                        st.subheader("コンディションサマリー（比較）")
                        current_data = current_record[FACTOR_SCORE_NAMES].to_dict()
                        previous_data = previous_record[FACTOR_SCORE_NAMES].to_dict() if previous_record is not None else None
                        
                        if selected_time == '夕':
                            current_label, previous_label, current_color, previous_color, current_style, previous_style = "当日 夕", "当日 朝", 'red', 'blue', '-', '-'
                        else:
                            current_label, previous_label, current_color, previous_color, current_style, previous_style = "当日 朝", "前日 夕", 'blue', 'red', '-', '--'
                        
                        fig_radar = create_radar_chart(
                            labels=FACTOR_SCORE_NAMES, current_data=current_data, previous_data=previous_data,
                            current_label=current_label, previous_label=previous_label, current_color=current_color,
                            previous_color=previous_color, current_style=current_style, previous_style=previous_style
                        )
                        st.pyplot(fig_radar)
                    else:
                        st.info(f"{selected_date.strftime('%Y-%m-%d')} {selected_time} のデータはありません。")
                    
                    st.write("---")
                    st.subheader("軌跡シート")
                    df_graph = display_df.copy()
                    if not df_graph.empty:
                        df_graph['日付'] = pd.to_datetime(df_graph['日付'])
                        df_graph['プロット用日時'] = df_graph.apply(lambda row: row['日付'].replace(hour=8 if row['時間帯'] == '朝' else 20), axis=1)
                        df_graph = df_graph.sort_values(by='プロット用日時')

                        fig, ax = plt.subplots(figsize=(12, 7))            
                        ax.plot(df_graph['プロット用日時'], pd.to_numeric(df_graph['総合スコア'], errors='coerce'), marker='o', linestyle='-', markersize=8)
                        
                        events_to_plot = df_graph.dropna(subset=['イベント'])
                        for _, row in events_to_plot.iterrows():
                            event_name, plot_time, plot_score = row['イベント'], row['プロット用日時'], pd.to_numeric(row['総合スコア'], errors='coerce')
                            if pd.isna(plot_score) or not event_name: continue
                            flag = EVENT_FLAGS.get(event_name)
                            if flag:
                                ax.scatter(plot_time, plot_score, color=flag['color'], marker=flag['marker'], s=200, zorder=12)
                                ax.annotate(event_name, (plot_time, plot_score), xytext=(0, 15), textcoords='offset points', ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc=flag['color'], alpha=0.7))
                        
                        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                        fig.autofmt_xdate(rotation=30)
                        
                        ax.set_ylim(-5, 105)
                        ax.set_title("治療フェーズの軌跡", fontsize=20, pad=20)
                        ax.set_ylabel("総合スコア", fontsize=16)
                        ax.set_xlabel("日付", fontsize=16)
                        ax.tick_params(axis='both', which='major', labelsize=14)
                        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

                        
                        ax.set_ylim(-5, 105)
                        ax.set_title("治療フェーズの軌跡", fontsize=20, pad=20)
                        ax.set_ylabel("総合スコア", fontsize=16)
                        ax.set_xlabel("日付", fontsize=16)
                        ax.tick_params(axis='both', which='major', labelsize=14)
                        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
                        
                        bbox_style = dict(boxstyle='round,pad=0.4', fc='white', ec='none', alpha=0.85)
                        ax.axhspan(0, 20, color=PHASE_COLORS["超急性期"], alpha=0.3)
                        ax.axhspan(20, 60, color=PHASE_COLORS["維持期"], alpha=0.3)
                        ax.axhspan(60, 80, color=PHASE_COLORS["回復期"], alpha=0.3)
                        ax.axhspan(80, 100, color=PHASE_COLORS["転棟期"], alpha=0.3)
                        ax.text(0.02, 0.1, "超急性期", fontsize=18, transform=ax.transAxes, bbox=bbox_style)
                        ax.text(0.02, 0.4, "維持期", fontsize=18, transform=ax.transAxes, bbox=bbox_style)
                        ax.text(0.02, 0.7, "回復期", fontsize=18, transform=ax.transAxes, bbox=bbox_style)
                        ax.text(0.02, 0.9, "転棟期", fontsize=18, transform=ax.transAxes, bbox=bbox_style)

                        plt.tight_layout(pad=2.0)
                        st.pyplot(fig)

                else: st.info(f"「{patient_id_to_use}」さんのデータはまだありません。")
            else: st.info("サイドバーで患者を選択または新規登録してください。")

            st.write("---")
            st.header("管理")
            if patient_id_to_use and not display_df.empty:
                if st.button(f"{patient_id_to_use} を退室済（アーカイブ）にする"):
                    st.session_state.df.loc[st.session_state.df['アプリ用患者ID'] == patient_id_to_use, 'ステータス'] = '退室済'
                    st.session_state.df.to_csv(DATA_FILE, index=False)
                    st.success(f"{patient_id_to_use} さんをアーカイブしました。")
                    st.rerun()

            show_archive = st.checkbox("アーカイブされた患者を表示")
            if show_archive:
                archived_df = st.session_state.df[st.session_state.df['ステータス'] == '退室済']
                st.write("#### 退室済（アーカイブ）患者一覧")
                st.dataframe(archived_df)
                for patient_id in sorted(archived_df['アプリ用患者ID'].unique()):
                    if st.button("在室中に戻す", key=f"reactivate_{patient_id}"):
                        st.session_state.df.loc[st.session_state.df['アプリ用患者ID'] == patient_id, 'ステータス'] = '在室中'
                        st.session_state.df.to_csv(DATA_FILE, index=False)
                        st.success(f"{patient_id}さんを在室中に戻しました。")
                        st.rerun()
# --- プログラムの実行開始点 ---
if __name__ == "__main__":
    run_app()