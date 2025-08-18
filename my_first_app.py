import streamlit as st
import pandas as pd
import datetime
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm 

# ★★★ フォント設定 ★★★
# アプリと同じディレクトリにあるフォントファイルへのパス
font_path = 'ipaexg.ttf'
# フォントプロパティをグローバル変数として定義
prop = fm.FontProperties(fname=font_path) if os.path.exists(font_path) else None

# --- 定数と設定 ---
DATA_FILE_PREFIX = "patient_data_"
LOG_FILE_PREFIX = "log_data_"
DISEASE_OPTIONS = ["敗血症性ショック", "心原性ショック", "心臓・大血管術後", "その他（自由記載）"]
PHASE_LABELS = ["超急性期", "維持期", "回復期", "転棟期"]
PHASE_COLORS = {
    "超急性期": "#ffc0cb", "維持期": "#ffe4c4",
    "回復期": "#ffd700", "転棟期": "#90ee90"
}
FACTOR_SCORE_NAMES = ["循環スコア", "呼吸スコア", "意識_鎮静スコア", "腎_体液スコア", "活動_リハスコア", "栄養_消化管スコア", "感染_炎症スコア"]
ALL_COLUMN_NAMES = ["アプリ用患者ID", "日付", "時間帯", "総合スコア"] + FACTOR_SCORE_NAMES + ["イベント", "ステータス", "疾患群", "要因タグ", "退室時転帰"]
EVENT_FLAGS = {
    "入室": {"category": "#その他", "color": "red", "marker": "s"},"挿管": {"category": "#呼吸", "color": "darkred", "marker": "v"},"再手術": {"category": "#その他", "color": "darkred", "marker": "X"},
    "転棟": {"category": "#その他", "color": "blue", "marker": "s"},"抜管": {"category": "#呼吸", "color": "green", "marker": "^"},"再挿管": {"category": "#呼吸", "color": "red", "marker": "v"},
    "気管切開": {"category": "#呼吸", "color": "blue", "marker": "v"},"SBT成功": {"category": "#呼吸", "color": "lightgreen", "marker": "s"},"SBT失敗": {"category": "#呼吸", "color": "darkgreen", "marker": "s"},
    "昇圧薬開始": {"category": "#循環", "color": "darkorange", "marker": "P"},"昇圧薬増量": {"category": "#循環", "color": "darkorange", "marker": "P"},"昇圧薬減量": {"category": "#循環", "color": "orange", "marker": "P"},
    "昇圧薬離脱": {"category": "#循環", "color": "gold", "marker": "P"},"補助循環開始": {"category": "#循環", "color": "deeppink", "marker": "h"},"補助循環weaning": {"category": "#循環", "color": "hotpink", "marker": "h"},
    "補助循環離脱": {"category": "#循環", "color": "lightpink", "marker": "h"},"新規不整脈": {"category": "#循環", "color": "red", "marker": "o"},"出血イベント": {"category": "#循環", "color": "darkred", "marker": "o"},"AKI": {"category": "#腎/体液", "color": "mediumpurple", "marker": "D"},
    "腎代替療法開始": {"category": "#腎/体液", "color": "purple", "marker": "D"},"腎代替療法終了": {"category": "#腎/体液", "color": "purple", "marker": "D"},"せん妄": {"category": "#意識/鎮静", "color": "magenta", "marker": "*"},
    "SAT成功": {"category": "#意識/鎮静", "color": "lightpink", "marker": "*"},"SAT失敗": {"category": "#意識/鎮静", "color": "deeppink", "marker": "*"},"新規感染症": {"category": "#感染/炎症", "color": "brown", "marker": "X"},
    "抗生剤de-escalation": {"category": "#感染/炎症", "color": "sandybrown", "marker": "X"},"ソースコントロール": {"category": "#感染/炎症", "color": "sienna", "marker": "X"}, 
    "端坐位": {"category": "#活動/リハ", "color": "cyan", "marker": "P"},"立位": {"category": "#活動/リハ", "color": "darkcyan", "marker": "P"},"歩行": {"category": "#活動/リハ", "color": "blue", "marker": "P"},
    "経管栄養開始": {"category": "#栄養/消化管", "color": "greenyellow", "marker": "+"},"経口摂取開始": {"category": "#栄養/消化管", "color": "lime", "marker": "+"}
}

def load_data(filename):
    if not os.path.exists(filename): return pd.DataFrame(columns=ALL_COLUMN_NAMES)
    try:
        df = pd.read_csv(filename, dtype={'イベント': str, '疾患群': str, '要因タグ': str})
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}"); return pd.DataFrame(columns=ALL_COLUMN_NAMES)
    if 'スコア' in df.columns: df = df.rename(columns={'スコア': '総合スコア'})
    for col in ALL_COLUMN_NAMES:
        if col not in df.columns: df[col] = pd.NA
    return df

def calculate_derived_columns(df):
    if df.empty or '総合スコア' not in df.columns or '日付' not in df.columns: return df.assign(フェーズ=None, 経過日数=None, プロット用日時=None)
    df_copy = df.copy()
    df_copy['日付'] = pd.to_datetime(df_copy['日付'])
    bins = [-1, 20, 60, 80, 100]; labels = PHASE_LABELS
    scores = pd.to_numeric(df_copy['総合スコア'], errors='coerce').fillna(-1)
    df_copy['フェーズ'] = pd.cut(scores, bins=bins, labels=labels, right=True)
    try:
        df_copy['入室日'] = df_copy.groupby('アプリ用患者ID')['日付'].transform('min')
        df_copy['経過日数'] = (df_copy['日付'] - df_copy['入室日']).dt.days + 1
        df_copy = df_copy.drop(columns=['入室日'])
    except Exception: df_copy['経過日数'] = None
    df_copy['プロット用日時'] = df_copy.apply(lambda row: row['日付'].replace(hour=8 if row['時間帯'] == '朝' else 20), axis=1)
    return df_copy
    
def create_radar_chart(labels, current_data, previous_data=None, current_label='最新', previous_label='前回', current_color='blue', previous_color='red', current_style='-', previous_style='--'):
    num_vars = len(labels); angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.bar(x=0, height=20, width=2*np.pi, bottom=80, color=PHASE_COLORS["転棟期"], alpha=0.3, zorder=0)
    ax.bar(x=0, height=20, width=2*np.pi, bottom=60, color=PHASE_COLORS["回復期"], alpha=0.3, zorder=0)
    ax.bar(x=0, height=40, width=2*np.pi, bottom=20, color=PHASE_COLORS["維持期"], alpha=0.3, zorder=0)
    ax.bar(x=0, height=20, width=2*np.pi, bottom=0, color=PHASE_COLORS["超急性期"], alpha=0.3, zorder=0)
    if previous_data:
        prev_values = [previous_data.get(label, 0) for label in labels]; prev_values = [v if pd.notna(v) else 0 for v in prev_values]; prev_values += prev_values[:1]
        ax.plot(angles, prev_values, color=previous_color, linestyle=previous_style, linewidth=2, label=previous_label, zorder=5)
        ax.fill(angles, prev_values, color=previous_color, alpha=0.1, zorder=4)
    curr_values = [current_data.get(label, 0) for label in labels]; curr_values = [v if pd.notna(v) else 0 for v in curr_values]; curr_values += curr_values[:1]
    ax.plot(angles, curr_values, color=current_color, linestyle=current_style, linewidth=2.5, label=current_label, zorder=10)
    ax.fill(angles, curr_values, color=current_color, alpha=0.25, zorder=9)
    ax.set_yticklabels([]); ax.set_xticks(angles[:-1])
    if prop:
        ax.set_xticklabels(labels, fontsize=16, fontproperties=prop)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), prop=prop)
    else:
        ax.set_xticklabels(labels, fontsize=16)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_rlim(0, 100); return fig

def create_score_input(label, default_value, key_prefix):
    slider_val = st.slider(f"{label} (大まか)", 0, 100, int(default_value), step=5, key=f"{key_prefix}_slider")
    number_val = st.number_input(f"{label} (細かく)", 0, 100, slider_val, step=1, key=f"{key_prefix}_number")
    return number_val

def write_log(log_file, facility_id, patient_id, action):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {"timestamp": [now], "facility_id": [facility_id], "patient_id": [patient_id], "action": [action]}
    log_df = pd.DataFrame(log_entry)
    if not os.path.exists(log_file): log_df.to_csv(log_file, index=False, encoding='utf-8-sig')
    else: log_df.to_csv(log_file, mode='a', header=False, index=False, encoding='utf-8-sig')

def run_app():
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        .title-box { background-color: #e8f0f7; padding: 12px; border-radius: 10px; border: 2px solid #a3c1de; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .title-box h1 { text-align: center; color: #1f497d; margin: 0; }
        .metric-container { border: 1px solid #ddd; border-radius: 10px; padding: 15px; text-align: center; background-color: #f9f9f9; }
        </style>
        <div class="title-box"><h1>軌跡シートアプリ 🏥</h1></div>
    """, unsafe_allow_html=True)
    st.write("")

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
     # ★★★ 機能A：画面の状態管理を追加 ★★★
    if 'view_mode' not in st.session_state: st.session_state.view_mode = 'main'
    
    if not st.session_state.get('logged_in'):
        st.header("ログイン")
        facility_id_input = st.text_input("施設ID"); password = st.text_input("パスワード", type="password")
        if st.button("ログイン"):
            try:
                master_id_secret = st.secrets.get("master_credentials", {}).get("id"); master_pw_secret = st.secrets.get("master_credentials", {}).get("password")
                passwords_secret = st.secrets.get("passwords", {})
                if facility_id_input == master_id_secret and password == master_pw_secret:
                    st.session_state.logged_in = True; st.session_state.facility_id = master_id_secret; st.rerun()
                elif facility_id_input in passwords_secret and password == passwords_secret[facility_id_input]:
                    st.session_state.logged_in = True; st.session_state.facility_id = facility_id_input; st.rerun()
                else: st.error("施設IDまたはパスワードが間違っています。")
            except Exception as e: st.error(f"認証中にエラーが発生しました。Secretsが正しく設定されているか確認してください。: {e}")
    else:
        facility_id = st.session_state.facility_id; patient_id_to_use = None
        # サイドバー
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
                    st.subheader("データ入力・修正"); st.write(f"**対象患者:** {patient_id_to_use}")
                    record_date = st.date_input("日付", datetime.date.today())
                    if record_date > datetime.date.today(): st.error("未来の日付は入力できません。"); st.stop()
                    time_of_day = st.selectbox("時間帯", options=["朝", "夕"])
                    default_values = {name: 10 for name in FACTOR_SCORE_NAMES}; default_values["総合スコア"] = 10; default_values["イベント"] = ""
                    patient_df = st.session_state.df[st.session_state.df['アプリ用患者ID'] == patient_id_to_use]
                    if not patient_df.empty:
                        latest_disease_group = patient_df.sort_values(by="日付", ascending=False).iloc[0]['疾患群']
                        default_values["疾患群"] = latest_disease_group if pd.notna(latest_disease_group) else DISEASE_OPTIONS[0]
                    else: default_values["疾患群"] = DISEASE_OPTIONS[0]
                    existing_data = patient_df[(patient_df['日付'] == str(record_date)) & (patient_df['時間帯'] == time_of_day)]
                    if not existing_data.empty:
                        record = existing_data.iloc[0].to_dict()
                        for col, val in record.items():
                            if pd.notna(val) and col in default_values: default_values[col] = val
                    else:
                        patient_df_copy = patient_df.copy()
                        if not patient_df_copy.empty:
                            patient_df_copy['日付'] = pd.to_datetime(patient_df_copy['日付'])
                            patient_df_copy['プロット用日時'] = patient_df_copy.apply(lambda row: row['日付'].replace(hour=8 if row['時間帯'] == '朝' else 20), axis=1)
                            current_selection_dt = pd.to_datetime(str(record_date)).replace(hour=8 if time_of_day == '朝' else 20)
                            previous_records = patient_df_copy[patient_df_copy['プロット用日時'] < current_selection_dt]
                            if not previous_records.empty:
                                last_record = previous_records.sort_values(by='プロット用日時').iloc[-1].to_dict()
                                for col, val in last_record.items():
                                    if pd.notna(val) and col in default_values and col != 'イベント': default_values[col] = val
                    disease_group_index = DISEASE_OPTIONS.index(default_values["疾患群"]) if default_values["疾患群"] in DISEASE_OPTIONS else 3
                    disease_group_select = st.selectbox("疾患群を選択", options=DISEASE_OPTIONS, index=disease_group_index)
                    disease_group = st.text_input("疾患群を自由記載", value=default_values["疾患群"]) if disease_group_select == "その他（自由記載）" else disease_group_select
                    st.write("---"); st.write("**多職種スコア入力**")
                # ★★★ ここにguidelines辞書の定義をまるごと挿入します ★★★
                    guidelines = {
                        "循環スコア": "- **0-19:** 昇圧薬(高用量) or 補助循環(ECMO/Impella)導入 or 致死的不整脈\n- **20-39:** 昇圧薬(中等量) or 補助循環化に安定\n- **40-59:** 昇圧薬(少量) or 補助循環weaning\n- **60-89:** 昇圧薬離脱 or 補助循環終了\n- **90-100:** 循環動態が安定",
                        "呼吸スコア": "- **0-19:** 高い呼吸器設定、筋弛緩使用\n- **20-39:** 自発呼吸モード、低い呼吸器設定、非挿管だが頻呼吸\n- **40-59:** SBT成功～抜管\n- **60-89:** 抜管～HFNC/NPPV離脱\n- **90-100:** 経鼻酸素～酸素なしで安定",
                        "意識_鎮静スコア": "- **0-19:** 深い鎮静(RASS-4~-5) or 意識障害\n- **20-39:** 浅い鎮静(RASS-1~-3) or せん妄\n- **40-59:** SAT成功\n- **60-89:** 会話可能 or 良好な筆談\n- **90-100:** 意識清明、良好な睡眠",
                        "腎_体液スコア": "- **0-19:** 大量輸液・輸血が必要\n- **20-39:** 大量輸液は不要だが除水はできず\n- **40-59:** バランス±0～-500mL/dayほどの緩徐なマイナスバランス\n- **60-89:** refilling、積極的な除水\n- **90-100:** 適正体重への除水達成",
                        "活動_リハスコア": "- **0-19:** 体位変換にも制限、ROM訓練のみ\n- **20-39:** ベッド上安静（ギャッジアップなど）\n- **40-59:** 端座位達成\n- **60-89:** 立位達成\n- **90-100:** 室内歩行開始",
                        "栄養_消化管スコア": "- **0-19:** 絶食、消化管トラブルあり\n- **20-39:** 経腸栄養(少量)開始\n- **40-59:** 経腸栄養を増量中\n- **60-89:** 目標カロリー達成、経口摂取開始\n- **90-100:** 経口摂取が自立",
                        "感染_炎症スコア": "- **0-19:** 敗血症性ショック\n- **20-39:** マーカー高値だがIL-6、PCT peak out\n- **40-59:** 解熱、CRPもpeak out\n- **60-89:** 抗菌薬のDe-escalation済み、CRP<10mg/dL\n- **90-100:** 抗菌薬終了、炎症反応正常化"
                    }
                    factor_scores = {}
                    all_selected_events = [] # 全てのイベントを一時的に保存するリスト
                    # 各スコアとイベント入力欄をペアで表示
                    score_event_map = {
                        "循環スコア": "#循環", "呼吸スコア": "#呼吸", "意識_鎮静スコア": "#意識/鎮静",
                        "腎_体液スコア": "#腎/体液", "活動_リハスコア": "#活動/リハ", "栄養_消化管スコア": "#栄養/消化管",
                        "感染_炎症スコア": "#感染/炎症"
                    }
                    default_event_list = [e.strip() for e in default_values.get("イベント", "").split(',')] if default_values.get("イベント", "") else []
                    for score_name, category in score_event_map.items():
                    # ★★★ ここから修正 ★★★
                        col1, col2 = st.columns([0.85, 0.15])
                        with col1:
                        # スコア入力欄の作成をここで行う
                            factor_scores[score_name] = create_score_input(score_name, default_values.get(score_name, 10), score_name)
                        with col2:
                        # 「？」アイコンを隣に配置
                            st.popover("❓", help="スコアリングの目安").markdown(guidelines[score_name])

                    # カテゴリに一致するイベントリストを作成
                        category_events = [event for event, props in EVENT_FLAGS.items() if props.get("category") == category]
                        # 既存データから、このカテゴリのイベントのみを抽出してデフォルト値とする
                        default_category_events = [e for e in default_event_list if e in category_events]

                        selected = st.multiselect(f"{score_name} 関連イベント", options=category_events, default=default_category_events, key=f"{score_name}_events")
                        all_selected_events.extend(selected)
                        st.write("---")

                    st.write("**ICU医師 最終判断**")
                    total_score = create_score_input("総合スコア", default_values.get("総合スコア", 10), "total_score")
                    # カテゴリに属さない一般イベントの入力
                    general_events_options = [event for event, props in EVENT_FLAGS.items() if props.get("category") == "#その他"]
                    default_general_events = [e for e in default_event_list if e in general_events_options]
                    selected_general = st.multiselect("その他イベント", options=general_events_options, default=default_general_events, key="general_events")
                    all_selected_events.extend(selected_general)

                    # 全てのイベントをカンマ区切り文字列に結合（データ形式を維持）
                    event_text = ", ".join(all_selected_events)
                    # ★★★ ここまで全面的に修正 ★★★
                    previous_total_score = None
                    if not existing_data.empty:
                        patient_df_copy = patient_df.copy(); patient_df_copy['日付'] = pd.to_datetime(patient_df_copy['日付'])
                        patient_df_copy['プロット用日時'] = patient_df_copy.apply(lambda row: row['日付'].replace(hour=8 if row['時間帯'] == '朝' else 20), axis=1)
                        current_selection_dt = pd.to_datetime(str(record_date)).replace(hour=8 if time_of_day == '朝' else 20)
                        previous_records = patient_df_copy[patient_df_copy['プロット用日時'] < current_selection_dt]
                        if not previous_records.empty: previous_total_score = previous_records.sort_values(by='プロット用日時').iloc[-1]['総合スコア']
                    else: previous_total_score = default_values.get("総合スコア")
                    if previous_total_score is not None and pd.notna(previous_total_score):
                        if abs(total_score - previous_total_score) >= 41: st.warning(f"注意：スコアが前回({int(previous_total_score)}点)から41点以上変動しています。内容を確認してください。")
                    if st.button("記録・修正する"):
                        new_data_dict = {"アプリ用患者ID": patient_id_to_use, "日付": str(record_date), "時間帯": time_of_day, "総合スコア": total_score, "イベント": event_text, "ステータス": "在室中", "疾患群": disease_group, "要因タグ": ""}
                        new_data_dict.update(factor_scores); st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame([new_data_dict])], ignore_index=True)
                        st.session_state.df = st.session_state.df.drop_duplicates(subset=['アプリ用患者ID', '日付', '時間帯'], keep='last').sort_values(by=["アプリ用患者ID", "日付", "時間帯"])
                        st.session_state.df.to_csv(DATA_FILE, index=False); LOG_FILE = f"{LOG_FILE_PREFIX}{facility_id}.csv"; write_log(LOG_FILE, facility_id, patient_id_to_use, "データ記録/修正")
                        st.success("データを記録しました！"); st.rerun()
            st.write("---")
            if st.button("ログアウト"):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()
        # メイン画面
        if facility_id == st.secrets.get("master_credentials", {}).get("id", "master_admin_fallback"):
            st.header("マスター管理者モード"); st.write("全施設のアーカイブデータを表示・管理します。")
            all_files = glob.glob(f"{DATA_FILE_PREFIX}*.csv")
            if not all_files: st.info("データファイルが見つかりません。")
            else:
                all_archived_dfs = []
                for f in all_files:
                    df_temp = load_data(f); archived = df_temp[df_temp['ステータス'] == '退室済']
                    if not archived.empty:
                        facility_name = os.path.basename(f).replace(DATA_FILE_PREFIX, '').replace('.csv', ''); archived.insert(0, '施設ID', facility_name); all_archived_dfs.append(archived)
                if all_archived_dfs:
                    master_df = pd.concat(all_archived_dfs, ignore_index=True); st.dataframe(master_df)
                    csv_master = master_df.to_csv(index=False).encode('utf-8-sig'); st.download_button("全アーカイブデータをCSVでダウンロード", csv_master, 'master_archived_data.csv', 'text/csv')
                else: st.info("アーカイブされたデータを持つ施設はありません。")
        else:
            if patient_id_to_use:
                display_df = st.session_state.df[st.session_state.df['アプリ用患者ID'] == patient_id_to_use].copy()
                display_df = calculate_derived_columns(display_df)
                if not display_df.empty:
                    st.header(f"患者: {patient_id_to_use}")
                # ★★★ 機能B：ここから画面分岐ロジックを追加 ★★★
                # 現在の画面モードに応じて表示を切り替え
                    if st.session_state.view_mode == 'main':
                    # --- メイン表示モード ---
                        if st.button("🖨️ 印刷用サマリーレポートを作成"):
                            st.session_state.view_mode = 'report'
                            st.rerun()
                        latest_record_main = display_df.sort_values(by="日付", ascending=False).iloc[0];
                        st.markdown(f"#### **疾患群:** {latest_record_main['疾患群']}");
                        st.write("---")

                        col1, col2 = st.columns([1, 2])
                        with col1:
                            available_dates = sorted(pd.to_datetime(display_df['日付']).dt.date.unique(), reverse=True)
                            selected_date = st.selectbox("日付を選択", options=available_dates, format_func=lambda d: d.strftime('%Y-%m-%d'))
                        with col2:
                            times_on_date = display_df[pd.to_datetime(display_df['日付']).dt.date == selected_date]['時間帯'].unique()
                            index_val = 1 if "夕" in times_on_date and len(times_on_date) > 1 else 0
                            selected_time = st.radio("時間帯を選択", ["朝", "夕"], horizontal=True, index=index_val)
                        df_sorted = display_df.copy(); df_sorted['日付'] = pd.to_datetime(df_sorted['日付'])
                        df_sorted['プロット用日時'] = df_sorted.apply(lambda row: row['日付'].replace(hour=8 if row['時間帯'] == '朝' else 20), axis=1)
                        df_sorted = df_sorted.sort_values(by='プロット用日時').reset_index(drop=True)
                        current_index = df_sorted.index[(df_sorted['日付'].dt.date == selected_date) & (df_sorted['時間帯'] == selected_time)].tolist()
                        if current_index:
                            current_idx = current_index[0]; current_record = df_sorted.iloc[current_idx]
                            previous_record = df_sorted.iloc[current_idx - 1] if current_idx > 0 else None
                            st.subheader("スコアサマリー"); cols_metric = st.columns(2)
                            with cols_metric[0]:
                                if previous_record is not None:
                                    phase_color = PHASE_COLORS.get(previous_record['フェーズ'], '#888')
                                    st.markdown(f'<div class="metric-container"> <div style="font-size: 14px; color: #888;">前回 ({previous_record["日付"].strftime("%m/%d")} {previous_record["時間帯"]})</div> <div style="font-size: 32px; font-weight: bold; color: #333;">{int(previous_record["総合スコア"])}</div> <div style="font-size: 18px; font-weight: bold; color: {phase_color};">{previous_record["フェーズ"]}</div> </div>', unsafe_allow_html=True)
                                else: st.info("比較対象の前回データがありません。")
                            with cols_metric[1]:
                                phase_color = PHASE_COLORS.get(current_record['フェーズ'], '#888')
                                st.markdown(f'<div class="metric-container"> <div style="font-size: 14px; color: #888;">今回 ({current_record["日付"].strftime("%m/%d")} {current_record["時間帯"]})</div> <div style="font-size: 32px; font-weight: bold; color: #1f497d;">{int(current_record["総合スコア"])}</div> <div style="font-size: 18px; font-weight: bold; color: {phase_color};">{current_record["フェーズ"]}</div> </div>', unsafe_allow_html=True)
                            st.write("---"); st.subheader("コンディションサマリー（比較）")
                            current_data = current_record[FACTOR_SCORE_NAMES].to_dict(); previous_data = previous_record[FACTOR_SCORE_NAMES].to_dict() if previous_record is not None else None
                            current_label, previous_label, current_color, previous_color, current_style, previous_style = ("当日 夕", "当日 朝", 'red', 'blue', '-', '-') if selected_time == '夕' else ("当日 朝", "前日 夕", 'blue', 'red', '-', '--')
                            fig_radar = create_radar_chart(labels=FACTOR_SCORE_NAMES, current_data=current_data, previous_data=previous_data, current_label=current_label, previous_label=previous_label, current_color=current_color, previous_color=previous_color, current_style=current_style, previous_style=previous_style)
                            st.pyplot(fig_radar)
                        else: st.info(f"{selected_date.strftime('%Y-%m-%d')} {selected_time} のデータはありません。")
                        st.write("---"); st.subheader("軌跡シート")
                        df_graph = display_df.copy()
                        if not df_graph.empty:
                            df_graph['日付'] = pd.to_datetime(df_graph['日付']); df_graph['プロット用日時'] = df_graph.apply(lambda row: row['日付'].replace(hour=8 if row['時間帯'] == '朝' else 20), axis=1)
                            df_graph = df_graph.sort_values(by='プロット用日時'); fig, ax = plt.subplots(figsize=(12, 7))
                            ax.plot(df_graph['プロット用日時'], pd.to_numeric(df_graph['総合スコア'], errors='coerce'), marker='o', linestyle='-', markersize=8)
                            events_to_plot = df_graph.dropna(subset=['イベント'])
                            for _, row in events_to_plot.iterrows():
                                event_string, plot_time, plot_score = row['イベント'], row['プロット用日時'], pd.to_numeric(row['総合スコア'], errors='coerce')
                                if pd.isna(plot_score) or not event_string: continue
                        # ★★★ ここからイベント描画ロジックを修正 ★★★
                                events = [e.strip() for e in event_string.split(',')]

                        # 最初のイベントでマーカーをプロット
                                first_event_flag = EVENT_FLAGS.get(events[0])
                                if first_event_flag:
                                    ax.scatter(plot_time, plot_score, color=first_event_flag['color'], marker=first_event_flag['marker'], s=200, zorder=12)

                        # 各イベントを縦に並べて、色分けして表示
                                vertical_offset = 10 # テキストの縦方向の初期オフセット
                                for event in events:
                                    flag = EVENT_FLAGS.get(event)
                                    if flag and prop:
                                        ax.text(plot_time, plot_score + vertical_offset, f" {event} ", # y座標をここで計算
                                        ha='center', va='bottom',
                                        bbox=dict(boxstyle='round,pad=0.2', fc=flag['color'], alpha=0.7),
                                        fontproperties=prop)
                                        vertical_offset += 10 # 次のテキストのためにオフセットを増やす
                                # ★★★ ここまでイベント描画ロジックを修正 ★★★
                            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d')); fig.autofmt_xdate(rotation=30)
                            ax.set_ylim(-5, 105); ax.grid(True, axis='y', linestyle='--', alpha=0.6)
                            if prop:
                                ax.set_title("治療フェーズの軌跡", fontsize=20, pad=20, fontproperties=prop); ax.set_ylabel("総合スコア", fontsize=16, fontproperties=prop); ax.set_xlabel("日付", fontsize=16, fontproperties=prop)
                                for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontproperties(prop); label.set_fontsize(16)
                                bbox_style = dict(boxstyle='round,pad=0.4', fc='white', ec='none', alpha=0.85)
                                ax.axhspan(0, 19, color=PHASE_COLORS["超急性期"], alpha=0.3); ax.axhspan(20, 59, color=PHASE_COLORS["維持期"], alpha=0.3)
                                ax.axhspan(60, 89, color=PHASE_COLORS["回復期"], alpha=0.3); ax.axhspan(90, 100, color=PHASE_COLORS["転棟期"], alpha=0.3)
                                ax.text(0.02, 0.1, "超急性期", fontsize=18, transform=ax.transAxes, bbox=bbox_style, fontproperties=prop)
                                ax.text(0.02, 0.4, "維持期", fontsize=18, transform=ax.transAxes, bbox=bbox_style, fontproperties=prop)
                                ax.text(0.02, 0.7, "回復期", fontsize=18, transform=ax.transAxes, bbox=bbox_style, fontproperties=prop)
                                ax.text(0.02, 0.9, "転棟期", fontsize=18, transform=ax.transAxes, bbox=bbox_style, fontproperties=prop)
                            else:
                                ax.set_title("Trajectory Sheet", fontsize=20, pad=20); ax.set_ylabel("Score", fontsize=16); ax.set_xlabel("Date", fontsize=16); ax.tick_params(axis='both', which='major', labelsize=16)
                            plt.tight_layout(pad=2.0); st.pyplot(fig)

                    elif st.session_state.view_mode == 'report':
                        # --- レポート表示モード ---
                        if st.button("🔙 メイン画面に戻る"):
                            st.session_state.view_mode = 'main'
                            st.rerun()

                        st.write("---")
                        st.subheader("患者サマリーレポート")
                        st.write("ここに、印刷に適したレイアウトで軌跡シートやサマリーが表示されます。")
                else: st.info(f"「{patient_id_to_use}」さんのデータはまだありません。")
            else: st.info("サイドバーで患者を選択または新規登録してください。")
            st.write("---"); st.header("管理")
            if patient_id_to_use and not display_df.empty:
                st.write(f"**{patient_id_to_use} の管理**")
                outcome_options = ["", "軽快", "転棟", "死亡", "その他"]; selected_outcome = st.selectbox("退室時転帰を選択してください:", options=outcome_options)
                if st.button(f"{patient_id_to_use} を退室済（アーカイブ）にする"):
                    if selected_outcome:
                        patient_indices = st.session_state.df[st.session_state.df['アプリ用患者ID'] == patient_id_to_use].index
                        if not patient_indices.empty:
                            last_index = patient_indices[-1]; st.session_state.df.loc[last_index, '退室時転帰'] = selected_outcome
                        st.session_state.df.loc[st.session_state.df['アプリ用患者ID'] == patient_id_to_use, 'ステータス'] = '退室済'
                        st.session_state.df.to_csv(DATA_FILE, index=False); st.success(f"{patient_id_to_use} さんを「{selected_outcome}」としてアーカイブしました。"); st.rerun()
                    else: st.warning("退室時転帰を選択してください。")
            show_archive = st.checkbox("アーカイブされた患者を表示")
            if show_archive:
                archived_df = st.session_state.df[st.session_state.df['ステータス'] == '退室済']; st.write("#### 退室済（アーカイブ）患者一覧"); st.dataframe(archived_df); st.write("---")
                for patient_id in sorted(archived_df['アプリ用患者ID'].unique()):
                    col1, col2 = st.columns([4, 1])
                    with col1: st.write(f"**患者ID:** {patient_id}")
                    with col2:
                        if st.button("在室中に戻す", key=f"reactivate_{patient_id}", use_container_width=True):
                            st.session_state.df.loc[st.session_state.df['アプリ用患者ID'] == patient_id, 'ステータス'] = '在室中'
                            st.session_state.df.to_csv(DATA_FILE, index=False); st.success(f"{patient_id}さんを在室中に戻しました。"); st.rerun()
            st.write("---"); st.subheader("データのエクスポート")
            csv_patient_data = st.session_state.df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="患者データをCSVでダウンロード", data=csv_patient_data, file_name=f"patient_data_{facility_id}_{datetime.date.today()}.csv", mime='text/csv')
            LOG_FILE = f"{LOG_FILE_PREFIX}{facility_id}.csv"
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, "rb") as file: st.download_button(label="操作ログをCSVでダウンロード", data=file, file_name=f"log_data_{facility_id}_{datetime.date.today()}.csv", mime='text/csv')
            st.write("---"); st.header("統計ダッシュボード")
            archived_df_dashboard = st.session_state.df[st.session_state.df['ステータス'] == '退室済'].copy()
            archived_df_dashboard = calculate_derived_columns(archived_df_dashboard)
            archived_df_dashboard['プロット用経過日数'] = archived_df_dashboard.apply(lambda row: row['経過日数'] + 0.5 if row['時間帯'] == '夕' else row['経過日数'], axis=1)
            if archived_df_dashboard.empty: st.info("分析対象となる、アーカイブされた患者データがまだありません。")
            else:
                with st.expander("ダッシュボードを表示する", expanded=True):
                    tab1, tab2 = st.tabs(["軌跡の比較", "数値サマリー"])
                    with tab1:
                        st.subheader("治療軌跡の重ね合わせプロット")
                        st.info("このグラフは、選択された疾患群の全患者の回復曲線（半透明の線）と、その平均軌跡（赤線）を示しています。これにより、その疾患の典型的な回復パターンと、個々の患者のばらつきを視覚的に把握できます。")
                        disease_groups = archived_df_dashboard['疾患群'].dropna().unique()
                        if len(disease_groups) > 0:
                            selected_disease_group = st.selectbox("分析したい疾患群を選択してください", options=disease_groups)
                            if selected_disease_group:
                                active_df = st.session_state.df[st.session_state.df['ステータス'] == '在室中'].copy(); active_df = calculate_derived_columns(active_df)
                                active_df['プロット用経過日数'] = active_df.apply(lambda row: row['経過日数'] + 0.5 if row['時間帯'] == '夕' else row['経過日数'], axis=1)
                                active_patients_in_group = active_df[active_df['疾患群'] == selected_disease_group]['アプリ用患者ID'].unique()
                                selected_active_patient = st.selectbox("比較したい治療中の患者を選択（任意）", options=["比較しない"] + list(active_patients_in_group))
                                group_df = archived_df_dashboard[archived_df_dashboard['疾患群'] == selected_disease_group]; patient_ids = group_df['アプリ用患者ID'].unique()
                                fig, ax = plt.subplots(figsize=(10, 6))
                                for patient_id in patient_ids:
                                    patient_df = group_df[group_df['アプリ用患者ID'] == patient_id]; patient_df = patient_df.sort_values(by='プロット用日時')
                                    ax.plot(patient_df['プロット用経過日数'], pd.to_numeric(patient_df['総合スコア'], errors='coerce'), marker='o', linestyle='-', alpha=0.3, label='_nolegend_')
                                if not group_df.empty:
                                    mean_trajectory = group_df.groupby('プロット用経過日数')['総合スコア'].mean().reset_index()
                                    ax.plot(mean_trajectory['プロット用経過日数'], mean_trajectory['総合スコア'], marker='o', linestyle='-', linewidth=3, color='red', label=f'{selected_disease_group} 平均')
                                if selected_active_patient != "比較しない":
                                    current_patient_df = active_df[active_df['アプリ用患者ID'] == selected_active_patient]; current_patient_df = current_patient_df.sort_values(by='プロット用日時')
                                    ax.plot(current_patient_df['プロット用経過日数'], pd.to_numeric(current_patient_df['総合スコア'], errors='coerce'), marker='o', linestyle='-', linewidth=3, color='springgreen', label=f'治療中: {selected_active_patient}', zorder=15)
                                if prop:
                                    ax.set_title(f"【{selected_disease_group}】治療軌跡の重ね合わせ", fontsize=16, fontproperties=prop); ax.set_xlabel("ICU入室後経過日数", fontsize=16, fontproperties=prop)
                                    ax.set_ylabel("総合スコア", fontsize=16, fontproperties=prop); ax.legend(prop=prop)
                                    for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontproperties(prop)
                                else:
                                    ax.set_title(f"[{selected_disease_group}] Trajectory Overlay"); ax.set_xlabel("Days since ICU admission"); ax.set_ylabel("Total Score"); ax.legend()
                                ax.set_ylim(0, 105); ax.grid(True, linestyle='--', alpha=0.6); st.pyplot(fig)
                                st.write("---"); st.subheader("回復速度の可視化（日次スコア変化の平均）")
                                st.info("このグラフは、スコアが1日あたり平均してどれくらい変化したかを示しています。正の値が大きいほど回復の勢いが強く、負の値は状態の悪化を示唆します。回復が加速・停滞するタイミングを分析できます。")
                                all_changes = []
                                for patient_id in patient_ids:
                                    patient_df = group_df[group_df['アプリ用患者ID'] == patient_id].copy(); patient_df = patient_df.sort_values(by='プロット用日時')
                                    patient_df['スコア変化量'] = pd.to_numeric(patient_df['総合スコア'], errors='coerce').diff(); all_changes.append(patient_df[['経過日数', 'スコア変化量']])
                                if all_changes:
                                    all_changes_df = pd.concat(all_changes); average_speed = all_changes_df.groupby('経過日数')['スコア変化量'].mean()
                                    fig_speed, ax_speed = plt.subplots(figsize=(10, 5))
                                    average_speed.plot(kind='bar', ax=ax_speed, color=['skyblue' if x >= 0 else 'salmon' for x in average_speed.values])
                                    ax_speed.axhline(0, color='grey', linewidth=0.8)
                                    if prop:
                                        ax_speed.set_title(f"【{selected_disease_group}】回復速度", fontsize=16, fontproperties=prop); ax_speed.set_xlabel("ICU入室後経過日数", fontsize=16, fontproperties=prop)
                                        ax_speed.set_ylabel("前日からの平均スコア変化量", fontsize=16, fontproperties=prop)
                                        for label in ax_speed.get_xticklabels() + ax_speed.get_yticklabels(): label.set_fontproperties(prop)
                                    else:
                                        ax_speed.set_title(f"[{selected_disease_group}] Recovery Speed"); ax_speed.set_xlabel("Days since ICU admission"); ax.set_ylabel("Avg. Daily Score Change")
                                    ax_speed.grid(True, axis='y', linestyle='--', alpha=0.6); st.pyplot(fig_speed)
                        else: st.info("分析対象の疾患群がデータにありません。")
                    with tab2:
                        st.subheader("各フェーズの滞在日数の分布")
                        st.info("この箱ひげ図は、各フェーズに滞在した日数の分布を疾患群ごとに比較しています。箱の長さが短いほど日数のばらつきが少なく、治療期間が安定していることを示唆します。治療が長引きやすいフェーズの特定に役立ちます。")
                        days_in_phase = archived_df_dashboard.groupby(['アプリ用患者ID', '疾患群', 'フェーズ'], observed=False).size().reset_index(name='勤務帯の数')
                        days_in_phase['日数'] = days_in_phase['勤務帯の数'] / 2.0
                        fig, ax = plt.subplots(figsize=(12, 7))
                        sns.boxplot(data=days_in_phase, x='疾患群', y='日数', hue='フェーズ', ax=ax)
                        if prop:
                            ax.set_title("疾患群ごとのフェーズ別滞在日数", fontsize=16, fontproperties=prop); ax.set_xlabel("疾患群", fontsize=16, fontproperties=prop)
                            ax.set_ylabel("滞在日数", fontsize=16, fontproperties=prop); legend = ax.legend(prop=prop, title='フェーズ'); plt.setp(legend.get_title(), fontproperties=prop)
                            for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontproperties(prop)
                        else:
                            ax.set_title("Days in Each Phase per Disease Group"); ax.set_xlabel("Disease Group"); ax.set_ylabel("Days"); ax.legend(title='Phase')
                        plt.xticks(rotation=30, ha='right'); st.pyplot(fig)
                        st.write("---"); st.subheader("重要指標サマリー")
                        st.info("以下の表は、疾患群ごとの主要な臨床指標をまとめたものです。日数は「中央値 [四分位範囲]」、率は「パーセント (該当者数/全体数)」で表示しています。")
                        patient_counts = archived_df_dashboard.groupby('疾患群')['アプリ用患者ID'].nunique()
                        los_per_patient = archived_df_dashboard.groupby('アプリ用患者ID')['経過日数'].max()
                        patient_to_group = archived_df_dashboard.drop_duplicates(subset='アプリ用患者ID').set_index('アプリ用患者ID')['疾患群']
                        los_df = pd.DataFrame({'ICU滞在日数': los_per_patient, '疾患群': patient_to_group}); grouped_los = los_df.groupby('疾患群')['ICU滞在日数']
                        median_los = grouped_los.median(); q1_los = grouped_los.quantile(0.25); q3_los = grouped_los.quantile(0.75)
                        milestone_events = ["抜管", "SBT成功", "昇圧薬離脱", "補助循環離脱", "腎代替療法終了"]; milestone_results = {}
                        for event in milestone_events:
                            event_df = archived_df_dashboard[archived_df_dashboard['イベント'].fillna('').str.split(r'\s*,\s*', regex=True).apply(lambda x: event in x)]
                            days_to_event = event_df.groupby('アプリ用患者ID')['経過日数'].min()
                            event_days_df = pd.merge(days_to_event, patient_to_group, on='アプリ用患者ID'); grouped_event_days = event_days_df.groupby('疾患群')['経過日数']
                            milestone_results[event] = {"median": grouped_event_days.median(), "q1": grouped_event_days.quantile(0.25), "q3": grouped_event_days.quantile(0.75)}
                        complication_events = ["再挿管", "気管切開", "新規不整脈", "出血イベント", "せん妄", "新規感染症"]; complication_results = {}
                        for event in complication_events:
                            patients_with_event = archived_df_dashboard[archived_df_dashboard['イベント'].fillna('').str.split(r'\s*,\s*', regex=True).apply(lambda x: event in x)]['アプリ用患者ID'].unique()
                            complication_rates = patient_to_group.to_frame().groupby('疾患群').apply(lambda g: pd.Series({'count': len([pid for pid in patients_with_event if pid in g.index]), 'total': len(g), 'rate': len([pid for pid in patients_with_event if pid in g.index]) / len(g) * 100 if len(g) > 0 else 0}), include_groups=False)
                            complication_results[event] = complication_rates
                        disease_groups = archived_df_dashboard['疾患群'].dropna().unique()
                        index_names = ["患者数 (人)", "ICU総滞在日数 (中央値 [IQR])"] + [f"{e}までの日数 (中央値 [IQR])" for e in milestone_events] + [f"{e} 経験率 (%)" for e in complication_events]
                        summary_df = pd.DataFrame(index=index_names, columns=disease_groups)
                        for group in disease_groups:
                            summary_df.loc["患者数 (人)", group] = f"{patient_counts.get(group, 0)}"
                            los_text = f"{median_los.get(group, 0):.1f} [{q1_los.get(group, 0):.1f} - {q3_los.get(group, 0):.1f}]"
                            summary_df.loc["ICU総滞在日数 (中央値 [IQR])", group] = los_text
                            for event in milestone_events:
                                median = milestone_results[event]['median'].get(group)
                                if pd.notna(median):
                                    q1 = milestone_results[event]['q1'].get(group); q3 = milestone_results[event]['q3'].get(group)
                                    summary_df.loc[f"{event}までの日数 (中央値 [IQR])", group] = f"{median:.1f} [{q1:.1f} - {q3:.1f}]"
                            for event in complication_events:
                                result_for_event = complication_results[event]
                                if not result_for_event.empty and group in result_for_event.index:
                                    rate_info = result_for_event.loc[group]
                                    summary_df.loc[f"{event} 経験率 (%)", group] = f"{rate_info['rate']:.1f} ({int(rate_info['count'])}/{int(rate_info['total'])})"
                        st.dataframe(summary_df.fillna("-"))

if __name__ == "__main__":
    run_app()