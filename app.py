import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import os
import traceback

# ----------------------------------------------------------
# 页面配置
# ----------------------------------------------------------
st.set_page_config(page_title="衰弱风险预测系统", layout="wide")

st.markdown("""
<style>
.card {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.section-title {
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 5px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 检查必要文件
# 【修改1】移除 le_asa.pkl（ASA 已在 R 中哑变量编码，不再需要）
# ----------------------------------------------------------
required_files = ["rf_model.pkl", "feature_ranges.csv", "asa_levels.csv"]
missing_files  = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"缺少以下文件：{', '.join(missing_files)}，请确保与 app.py 在同一目录。")
    st.stop()

# ----------------------------------------------------------
# 加载模型（缓存）
# 【修改2】移除 le_asa 加载
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    return model

# ----------------------------------------------------------
# 构建特征范围字典（缓存）
# 【修改3】
#   - 连续变量从 feature_ranges.csv 读取（HGS/KPS/PG_SGA/SDS/Age/TP）
#   - ASA 从 asa_levels.csv 读取水平，映射到哑变量列名
#     R 的 step_dummy(ASA) 保留非参考水平（Ⅲ~Ⅳ）作为哑变量
#     列名由 model.feature_names_in_ 自动获取，无需硬编码
# ----------------------------------------------------------
@st.cache_data
def load_feature_ranges(asa_dummy_col):
    range_df   = pd.read_csv("feature_ranges.csv")
    asa_df     = pd.read_csv("asa_levels.csv")
    asa_levels = asa_df["asa_levels"].tolist()   # ["Ⅰ~Ⅱ", "Ⅲ~Ⅳ"]

    feature_ranges = {}

    # 连续变量
    for _, row in range_df.iterrows():
        feat = row["Feature"]
        step = 0.1 if feat in ["HGS", "TP"] else 1.0
        feature_ranges[feat] = {
            "type":    "numerical",
            "min":     float(row["Min"])    if pd.notna(row["Min"])    else None,
            "max":     float(row["Max"])    if pd.notna(row["Max"])    else None,
            "default": float(row["Median"]) if pd.notna(row["Median"]) else 0.0,
            "step":    step
        }

    # ASA 分类变量
    # asa_levels[0] = 参考水平（Ⅰ~Ⅱ） → 哑变量 = 0
    # asa_levels[1] = 非参考水平（Ⅲ~Ⅳ）→ 哑变量 = 1
    feature_ranges["ASA"] = {
        "type":       "categorical",
        "dummy_col":  asa_dummy_col,   # 实际哑变量列名，如 "ASA_Ⅲ.Ⅳ"
        "options":    [0, 1],          # 0 = Ⅰ~Ⅱ，1 = Ⅲ~Ⅳ
        "label":      asa_levels,      # 界面显示的文字
        "default":    0
    }
    return feature_ranges

# ── 加载模型，自动获取 ASA 哑变量列名 ────────────────────────────
model = load_model()

# 从模型特征列表中找到 ASA 相关的哑变量列名
asa_dummy_col = None
for col in model.feature_names_in_:
    if col.startswith("ASA"):
        asa_dummy_col = col
        break

if asa_dummy_col is None:
    st.error("模型特征中未找到 ASA 相关列，请检查 rf_model.pkl 是否正确导出。")
    st.stop()

feature_ranges = load_feature_ranges(asa_dummy_col)

# ----------------------------------------------------------
# 页面标题
# ----------------------------------------------------------
st.title("衰弱风险预测系统")
st.markdown("请输入患者临床指标，点击预测按钮获取风险概率和特征贡献解释。")

# ----------------------------------------------------------
# 临床指标输入模块
# ----------------------------------------------------------
with st.container():
    st.markdown('<div class="card"><h3 class="section-title">临床指标输入</h3>',
                unsafe_allow_html=True)
    cols = st.columns(3)

    # 用字典收集输入值，key = 显示名称，value = 实际输入值
    input_display = {}

    for idx, (feature, props) in enumerate(feature_ranges.items()):
        with cols[idx % 3]:
            if props["type"] == "numerical":
                val = st.number_input(
                    label     = feature,
                    min_value = props["min"],
                    max_value = props["max"],
                    value     = props["default"],
                    step      = props["step"],
                    format    = "%.1f" if props["step"] < 1 else "%.0f"
                )
                input_display[feature] = val

            else:   # ASA 分类变量
                selected_label = st.selectbox(
                    label   = feature,
                    options = props["label"],
                    index   = props["default"]
                )
                # 将界面标签映射为哑变量值（0 或 1）
                dummy_val = props["options"][props["label"].index(selected_label)]
                input_display[feature] = dummy_val

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------
# 预测按钮
# ----------------------------------------------------------
if st.button("预测风险", type="primary"):

    # ── 1. 构建输入 DataFrame（列名与模型一致）──────────────────
    # 【修改4】ASA 直接用哑变量列名，不再经过 LabelEncoder
    row = {}
    for feature, val in input_display.items():
        if feature == "ASA":
            # 用实际哑变量列名（如 ASA_Ⅲ.Ⅳ）
            row[asa_dummy_col] = val
        else:
            row[feature] = val

    input_df = pd.DataFrame([row])

    # 确保列顺序与训练时完全一致
    input_df = input_df[model.feature_names_in_]

    # ── 2. 预测概率 ─────────────────────────────────────────
    prob       = model.predict_proba(input_df)[0, 1]
    pred_class = model.predict(input_df)[0]

    # ── 3. 结果展示 ──────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("预测风险概率", f"{prob:.1%}")
    with col2:
        risk_level = "高风险 🔴" if prob > 0.5 else "低风险 🟢"
        st.metric("风险等级", risk_level)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 4. SHAP 瀑布图 ────────────────────────────────────────
    st.markdown(
        '<div class="card"><h3 class="section-title">特征贡献解释（SHAP 瀑布图）</h3>',
        unsafe_allow_html=True
    )
    try:
        explainer       = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(input_df)
        expected_raw    = explainer.expected_value

        # 兼容新旧 SHAP API
        if isinstance(shap_values_raw, list):
            shap_vals  = shap_values_raw[1][0]
            base_value = float(np.squeeze(expected_raw[1]))
        else:
            arr = np.array(shap_values_raw)
            if arr.ndim == 3:
                shap_vals  = arr[0, :, 1]
                base_value = float(np.squeeze(expected_raw[1])) \
                             if hasattr(expected_raw, "__len__") \
                             else float(expected_raw)
            else:
                shap_vals  = arr[0]
                base_value = float(np.squeeze(expected_raw))

        exp = shap.Explanation(
            values        = shap_vals,
            base_values   = base_value,
            data          = input_df.iloc[0].values.astype(float).tolist(),
            feature_names = input_df.columns.tolist()
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(exp, show=False,
                              max_display=len(input_df.columns))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"SHAP 解释生成失败：{e}")
        st.code(traceback.format_exc())

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------
# 页脚
# ----------------------------------------------------------
st.markdown("---")
st.caption("基于加权随机森林模型开发，SHAP 提供可解释性。")
