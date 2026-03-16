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

# 自定义 CSS（美化）
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
.label-col {
    width: 150px;
    font-weight: bold;
}
.input-col {
    flex: 1;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 检查必要文件是否存在
# ----------------------------------------------------------
required_files = ["rf_model.pkl", "le_asa.pkl", "feature_ranges.csv", "asa_levels.csv"]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"缺少以下文件：{', '.join(missing_files)}。请确保这些文件与 app.py 在同一目录，并且已从 R 导出。")
    st.stop()

# ----------------------------------------------------------
# 加载模型和编码器（缓存）
# ----------------------------------------------------------
@st.cache_resource
def load_model():
    model  = joblib.load("rf_model.pkl")
    le_asa = joblib.load("le_asa.pkl")
    return model, le_asa

# ----------------------------------------------------------
# 从 CSV 构建特征范围字典（缓存）
# ----------------------------------------------------------
@st.cache_data
def load_feature_ranges():
    range_df   = pd.read_csv("feature_ranges.csv")
    asa_df     = pd.read_csv("asa_levels.csv")
    asa_levels = asa_df['asa_levels'].tolist()

    feature_ranges = {}
    for _, row in range_df.iterrows():
        feature = row['Feature']
        step = 0.1 if feature in ["HGS", "TP"] else 1.0
        if feature == "KPS":
            feature_ranges[feature] = {
                "type":    "numerical",
                "min":     None,
                "max":     None,
                "default": float(row['Median']),
                "step":    step
            }
        else:
            feature_ranges[feature] = {
                "type":    "numerical",
                "min":     float(row['Min']),
                "max":     float(row['Max']),
                "default": float(row['Median']),
                "step":    step
            }

    # 添加 ASA 分类变量
    feature_ranges["ASA"] = {
        "type":    "categorical",
        "options": [0, 1],
        "label":   asa_levels,
        "default": 0
    }
    return feature_ranges

# ----------------------------------------------------------
# 加载数据
# ----------------------------------------------------------
model, le_asa    = load_model()
feature_ranges   = load_feature_ranges()

# ----------------------------------------------------------
# 页面标题
# ----------------------------------------------------------
st.title("衰弱风险预测系统")
st.markdown("请输入患者临床指标，点击预测按钮获取风险概率和特征贡献解释。")

# ----------------------------------------------------------
# 临床指标输入模块
# ----------------------------------------------------------
with st.container():
    st.markdown('<div class="card"><h3 class="section-title">临床指标输入</h3>', unsafe_allow_html=True)
    cols = st.columns(3)
    feature_values = []

    for idx, (feature, props) in enumerate(feature_ranges.items()):
        with cols[idx % 3]:
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 15px;">'
                f'<div class="label-col">{feature}</div>'
                f'<div class="input-col">',
                unsafe_allow_html=True
            )
            if props["type"] == "numerical":
                value = st.number_input(
                    label            = feature,
                    min_value        = props["min"],
                    max_value        = props["max"],
                    value            = props["default"],
                    step             = props["step"],
                    format           = "%.1f" if props["step"] < 1 else "%.0f",
                    label_visibility = "collapsed"
                )
            else:   # categorical (ASA)
                selected_label = st.selectbox(
                    label            = feature,
                    options          = props["label"],
                    index            = props["default"],
                    label_visibility = "collapsed"
                )
                value = props["options"][props["label"].index(selected_label)]

            feature_values.append(value)
            st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------
# 预测按钮
# ----------------------------------------------------------
if st.button("预测风险", type="primary"):

    # ── 1. 构建输入 DataFrame ───────────────────────────────
    input_df = pd.DataFrame([feature_values], columns=list(feature_ranges.keys()))
    input_df['ASA'] = input_df['ASA'].astype(int)
    # 确保列顺序与模型训练时完全一致
    input_df = input_df[model.feature_names_in_]

    # ── 2. 预测概率和类别 ────────────────────────────────────
    prob       = model.predict_proba(input_df)[0, 1]
    pred_class = model.predict(input_df)[0]

    # ── 3. 显示结果卡片 ──────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("预测风险概率", f"{prob:.1%}")
    with col2:
        risk_level = "高风险 🔴" if prob > 0.5 else "低风险 🟢"
        st.metric("风险等级", risk_level)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 4. SHAP 瀑布图 ───────────────────────────────────────
    st.markdown(
        '<div class="card"><h3 class="section-title">特征贡献解释 (SHAP 瀑布图)</h3>',
        unsafe_allow_html=True
    )

    try:
        # 4-1. 创建解释器并计算 SHAP 值
        explainer       = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(input_df)
        expected_raw    = explainer.expected_value

        # 4-2. 兼容新旧两种 SHAP API
        #   旧版（shap < 0.42）：shap_values 返回 list，
        #                        list[0] = 负类数组，list[1] = 正类数组
        #   新版（shap >= 0.42）：shap_values 返回单个 ndarray，
        #                        形状可能为 (n_samples, n_features, n_classes)
        #                        或 (n_samples, n_features)

        if isinstance(shap_values_raw, list):
            # ── 旧版 API ──────────────────────────────────────
            shap_vals  = shap_values_raw[1][0]               # 正类，第一个样本
            base_value = float(np.squeeze(expected_raw[1]))  # 正类基准值

        else:
            # ── 新版 API ──────────────────────────────────────
            if shap_values_raw.ndim == 3:
                # 形状：(n_samples, n_features, n_classes)
                shap_vals = shap_values_raw[0, :, 1]         # 第一个样本，正类
                if hasattr(expected_raw, '__len__'):
                    base_value = float(np.squeeze(expected_raw[1]))
                else:
                    base_value = float(expected_raw)
            else:
                # 形状：(n_samples, n_features)
                shap_vals  = shap_values_raw[0]
                base_value = float(np.squeeze(expected_raw))

        # 4-3. 构建 Explanation 对象
        data_vals     = input_df.iloc[0].values.astype(float).tolist()
        feature_names = input_df.columns.tolist()

        exp = shap.Explanation(
            values        = shap_vals,
            base_values   = base_value,
            data          = data_vals,
            feature_names = feature_names
        )

        # 4-4. 绘制瀑布图
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(exp, show=False, max_display=len(feature_names))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)   # 释放内存，防止泄漏

    except Exception as e:
        st.error(f"SHAP 解释生成失败：{e}")
        st.code(traceback.format_exc())   # 显示完整报错，方便排查

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------
# 页脚
# ----------------------------------------------------------
st.markdown("---")
st.caption("基于加权随机森林模型开发，SHAP 提供可解释性。")