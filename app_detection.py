"""
多线程 Web 界面，默认不启用分类器过滤（稳定输出检测结果），用户可勾选启用实验性过滤
"""

from classifier_filter import load_classifier, predict_patch
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from io import BytesIO
import tempfile
import os
from PIL import Image
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# ========== 页面配置 ==========
st.set_page_config(page_title="肺结节检测系统", layout="wide", page_icon="🫁")

# ========== 自定义 CSS 美化（全局 + 侧边栏细节） ==========
st.markdown("""
<style>
    /* 全局背景渐变 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fc 0%, #eef2f9 100%);
    }
    /* 主容器卡片效果 */
    .main .block-container {
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(2px);
        border-radius: 32px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 20px 35px -12px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.02);
        border: 1px solid rgba(255,255,255,0.6);
    }
    /* 侧边栏磨砂玻璃效果 + 圆润内边距 */
    .css-1d391kg, .css-12oz5g0 {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 1.5rem;
        margin: 1rem 0.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.8);
    }
    /* 侧边栏内部元素间距优化 */
    .sidebar-content {
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
    }
    /* 参数分组标题 */
    .param-group {
        font-weight: 600;
        color: #1e466e;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
        font-size: 1rem;
        border-left: 3px solid #2c7be5;
        padding-left: 0.8rem;
    }
    /* 主标题渐变色 */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #5a6e85;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        border-bottom: 1px solid #e2e8f0;
        display: inline-block;
        padding-bottom: 0.5rem;
    }
    /* 按钮精致动画 */
    .stButton > button {
        background: linear-gradient(95deg, #2c7be5 0%, #1a68d1 100%);
        color: white;
        border-radius: 40px;
        border: none;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.2, 0.9, 0.4, 1.1);
        box-shadow: 0 4px 10px rgba(44,123,229,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px rgba(44,123,229,0.3);
        background: linear-gradient(95deg, #3a87f5 0%, #2c7be5 100%);
    }
    /* 成功/信息框微光 */
    .stAlert {
        border-radius: 20px;
        border-left: 4px solid #2c7be5;
        background: rgba(248,250,252,0.8);
        backdrop-filter: blur(4px);
        padding: 0.8rem 1.2rem;
    }
    /* 图像圆角深阴影 */
    .stImage {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 12px 24px -12px rgba(0,0,0,0.2);
        transition: transform 0.25s ease;
    }
    .stImage:hover {
        transform: scale(1.02);
    }
    /* 检测卡片 - 细腻玻璃态 */
    .detection-card {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(4px);
        border-radius: 28px;
        padding: 1.2rem;
        box-shadow: 0 10px 25px -8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.6);
    }
    .detection-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 35px -12px rgba(0,0,0,0.15);
        background: rgba(255,255,255,0.98);
    }
    /* 指标卡片 - 精致内阴影 */
    .metric-card {
        background: #f8fafc;
        border-radius: 24px;
        padding: 0.8rem;
        text-align: center;
        border: 1px solid #eef2f9;
        box-shadow: inset 0 1px 1px rgba(0,0,0,0.02), 0 2px 4px rgba(0,0,0,0.02);
    }
    /* 滑块与输入框圆润 */
    .stSlider > div > div > div {
        background-color: #e2e8f0;
        border-radius: 20px;
    }
    .stNumberInput input {
        border-radius: 40px;
        border: 1px solid #e2e8f0;
        padding: 0.5rem 1rem;
    }
    /* 复选框美化 */
    .stCheckbox label span {
        font-weight: 500;
        color: #2c3e50;
    }
    /* 扩展器圆角与动画 */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 28px;
        transition: all 0.2s;
    }
    .streamlit-expanderHeader:hover {
        background: #eef2f9;
    }
    /* 页脚细线 */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(0,0,0,0.05);
        color: #7f8c8d;
        font-size: 0.8rem;
        backdrop-filter: blur(2px);
    }
    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    /* 加载动画轻量优化 */
    .stSpinner > div {
        border-color: #2c7be5 transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)

# 主页面标题
st.markdown('<div class="main-title">肺结节智能检测系统</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">基于YOLOv8的自动化结节检测 | 支持假阳性过滤 | 一键生成诊断报告</div>', unsafe_allow_html=True)

# ========== 模型加载 ==========
@st.cache_resource
def load_detection_model(model_path="detection_model/best.pt"):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"检测模型加载失败: {e}")
        return None

@st.cache_resource
def load_classifier_model(model_path="classifier_model/best_resnet18.pth"):
    try:
        model = load_classifier(model_path, device='cpu')
        return model
    except Exception as e:
        st.warning(f"此网站只展示预测，暂不展示分类器的模型功能")
        return None

detection_model = load_detection_model()
classifier_model = load_classifier_model()

if detection_model is None:
    st.stop()

# ========== 侧边栏（参数栏）优化布局 ==========
with st.sidebar:
    # 顶部图标和欢迎语
    st.markdown("### 🫁 检测参数配置")
    st.markdown("---")
    
    # 线程数
    st.markdown('<div class="param-group">⚙️ 性能</div>', unsafe_allow_html=True)
    MAX_WORKERS = st.number_input("🧵 线程个数", value=2, step=1, min_value=1, max_value=8, 
                                   help="线程越多速度越快，建议≤4", key="max_workers")
    
    st.markdown('<div class="param-group">🎯 检测阈值</div>', unsafe_allow_html=True)
    det_conf = st.slider("检测置信度阈值", 0.1, 1.0, 0.3, 0.05, 
                         help="越高漏检越少，但可能增加假阳性", key="det_conf")
    
    st.markdown('<div class="param-group">🔬 实验性过滤</div>', unsafe_allow_html=True)
    enable_filter = st.checkbox("启用智能假阳性过滤（实验性）", value=False, 
                                help="开启后使用分类器过滤假阳性，可能影响召回率", key="enable_filter")
    if enable_filter:
        final_threshold = st.slider("📊 最终得分阈值 (检测置信度 × 分类器概率)", 0.0, 1.0, 0.2, 0.05, 
                                    help="推荐0.2~0.3", key="final_threshold")
    else:
        final_threshold = 0.0
    
    st.markdown('<div class="param-group">📐 图像参数</div>', unsafe_allow_html=True)
    spacing = st.number_input("📏 像素间距 (mm/pixel)", value=1.0, step=0.1, 
                              help="CT图像中每个像素对应的实际尺寸", key="spacing")
    
    st.markdown("---")
    uploaded_files = st.file_uploader("📤 选择 CT 图像", type=["png", "jpg", "jpeg"], 
                                      accept_multiple_files=True, key="uploader")
    st.markdown("<p style='font-size:0.8rem; color:#5a6e85; margin-top:-0.5rem;'>支持PNG/JPG格式，可批量上传</p>", unsafe_allow_html=True)

# ========== 检测与过滤函数 ==========
def detect_and_filter(img_data, det_conf, cls_model, final_thresh, spacing, enable_filter):
    try:
        img_rgb = img_data["rgb"]
        results = detection_model(img_rgb, conf=det_conf)
        result = results[0]
        filtered_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                if not enable_filter:
                    filtered_boxes.append(box)
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                roi = img_rgb[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                if cls_model is not None:
                    cls_prob = predict_patch(roi, cls_model, device='cpu')
                    final_score = float(box.conf[0]) * cls_prob
                    if final_score < final_thresh:
                        continue
                    box.cls_conf = cls_prob
                    box.final_conf = final_score
                else:
                    box.cls_conf = None
                    box.final_conf = float(box.conf[0])
                filtered_boxes.append(box)

        annotated_img = img_rgb.copy()
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = getattr(box, 'final_conf', float(box.conf[0]))
            if enable_filter:
                label = f"Fused Score: {conf:.2f}"
                if hasattr(box, 'cls_conf') and box.cls_conf is not None:
                    label += f" | cls:{box.cls_conf:.2f}"
            else:
                label = f"Nodule {conf:.2f}"
                if hasattr(box, 'cls_conf') and box.cls_conf is not None:
                    label += f" | cls:{box.cls_conf:.2f}"
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        detections = []
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = getattr(box, 'final_conf', float(box.conf[0]))
            cls_conf = getattr(box, 'cls_conf', None)
            w = x2 - x1
            h = y2 - y1
            diameter_mm = ((w + h) / 2) * spacing
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append({
                'det_conf': conf,
                'cls_conf': cls_conf,
                'bbox': [x1, y1, x2, y2],
                'center': center,
                'diameter_mm': diameter_mm
            })
        return {
            "name": img_data["name"],
            "annotated_img": annotated_img,
            "detections": detections,
            "num": len(detections)
        }
    except Exception as e:
        return {"name": img_data["name"], "error": str(e)}

# ========== 多线程处理 ==========
if uploaded_files:
    st.success(f"✅ 已上传 {len(uploaded_files)} 张图像")
    st.divider()

    image_list = []
    for f in uploaded_files:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image_list.append({
            "name": f.name,
            "rgb": img_rgb
        })

    results = []
    ctx = get_script_run_ctx()

    with st.spinner(f"正在使用 {MAX_WORKERS} 线程并发检测..."):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {}
            for img_data in image_list:
                future = executor.submit(
                    detect_and_filter,
                    img_data,
                    det_conf,
                    classifier_model,
                    final_threshold,
                    spacing,
                    enable_filter
                )
                future_map[future] = img_data
                add_script_run_ctx(future, ctx)

            for future in as_completed(future_map):
                res = future.result()
                if "error" in res:
                    st.error(f"❌ {res['name']} 检测失败：{res['error']}")
                    continue
                results.append(res)

    # 展示结果
    filter_status = "✅ 已启用" if enable_filter else "⚪ 未启用"
    st.markdown(f"### 🩺 检测结果（假阳性过滤{filter_status}）")
    total = sum(r["num"] for r in results)
    st.markdown(f"<div style='background:#eef2f7; padding:0.8rem; border-radius:16px; margin-bottom:1rem; border-left:4px solid #2c7be5;'>📊 共检测到 <strong>{total}</strong> 个结节</div>", unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, res in enumerate(results):
        with cols[idx % 2]:
            with st.container():
                st.markdown(f"<div class='detection-card'>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image_list[idx]["rgb"], caption="原始图像", use_container_width=True)
                with c2:
                    st.image(res["annotated_img"], caption="检测结果", use_container_width=True)
                
                if res["num"] > 0:
                    st.success(f"📋 检测到 {res['num']} 个结节")
                    for i, d in enumerate(res["detections"], 1):
                        with st.expander(f"🔍 结节 {i} 详细信息"):
                            col1, col2 = st.columns(2)
                            with col1:
                                if enable_filter:
                                    st.metric("融合得分", f"{d['det_conf']:.2%}")
                                else:
                                    st.metric("检测置信度", f"{d['det_conf']:.2%}")
                                if d['cls_conf'] is not None:
                                    st.metric("分类器概率", f"{d['cls_conf']:.2%}")
                            with col2:
                                st.metric("直径 (mm)", f"{d['diameter_mm']:.1f}")
                                st.metric("中心坐标", f"({d['center'][0]:.0f}, {d['center'][1]:.0f})")
                else:
                    st.info("🎉 未检测到结节")
                st.markdown(f"</div>", unsafe_allow_html=True)
                st.markdown("---")

    st.subheader("📄 批量诊断报告")
    
    def gen_report(results, spacing, det_conf, final_thresh, enable_filter):
        lines = ["="*50, "肺结节检测报告", "="*50]
        lines.append(f"时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"图像总数：{len(results)}")
        lines.append(f"线程数：{MAX_WORKERS}")
        lines.append(f"检测阈值：{det_conf}")
        if enable_filter:
            lines.append(f"最终得分阈值：{final_thresh}")
        else:
            lines.append("假阳性过滤：未启用")
        lines.append(f"像素间距：{spacing} mm/pixel")
        lines.append("="*50)
        for r in results:
            lines.append(f"\n【{r['name']}】")
            if r["num"] == 0:
                lines.append("  未检测到结节")
            else:
                lines.append(f"  检测到 {r['num']} 个结节：")
                for i, d in enumerate(r["detections"], 1):
                    cls_info = f" | 分类器概率：{d['cls_conf']:.2%}" if d['cls_conf'] else ""
                    lines.append(f"  结节{i}: 直径 {d['diameter_mm']:.1f}mm | 最终得分 {d['det_conf']:.2%}{cls_info}")
        lines.append("\n仅供辅助参考，不构成医疗诊断")
        return "\n".join(lines)

    # PDF生成部分
    _pdf_lock = threading.Lock()
    def generate_pdf_report(results, spacing, det_conf, final_thresh, enable_filter, is_single=False, single_res=None):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        c.setFont("Helvetica", 16)
        c.drawString(20 * mm, height - 20 * mm, "Lung Nodule Detection Report")
        c.setFont("Helvetica", 10)
        c.drawString(20 * mm, height - 25 * mm, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        c.drawString(20 * mm, height - 30 * mm, f"Pixel Spacing: {spacing} mm/pixel")
        c.drawString(20 * mm, height - 35 * mm, f"Detection Threshold: {det_conf}")
        if enable_filter:
            c.drawString(20 * mm, height - 40 * mm, f"Final Score Threshold: {final_thresh}")
        else:
            c.drawString(20 * mm, height - 40 * mm, "False Positive Filter: Disabled")
        y_pos = height - 50 * mm
        data_list = [single_res] if is_single else results
        for res in data_list:
            c.setFont("Helvetica", 12)
            c.drawString(20 * mm, y_pos, f"【{res['name']}】")
            y_pos -= 8 * mm
            c.setFont("Helvetica", 10)
            if res["num"] == 0:
                c.drawString(20 * mm, y_pos, "No nodules detected")
                y_pos -= 8 * mm
            else:
                c.drawString(20 * mm, y_pos, f"Detected {res['num']} nodule(s):")
                y_pos -= 8 * mm
                for i, d in enumerate(res["detections"], 1):
                    cls_info = f" | Classifier prob: {d['cls_conf']:.2%}" if d['cls_conf'] else ""
                    c.drawString(25 * mm, y_pos, f"Nodule {i}: Diameter {d['diameter_mm']:.1f}mm | Final Score {d['det_conf']:.2%}{cls_info}")
                    y_pos -= 6 * mm
            img = res["annotated_img"]
            img_w, img_h = 170 * mm, 120 * mm
            with _pdf_lock:
                tmp_filename = f"tmp_{uuid.uuid4().hex}.png"
                tmp_img_path = os.path.join(tempfile.gettempdir(), tmp_filename)
                Image.fromarray(img).save(tmp_img_path, format="PNG")
                c.drawImage(tmp_img_path, 20 * mm, y_pos - img_h, img_w, img_h)
                os.unlink(tmp_img_path)
            y_pos -= (img_h + 10 * mm)
            if y_pos < 40 * mm:
                c.showPage()
                y_pos = height - 20 * mm
        c.save()
        buffer.seek(0)
        return buffer

    if len(uploaded_files) == 1:
        if st.button("📄 生成并下载 PDF 诊断报告", use_container_width=True):
            pdf_buffer = generate_pdf_report(results, spacing, det_conf, final_threshold, enable_filter)
            filename = f"LungNodule_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button("💾 点击下载报告", pdf_buffer, file_name=filename, mime="application/pdf", use_container_width=True)
    else:
        st.info(f"🚀 正在用 {MAX_WORKERS} 线程并发生成PDF报告...")
        ctx = get_script_run_ctx()
        pdf_results = []
        def generate_single_pdf(res):
            try:
                pdf_buffer = generate_pdf_report(
                    results, spacing, det_conf, final_threshold, enable_filter,
                    is_single=True, single_res=res
                )
                return {"name": res["name"], "buffer": pdf_buffer, "success": True}
            except Exception as e:
                return {"name": res["name"], "error": str(e), "success": False}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {}
            for res in results:
                future = executor.submit(generate_single_pdf, res)
                future_map[future] = res
                add_script_run_ctx(future, ctx)
            for future in as_completed(future_map):
                res = future.result()
                if res["success"]:
                    pdf_results.append(res)
                else:
                    st.error(f"❌ {res['name']} PDF生成失败：{res['error']}")
        st.success("✅ 所有PDF生成完成")
        for res in pdf_results:
            filename = f"Report_{res['name']}.pdf"
            st.download_button(label=f"📄 下载 {res['name']} 的带图报告", data=res["buffer"], file_name=filename, mime="application/pdf")
else:
    st.info("📌 请在左侧上传图片开始检测")

# ========== 页脚 ==========
st.markdown("""
<div class="footer">
    🫁 模型基于 LUNA16 数据集训练 · YOLOv8 检测引擎 · 可选假阳性过滤为实验功能<br>
    📢 本系统仅供辅助参考，不构成医疗诊断
</div>
""", unsafe_allow_html=True)
