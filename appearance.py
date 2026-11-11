"""Industrial illustration and dashboard layout."""
import base64
from pathlib import Path

import streamlit as st


@st.cache_data
def hero_image():
    asset = Path(__file__).resolve().parent / "assets" / "motor_industrial.webp"
    return base64.b64encode(asset.read_bytes()).decode("ascii")


def render_hero():
    st.markdown("""
    <style>
    .stMain .block-container{max-width:1340px;padding-top:3.15rem;padding-bottom:3rem}
    .app-hero{position:relative;display:flex;align-items:center;isolation:isolate;overflow:hidden;
      min-height:310px;margin-bottom:1.45rem;border:1px solid #233c52;border-radius:24px;background:#0c1727;
      box-shadow:0 15px 35px rgba(12,30,47,.21),0 3px 7px rgba(12,30,47,.14)}
    .hero-render{position:absolute;inset:0;background-size:cover;background-position:center 53%;z-index:-2}
    .app-hero:after{content:"";position:absolute;inset:0;z-index:-1;
      background:linear-gradient(90deg,#0a1627 0%,rgba(10,22,39,.97) 24%,rgba(10,22,39,.72) 48%,rgba(10,22,39,.09) 76%)}
    .hero-copy{width:66%;max-width:700px;padding:2.4rem 2.8rem}
    .app-hero h1{color:#f1f9fc;padding:.12rem 0 .65rem;margin:0;max-width:550px;font-size:clamp(1.7rem,4.5vw,3.45rem);line-height:1.06;letter-spacing:-.035em}
    .app-hero p{color:#dae8f1;margin:0;max-width:470px;font-size:1rem}
    .hero-source{color:#87e7df;font-weight:720;letter-spacing:.13em;font-size:.73rem}
    [data-testid="stMetric"]{border:1px solid #dbe5ef;border-radius:13px;padding:.65rem .8rem;
      box-shadow:0 3px 9px rgba(18,41,60,.05);background:var(--secondary-background-color)}
    [data-testid="stMetricValue"]{font-size:clamp(1.02rem,2.6vw,1.85rem)}
    [data-testid="stForm"]{border-radius:16px;border-color:#cfdde9;box-shadow:0 4px 13px rgba(18,41,60,.05)}
    [data-testid="stExpander"] details{border-radius:12px}
    @media(max-width:720px){
      .stMain .block-container{padding-top:2.7rem}.app-hero{min-height:275px;border-radius:17px}
      .hero-copy{width:100%;padding:1.3rem 1.25rem}.hero-render{background-position:65% center}
      .app-hero:after{background:linear-gradient(90deg,rgba(10,22,39,.97),rgba(10,22,39,.76) 69%,rgba(10,22,39,.31))}
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <style>.hero-render{{background-image:url("data:image/webp;base64,{hero_image()}")}}</style>
    <header class="app-hero">
      <div class="hero-render" role="img" aria-label="Ảnh minh họa 3D động cơ công nghiệp"></div>
      <div class="hero-copy">
        <span class="hero-source">AI4I 2020 · BẢO TRÌ DỰ ĐOÁN</span>
        <h1>Dự đoán hỏng hóc máy</h1>
        <p>Dùng số đo vận hành, ảnh phiếu đo hoặc bảng tính để nhận cảnh báo và kiểm tra kết quả.</p>
      </div>
    </header>
    """, unsafe_allow_html=True)
