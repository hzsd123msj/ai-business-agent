import streamlit as st
from src.hybrid_agent import answer_question

st.set_page_config(page_title="AI业务分析助手", layout="wide")

st.title("📊 AI业务分析助手（RAG + Agent）")

st.markdown("请输入你的业务问题，例如：")
st.code("为什么AI营销助手销量下降？请结合产品资料给出建议")

query = st.text_input("请输入问题")

if st.button("开始分析"):
    if not query.strip():
        st.warning("请输入问题")
    else:
        with st.spinner("正在分析中..."):
            try:
                result = answer_question(query)

                st.success("分析完成")

                # =====================
                # 1️⃣ 分析结果
                # =====================
                st.markdown("## 📌 分析结果")
                st.write(result["answer"])

                # =====================
                # 2️⃣ 数据表展示
                # =====================
                if result["data"] is not None:
                    st.markdown("## 📊 数据依据")
                    st.dataframe(result["data"])

                # =====================
                # 3️⃣ 文档来源
                # =====================
                if result["docs"]:
                    st.markdown("## 📄 知识来源")

                    for i, doc in enumerate(result["docs"], start=1):
                        source = doc.metadata.get("source", "未知来源")
                        content = doc.page_content[:300]

                        st.markdown(f"**文档 {i}：{source}**")
                        st.write(content + "...")

            except Exception as e:
                st.error(f"出错了: {e}")