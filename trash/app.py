import streamlit as st
import requests

# تكوين عنوان الـ backend (شغّل FastAPI على منفذ 8000 محلياً)
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Chatbot السفر", page_icon="🚌")
st.title("🚌 Chatbot الاستعلام عن خطوط السفر")

# تهيئة حالة الجلسة لتخزين سجل المحادثة
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثات السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# مربع إدخال السؤال
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    # إضافة سؤال المستخدم إلى سجل المحادثة
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # إرسال السؤال إلى FastAPI
    with st.chat_message("assistant"):
        with st.spinner("جاري البحث..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": prompt}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    # عرض الإجابة
                    st.markdown(answer)

                    # عرض المصادر في expander
                    with st.expander("عرض المصادر"):
                        for i, src in enumerate(sources, 1):
                            st.write(f"**المصدر {i}:**")
                            st.write(src)
                            st.write("---")

                    # حفظ رد المساعد في سجل المحادثة
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"حدث خطأ في الاتصال بالخادم: {response.status_code}")
            except Exception as e:
                st.error(f"تعذر الاتصال بالخادم: {e}")