import streamlit as st
import requests

# ── config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Chatbot السفر",
    page_icon="🚌",
    layout="centered",
)

# ── RTL styling ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* enforce RTL for all text elements */
        body, .stMarkdown, .stChatMessage, p, div {
            direction: rtl;
            text-align: right;
        }
        /* keep chat input LTR-aware but text RTL */
        .stChatInputContainer textarea {
            direction: rtl;
            text-align: right;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ────────────────────────────────────────────────────────────────────
st.title("🚌 Chatbot الاستعلام عن خطوط السفر")

# ── backend health check ──────────────────────────────────────────────────────
@st.cache_data(ttl=10, show_spinner=False)
def is_backend_alive() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/docs", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

if not is_backend_alive():
    st.error("⚠️ لا يمكن الاتصال بالخادم. تأكد من تشغيل FastAPI على المنفذ 8000.")
    st.stop()

# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ الإعدادات")
    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"عدد الرسائل: {len(st.session_state.messages)}")

# ── render conversation history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("اكتب سؤالك هنا..."):

    # show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # call FastAPI and show assistant reply
    with st.chat_message("assistant"):
        with st.spinner("جاري البحث..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",          # ✅ correct endpoint
                    json={"query": prompt},           # ✅ correct field name
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["response"]         # ✅ correct response field

                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                elif response.status_code == 503:
                    st.warning("النظام لا يزال يُحمَّل، يرجى الانتظار لحظة والمحاولة مجدداً.")
                else:
                    st.error(f"خطأ من الخادم: {response.status_code} — {response.text}")

            except requests.exceptions.Timeout:
                st.error("انتهت مهلة الاتصال بالخادم. يرجى المحاولة مرة أخرى.")
            except requests.exceptions.ConnectionError:
                st.error("تعذّر الاتصال بالخادم. تأكد من أنه يعمل على المنفذ 8000.")
            except Exception as e:
                st.error(f"خطأ غير متوقع: {e}")
