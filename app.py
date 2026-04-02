import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle
import re
from groq import Groq
from dotenv import load_dotenv
import os

# ---------------------------
# Streamlit UI Modern Styling
# ---------------------------
st.set_page_config(page_title="AI Finance Tracker", page_icon="💰", layout="wide")

st.markdown("""
<style>
/* Main UI tune */
body, .stApp {
    background: linear-gradient(to bottom right, #0d1117, #1f2937);
    color: #e6e6e6;
    font-family: "Inter", sans-serif;
}

/* Header */
h1, h2, h3, .stTabs [data-baseweb="tab"] {
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    padding: 8px 14px;
    border-radius: 8px;
}
.stTabs [aria-selected="true"] {
    background: #111827;
    color: #10b981 !important;
    font-weight: 700;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #111827;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #10b98133;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput input, .stDateInput input, .stFileUploader {
    border-radius: 10px;
    background: #0f1724;
    color: #e6e6e6;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#10b981,#34d399);
    color: white;
    padding: 10px 16px;
    font-weight: 700;
    border-radius: 10px;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg,#059669,#10b981);
}

/* Chat bubbles */
.user-bubble {
    background: #1f2937;
    padding: 10px;
    border-radius: 14px;
    margin-bottom: 8px;
    color: white;
    width: fit-content;
    max-width: 75%;
    margin-left: 20%;
}
.bot-bubble {
    background: #064e3b;
    padding: 10px;
    border-radius: 14px;
    margin-bottom: 8px;
    color: white;
    width: fit-content;
    max-width: 75%;
    margin-right: 20%;
}

/* Small headings */
h2 {
    color: #e6e6e6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Groq client (loads API key from .env)
# ---------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        client = None

# ---------------------------------------
# ✅ Load classification model & vectorizer
# ---------------------------------------
with open("expense_model.pkl", "rb") as f:
    clf_model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    clf_vectorizer = pickle.load(f)


def predict_category(text):
    text_vec = clf_vectorizer.transform([text])
    return clf_model.predict(text_vec)[0]

# ---------------------------------------
# ✅ Auto-extract total amount
# ---------------------------------------
def extract_total_amount(text):
    patterns = [
        r"total\s*[:\-]?\s*₹?(\d+[\.,]?\d*)",
        r"grand\s*total\s*[:\-]?\s*₹?(\d+[\.,]?\d*)",
        r"amount\s*[:\-]?\s*₹?(\d+[\.,]?\d*)",
        r"rs\.?\s*(\d+[\.,]?\d*)",
        r"inr\s*(\d+[\.,]?\d*)",
        r"₹\s*(\d+[\.,]?\d*)",
    ]
    text = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

# ---------------------------------------
# ✅ Load or create DB
# ---------------------------------------
DATA_FILE = "expenses.csv"
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    df = pd.DataFrame(columns=["Date","Category","Description","Amount","Source"])

# ---------------------------------------
# ✅ UI Setup
# ---------------------------------------
st.title("💰 AI-Powered Finance Tracker")
st.caption("Upload receipts or enter expenses — OCR + AI auto categorization")

tab1, tab2, tab3, tab4 = st.tabs(["📸 Upload Receipt", "✍️ Manual Entry", "📊 Dashboard", "🤖 Finance Assistant"])

# ---------------------------------------
# ✅ TAB 1 — OCR Receipt Upload
# ---------------------------------------
with tab1:
    st.subheader("📸 Upload Receipt")
    uploaded_file = st.file_uploader("Upload Receipt", type=["png","jpg","jpeg"], key="upload_receipt")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Receipt Uploaded", use_container_width=True)

        raw_text = pytesseract.image_to_string(image)
        st.text_area("📜 Extracted Text", raw_text, height=180, key="extracted_text_area")

        predicted_category = predict_category(raw_text)
        st.info(f"🤖 Suggested Category: **{predicted_category}**")

        category = st.selectbox("Category",
            ["Food","Groceries","Transport","Shopping","Bills","Fuel","Other"],
            key="ocr_category",
            index=( ["Food","Groceries","Transport","Shopping","Bills","Fuel","Other"].index(predicted_category)
                    if predicted_category in ["Food","Groceries","Transport","Shopping","Bills","Fuel","Other"]
                    else 6)
        )

        auto_amount = extract_total_amount(raw_text)
        if auto_amount:
            st.success(f"🧾 Auto-Total Detected: ₹{auto_amount}")
            amount = st.number_input("Amount", value=float(auto_amount), key="ocr_amount")
        else:
            st.warning("⚠️ Total not detected. Enter manually.")
            amount = st.number_input("Amount", min_value=0.0, step=0.1, key="ocr_amount_manual")

        desc = st.text_input("Description", value="Receipt Expense", key="ocr_desc")

        if st.button("💾 Save", key="save_ocr"):
            new_row = {
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Category": category,
                "Description": desc,
                "Amount": float(amount),
                "Source": "OCR"
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success("✅ Saved successfully!")

# ---------------------------------------
# ✅ TAB 2 — Manual Entry
# ---------------------------------------
with tab2:
    st.subheader("✍️ Add Expense Manually")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", datetime.now(), key="manual_date")
        category = st.selectbox("Category", ["Food","Groceries","Transport","Shopping","Bills","Fuel","Other"], key="manual_category")
    with col2:
        amount = st.number_input("Amount", min_value=0.0, step=0.1, key="manual_amount")
        desc = st.text_input("Description", key="manual_desc")

    if st.button("➕ Add Expense", key="add_manual"):
        new_row = {
            "Date": date.strftime("%Y-%m-%d"),
            "Category": category,
            "Description": desc,
            "Amount": float(amount),
            "Source": "Manual"
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        st.success("✅ Expense Added!")

# ---------------------------------------
# ✅ TAB 3 — Dashboard
# ---------------------------------------
with tab3:
    st.subheader("📊 Expense Dashboard")

    if df.empty:
        st.warning("No data yet.")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
        total_spent = df["Amount"].sum()

        colA, colB = st.columns(2)
        colA.metric("Total Spent", f"₹{total_spent:,.2f}")
        colB.metric("Total Transactions", len(df))

        st.write("### 📂 By Category")
        cat_sums = df.groupby("Category")["Amount"].sum()
        fig, ax = plt.subplots()
        ax.pie(cat_sums, labels=cat_sums.index, autopct="%1.1f%%")
        st.pyplot(fig)

        st.write("### 📈 Daily Spending")
        ts = df.groupby("Date")["Amount"].sum()
        fig2, ax2 = plt.subplots()
        ax2.plot(ts.index, ts.values, marker="o")
        ax2.set_ylabel("Amount")
        st.pyplot(fig2)

        st.write("### 📋 Expense History")
        st.dataframe(df.sort_values("Date", ascending=False))

# ---------------------------------------
# ✅ TAB 4 — Smart Finance Assistant (Groq + fallback)
# ---------------------------------------
with tab4:
    st.subheader("🤖 Smart Finance Assistant")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user_msg = st.text_input("Ask anything about your expenses:", key="llm_user_input")

    def rule_based_reply(query, df_local):
        q = query.lower()
        if "total" in q and ("spend" in q or "spent" in q):
            total = df_local["Amount"].sum() if not df_local.empty else 0.0
            return f"You've spent ₹{total:,.2f} so far."
        for cat in df_local["Category"].unique() if not df_local.empty else []:
            if cat.lower() in q:
                total_cat = df_local[df_local["Category"] == cat]["Amount"].sum()
                return f"You've spent ₹{total_cat:,.2f} on {cat}."
        if any(x in q for x in ["tip","advice","save"]):
            return "Tip: Set monthly limits and review recurring subscriptions."
        if any(g in q for g in ["hi","hello","hey"]):
            return "Hello! Ask me 'total spent', 'spent on food', or 'give saving tips'."
        return None

    def build_spending_context(df_local, top_n=5):
        if df_local.empty:
            return "No expenses recorded yet."
        df_c = df_local.copy()
        df_c["Amount"] = pd.to_numeric(df_c["Amount"], errors="coerce").fillna(0)
        totals = df_c.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        totals_text = "\n".join([f"- {cat}: ₹{amt:,.2f}" for cat, amt in totals.items()])
        last_items = df_c.sort_values("Date").tail(top_n)[["Date","Category","Description","Amount"]]
        last_text = "\n".join([f"{r.Date} | {r.Category} | {r.Description} | ₹{r.Amount}" for r in last_items.itertuples(index=False)])
        overall_total = df_c["Amount"].sum()
        ctx = (
            f"Overall total: ₹{overall_total:,.2f}\n"
            f"Category totals:\n{totals_text}\n\n"
            f"Last {top_n} transactions:\n{last_text}"
        )
        return ctx

    if st.button("Send", key="llm_send"):
        if user_msg.strip() == "":
            st.warning("Please type a question first.")
        else:
            # Try quick local answer first
            local_ans = rule_based_reply(user_msg, df)
            if local_ans:
                reply_text = local_ans
            else:
                # build prompt with spending summary
                context = build_spending_context(df)
                system_prompt = (
                    "You are a helpful finance assistant. Use the spending summary to answer the user's question. "
                    "Keep answers short and include currency symbol ₹ where appropriate."
                )
                prompt = system_prompt + "\n\nSPENDING SUMMARY:\n" + context + "\n\nUSER QUESTION:\n" + user_msg + "\n\nAnswer:"

                # Call Groq LLM if client available
                if client is not None:
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": context},
                                {"role": "user", "content": user_msg}
                            ],
                            max_tokens=300,
                            temperature=0.15
                        )
                        reply_text = response.choices[0].message.content
                    except Exception:
                        reply_text = "LLM call failed — falling back to local assistant.\n\n" + (rule_based_reply(user_msg, df) or "I couldn't answer that.")
                else:
                    reply_text = "No LLM key configured — using local assistant.\n\n" + (rule_based_reply(user_msg, df) or "I couldn't answer that.")

            st.session_state.chat.append(("You", user_msg))
            st.session_state.chat.append(("Assistant", reply_text))

    # Render chat bubbles
    for sender, msg in st.session_state.chat:
        if sender == "You":
            st.markdown(f"<div class='user-bubble'><b>You:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'><b>Assistant:</b> {msg}</div>", unsafe_allow_html=True)
