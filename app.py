import streamlit as st
import pandas as pd
import openai
import faiss
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="البحث الدلالي في نصوص ابن تيمية", layout="wide")
st.title("📚 البحث الدلالي في نصوص شيخ الإسلام ابن تيمية")

# الشريط الجانبي
openai_key = st.sidebar.text_input("🔐 أدخل مفتاح OpenAI", type="password")

model_choice = st.sidebar.selectbox(
    "🤖 اختر نموذج OpenAI",
    options=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    index=0
)

threshold = st.sidebar.slider(
    "🎚️ حد المسافة القصوى (كلما قلّ الرقم زادت الدقة)",
    min_value=0.1,
    max_value=1.0,
    value=0.35,
    step=0.05
)

embedding_model = "text-embedding-ada-002"

# تحميل البيانات
@st.cache_resource
def load_data():
    df = pd.read_csv("dar1_with_embeddings001.csv")
    index = faiss.read_index("faiss_dar1001.index")
    return df, index

df, index = load_data()

# دالة توليد embedding
def get_embedding(text):
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=embedding_model
    )
    return response.data[0].embedding

# البحث الدلالي مع فلترة النتائج
def search_semantic(query, top_k=10, threshold=0.35):
    query_vec = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, dist in enumerate(distances[0]):
        if dist < threshold:
            match = df.iloc[indices[0][i]]
            results.append((match, dist))

    return results

# توليد الشرح من الذكاء الاصطناعي
def explain_match(query, match_text):
    prompt = f"""سؤال المستخدم: "{query}"
النص من كلام ابن تيمية: "{match_text}"

اشرح العلاقة بين النص والسؤال بلغة علمية واضحة:"""

    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model=model_choice,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# إدخال السؤال
query = st.text_input("📝 أدخل سؤالك", placeholder="مثال: ما موقف ابن تيمية من تقديم العقل على النقل؟")

if query and openai_key:
    with st.spinner("🔎 جاري البحث..."):
        results = search_semantic(query, top_k=10, threshold=threshold)

        if not results:
            st.warning("❗ لم يتم العثور على نص قريب بما يكفي لهذا السؤال ضمن الحد المحدد.")
        else:
            for i, (row, dist) in enumerate(results):
                st.markdown(f"### 🔹 النص {i+1} (المسافة: {dist:.3f})")
                st.write(row['text'])

                with st.expander("🧠 تفسير العلاقة"):
                    explanation = explain_match(query, row['text'])
                    st.write(explanation)
