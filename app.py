import streamlit as st
import requests

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

st.title("SHL Assessment Recommender")

st.write("Enter a job description or query to get relevant SHL assessment recommendations:")

query = st.text_area("Your input", placeholder="e.g., We are hiring an entry-level sales associate...")

if st.button("Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Getting recommendations..."):
            try:
                response = requests.post(
                    "https://shl-assessment-recommender-lq61.onrender.com/recommend",
                    json={"query": query}
                )
                if response.status_code == 200:
                    results = response.json()["results"]
                    if not results:
                        st.info("No relevant assessments found.")
                    else:
                        for i, r in enumerate(results, 1):
                            st.markdown(f"### {i}. [{r['name']}]({r['url']})")
                            st.write(f"**Remote Support:** {r['remote']}")
                            st.write(f"**IRT/Adaptive:** {r['adaptive']}")
                            st.write(f"**Duration:** {r['duration']}")
                            st.write(f"**Type:** {r['type']}")
                            st.markdown("---")
                else:
                    st.error(f"API error: {response.status_code}")
            except Exception as e:
                st.error(f"Request failed: {e}")
