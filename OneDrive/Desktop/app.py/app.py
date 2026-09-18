import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


faqs = {
    "What is CodeAlpha?": "CodeAlpha is a software development company providing internship programs in emerging technologies like AI.",
    "How many tasks do I need to complete?": "You need to complete at least 2 or 3 tasks to receive your certificate.",
    "Where do I submit my tasks?": "You must submit your tasks through the Submission Form link shared in your official WhatsApp group.",
    "How should I name my GitHub repository?": "Name your repository using the format: CodeAlpha_ProjectName.",
    "What are the requirements for completion?": "You must push your code to GitHub, share a demo video on LinkedIn tagging @CodeAlpha, and fill out the submission form.",
}

faq_questions = list(faqs.keys())


st.set_page_config(page_title="FAQ Chatbot", page_icon="֎")
st.title("🤖 FAQ Assistant")
st.write("Ask a question related to your internship or project details!")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



def get_best_match(user_query):
    
    corpus = faq_questions + [user_query]
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()

    
    similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]
    best_idx = similarities.argmax()

    
    if similarities[best_idx] < 0.2:
        return "I'm sorry, I don't have information on that topic. Please ask about CodeAlpha tasks or submissions."

    return faqs[faq_questions[best_idx]]



if user_input := st.chat_input("Type your question here..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    
    bot_response = get_best_match(user_input)
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_response}
    )
    with st.chat_message("assistant"):
        st.markdown(bot_response)

#credit to shan obadiah.s.A collge student
