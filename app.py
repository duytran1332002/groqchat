import streamlit as st
from typing import Generator
from groq import Groq
class AhaChatApp:
    def __init__(self):
        st.set_page_config(page_icon="💬", layout="wide", page_title="@ Aha AI Chat Exam...")
        self.client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        self.initialize_chat_history_and_model()
        self.models = {
            "llama3-70b-8192": {"name": "LLaMA3-70b", "tokens": 8192, "developer": "Meta"},
            "llama3-8b-8192": {"name": "LLaMA3-8b", "tokens": 8192, "developer": "Meta"},
        }
        self.create_layout()

    def icon(self, emoji: str):
        """Shows an emoji as a Notion-style page icon."""
        st.write(
            f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
            unsafe_allow_html=True,
        )

    def initialize_chat_history_and_model(self):
        if "messages" not in st.session_state:
            system_prompt = "You are a professional AI was implemented by Duy Tran for the Aha AI Exam. Please generate responses in English to all user inputs."
            st.session_state.messages = [{"role": "system", "content": system_prompt}]
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None

    def create_layout(self):
        self.icon("🧠")
        st.subheader("Get wow with my chat", divider="rainbow", anchor=False)

        col1, col2 = st.columns(2)

        with col1:
            self.model_option = st.selectbox(
                "Choose a model:",
                options=list(self.models.keys()),
                format_func=lambda x: self.models[x]["name"],
                index=0  # Default to llama3-70b-8192
            )

        # Detect model change and clear chat history if model has changed
        if st.session_state.selected_model != self.model_option:
            st.session_state.selected_model = self.model_option

        max_tokens_range = self.models[self.model_option]["tokens"]

        with col2:
            self.max_tokens = st.slider(
                "Max Tokens:",
                min_value=512,
                max_value=max_tokens_range,
                value=min(8192, max_tokens_range),
                step=512,
                help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
            )

        self.display_chat_history()
        self.handle_user_input()

    def display_chat_history(self):
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    def generate_chat_responses(self, chat_completion) -> Generator[str, None, None]:
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def handle_user_input(self):

        if prompt := st.chat_input("Enter your message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user", avatar='👨‍💻'):
                st.markdown(prompt)

            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model_option,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    max_tokens=self.max_tokens,
                    stream=True
                )

                with st.chat_message("assistant", avatar="🤖"):
                    chat_responses_generator = self.generate_chat_responses(chat_completion)
                    full_response = st.write_stream(chat_responses_generator)

                if isinstance(full_response, str):
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response})
                else:
                    combined_response = "\n".join(str(item) for item in full_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": combined_response})
            except Exception as e:
                st.error(e, icon="🚨")

if __name__ == "__main__":
    app = AhaChatApp()

    