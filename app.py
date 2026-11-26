import os
import uuid
import base64
import tempfile

import streamlit as st

from groq import Groq
from gtts import gTTS

from streamlit_mic_recorder import mic_recorder

from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

# --------------------------------------------------
# PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Voice AI Assistant",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Voice AI Assistant")

# --------------------------------------------------
# API KEYS
# --------------------------------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=GROQ_API_KEY
)

llm = ChatGroq(
   groq_api_key=GROQ_API_KEY,
   model="openai/gpt-oss-120b",
   temperature=0.3
)

# --------------------------------------------------
# SESSION MEMORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        SystemMessage(
            content="""
            You are a helpful voice assistant.

            Speak naturally.

            Avoid markdown.

            Avoid bullet points.

            Avoid tables.

            Give answers optimized for text-to-speech.
            """
        )
    ]

# --------------------------------------------------
# AUTOPLAY AUDIO
# --------------------------------------------------

def autoplay_audio(file_path):

    with open(file_path, "rb") as f:
        data = f.read()

    b64 = base64.b64encode(data).decode()

    audio_html = f"""
        <audio autoplay>
            <source
                src="data:audio/mp3;base64,{b64}"
                type="audio/mp3">
        </audio>
    """

    st.markdown(
        audio_html,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# CLEAN TEXT FOR SPEECH
# --------------------------------------------------

def clean_for_speech(text):

    text = text.replace("*", "")

    text = text.replace("-", " ")

    text = text.replace("#", "")

    return text

# --------------------------------------------------
# TEXT TO SPEECH
# --------------------------------------------------

def text_to_speech(text):

    cleaned_text = clean_for_speech(text)

    output_file = f"{uuid.uuid4()}.mp3"

    tts = gTTS(
        text=cleaned_text,
        lang="en"
    )

    tts.save(output_file)

    return output_file

# --------------------------------------------------
# SPEECH TO TEXT USING GROQ
# --------------------------------------------------

def transcribe_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:

        with open(temp_path, "rb") as audio_file:

            transcript = (
                client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3"
                )
            )

        return transcript.text

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for msg in st.session_state.messages:

    if isinstance(
        msg,
        HumanMessage
    ):

        st.chat_message(
            "user"
        ).write(
            msg.content
        )

    elif isinstance(
        msg,
        AIMessage
    ):

        st.chat_message(
            "assistant"
        ).write(
            msg.content
        )

# --------------------------------------------------
# VOICE INPUT
# --------------------------------------------------

st.subheader("🎤 Voice Input")

audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    key="voice_recorder"
)

if audio:

    try:

        if "bytes" not in audio:

            st.error(
                "Audio bytes not received."
            )

        elif len(audio["bytes"]) == 0:

            st.error(
                "Empty audio file."
            )

        else:

            with st.spinner(
                "Transcribing..."
            ):

                user_text = (
                    transcribe_audio(
                        audio["bytes"]
                    )
                )

            st.chat_message(
                "user"
            ).write(
                user_text
            )

            st.session_state.messages.append(
                HumanMessage(
                    content=user_text
                )
            )

            with st.spinner(
                "Thinking..."
            ):

                response = llm.invoke(
                    st.session_state.messages
                )

            answer = response.content

            st.chat_message(
                "assistant"
            ).write(
                answer
            )

            st.session_state.messages.append(
                AIMessage(
                    content=answer
                )
            )

            with st.spinner(
                "Generating voice..."
            ):

                mp3_file = (
                    text_to_speech(
                        answer
                    )
                )

            autoplay_audio(
                mp3_file
            )

    except Exception as e:

        st.exception(e)

# --------------------------------------------------
# TEXT CHAT INPUT
# --------------------------------------------------

user_text = st.chat_input(
    "Ask anything..."
)

if user_text:

    st.chat_message(
        "user"
    ).write(
        user_text
    )

    st.session_state.messages.append(
        HumanMessage(
            content=user_text
        )
    )

    response = llm.invoke(
        st.session_state.messages
    )

    answer = response.content

    st.chat_message(
        "assistant"
    ).write(
        answer
    )

    st.session_state.messages.append(
        AIMessage(
            content=answer
        )
    )

    mp3_file = text_to_speech(
        answer
    )

    autoplay_audio(
        mp3_file
    )
