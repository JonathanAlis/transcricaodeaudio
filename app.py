import streamlit as st
from faster_whisper import WhisperModel
import time

# Função para transcrever o áudio e exibir as informações
def transcrever_audio(uploaded_file, model_size):
    # Carregar o modelo com base na escolha do usuário
    model = WhisperModel(model_size, device="cpu")  # Use "cuda" se tiver GPU
    
    # Processar o áudio e obter transcrição com timestamps para palavras
    segments, info = model.transcribe(uploaded_file, word_timestamps=True)
    
    # Exibir as informações
    st.subheader("Informações sobre o áudio:")
    st.write(f"Língua detectada: {info.language} (probabilidade: {info.language_probability:.2f})")
    
    st.subheader("Transcrição Completa:")
    full_text = ""
    
    # Criar a barra de progresso
    progress_bar = st.progress(0)  # Iniciar a barra de progresso
    
    # Transformar o generator em lista para poder calcular o progresso
    segments = list(segments)  
    num_segments = len(segments)   # Número de segmentos para calcular o progresso

    for idx, segment in enumerate(segments):
        st.write(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + " "
        
        # Atualizar a barra de progresso
        progress_bar.progress((idx + 1) / num_segments)  # Atualiza a barra com base no número de segmentos
    
    st.text_area("Texto Final", full_text, height=200)
    
    st.subheader("Probabilidades de Tokens:")
    for segment in segments:
        st.write(f"Segmento: [{segment.start:.2f}s - {segment.end:.2f}s]")
        if segment.words:  # Verifique se há palavras para iterar
            for word in segment.words:
                # Verificar se a probabilidade é menor que 80%
                if word.probability < 0.8:
                    # Exibir a palavra com probabilidade baixa em vermelho
                    st.markdown(f"<span style='color:red;'>Palavra: '{word.word}' | Probabilidade: {word.probability:.4f}</span>", unsafe_allow_html=True)
                else:
                    # Se a probabilidade for alta, exibir normalmente
                    st.write(f"Palavra: '{word.word}' | Probabilidade: {word.probability:.4f}")
        else:
            st.write("Nenhuma palavra transcrita para este segmento.")

# Cabeçalho da aplicação
st.title("Transcrição de Áudio")

# Menu dropdown para escolher o modelo
model_size = st.selectbox(
    "Escolha o tamanho do modelo Whisper:",
    ["tiny", "small", "base", "large"],
    index=1  # "small" é o segundo item, então índice 1

)

# Subir o arquivo de áudio
uploaded_file = st.file_uploader("Escolha um arquivo de áudio (WAV, MP3, etc.)", type=["wav", "mp3"])

# Se o arquivo foi carregado, realizar a transcrição
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")  # Exibir o áudio para ouvir
    transcrever_audio(uploaded_file, model_size)  # Chamar a função de transcrição
