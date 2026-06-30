import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Análise Sentar-Levantar (Eixo Y)",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================
@st.cache_data
def carregar_dados(arquivo):
    """Carrega o CSV do acelerômetro."""
    try:
        df = pd.read_csv(arquivo, sep=';')
        df.columns = ['tempo_ms', 'X', 'Y', 'Z']
        df['tempo_s'] = df['tempo_ms'] / 1000.0
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None


def interpolar_100hz(df, fs_alvo=100):
    """Interpola para frequência uniforme."""
    t_original = df['tempo_s'].values
    t_novo = np.arange(t_original[0], t_original[-1], 1.0/fs_alvo)
    
    df_interp = pd.DataFrame({'tempo_s': t_novo})
    for eixo in ['X', 'Y', 'Z']:
        f_interp = interp1d(t_original, df[eixo].values, 
                           kind='linear', fill_value='extrapolate')
        df_interp[eixo] = f_interp(t_novo)
    return df_interp, fs_alvo


def aplicar_detrend(df, eixos=['X', 'Y', 'Z']):
    """Remove tendência linear."""
    df_detrend = df.copy()
    for eixo in eixos:
        df_detrend[eixo] = signal.detrend(df[eixo].values, type='linear')
    return df_detrend


def filtrar_sinal(df, fs, fc=8.0, ordem=4, eixos=['X', 'Y', 'Z']):
    """Filtro Butterworth passa-baixa."""
    nyq = fs / 2.0
    b, a = signal.butter(ordem, fc/nyq, btype='low')
    df_filtrado = df.copy()
    for eixo in eixos:
        df_filtrado[eixo] = signal.filtfilt(b, a, df[eixo].values)
    return df_filtrado


def processar_acelerometro(df_original, fs_alvo=100, fc=8.0):
    """Pipeline completo."""
    df_interp, fs = interpolar_100hz(df_original, fs_alvo)
    df_detrend = aplicar_detrend(df_interp)
    df_filtrado = filtrar_sinal(df_detrend, fs, fc=fc)
    return df_filtrado, fs


# ============================================================
# INTERFACE STREAMLIT
# ============================================================
st.title("📊 Análise do Teste Sentar-Levantar — Eixo Y")
st.markdown("---")

# --- UPLOAD DO ARQUIVO ---
st.sidebar.header("⚙️ Configurações")
arquivo = st.sidebar.file_uploader(
    "📁 Carregue o arquivo do acelerômetro (.txt ou .csv)",
    type=['txt', 'csv'],
    help="Arquivo com colunas: DURACAO;ACC EIXO X;ACC EIXO Y;ACC EIXO Z"
)

# --- PARÂMETROS DO FILTRO ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Parâmetros")
fc = st.sidebar.slider("Frequência de corte (Hz)", 1.0, 20.0, 8.0, 0.5)
fs_alvo = st.sidebar.slider("Frequência de amostragem alvo (Hz)", 50, 200, 100, 10)

# ============================================================
# PROCESSAMENTO E VISUALIZAÇÃO
# ============================================================
if arquivo is not None:
    with st.spinner("⏳ Carregando e processando dados..."):
        df_original = carregar_dados(arquivo)
        
        if df_original is not None:
            st.success("✅ Arquivo carregado com sucesso!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📏 Amostras", f"{len(df_original):,}")
            col2.metric("⏱️ Duração", f"{df_original['tempo_s'].iloc[-1]:.2f} s")
            col3.metric("📡 Fs Original", 
                       f"{1000/np.mean(np.diff(df_original['tempo_ms'])):.1f} Hz")
            
            # Processamento
            df_proc, fs = processar_acelerometro(df_original, fs_alvo=fs_alvo, fc=fc)
            
            st.markdown("---")
            st.subheader("📈 Sinal Processado — Eixo Y (Vertical)")
            
            # Gráfico apenas do eixo Y
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_proc['tempo_s'], df_proc['Y'], 
                   linewidth=0.8, color='green')
            ax.set_xlabel('Tempo (s)', fontsize=12)
            ax.set_ylabel('Aceleração - Eixo Y (g)', fontsize=12)
            ax.set_title(f'Eixo Y (Fs={fs} Hz, LP={fc} Hz, detrend)', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(df_proc['tempo_s'].iloc[0], df_proc['tempo_s'].iloc[-1])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Estatísticas do eixo Y
            st.markdown("---")
            st.subheader("📋 Estatísticas do Eixo Y")
            stat1, stat2, stat3, stat4 = st.columns(4)
            stat1.metric("Média", f"{df_proc['Y'].mean():.4f} g")
            stat2.metric("Desvio Padrão", f"{df_proc['Y'].std():.4f} g")
            stat3.metric("Máximo", f"{df_proc['Y'].max():.4f} g")
            stat4.metric("Mínimo", f"{df_proc['Y'].min():.4f} g")
            
            # Download dos dados processados
            st.markdown("---")
            csv = df_proc.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Baixar dados processados (CSV)",
                data=csv,
                file_name="acelerometro_eixoY_processado.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Faça upload de um arquivo na barra lateral para começar a análise.")
    
    st.markdown("""
    ### 📖 Como usar
    
    1. **Carregue o arquivo** `.txt` ou `.csv` na barra lateral
    2. **Ajuste os parâmetros** do filtro se necessário
    3. **Analise** o sinal do eixo Y processado
    
    ### 🎯 Sobre o Eixo Y
    
    O eixo Y é o **eixo vertical** quando o celular está em modo **paisagem** (deitado). 
    Ele registra a aceleração da gravidade e os movimentos de sentar e levantar.
    
    ### ⚙️ Processamento aplicado
    
    - ✅ Interpolação para frequência uniforme
    - ✅ Remoção de tendência linear (detrend)
    - ✅ Filtro passa-baixa Butterworth (fase zero)
    """)
