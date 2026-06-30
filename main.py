import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(page_title="Detecção de Início - Sit-to-Stand", layout="wide")

# ============================================================
# FUNÇÕES DE PROCESSAMENTO
# ============================================================
@st.cache_data
def carregar_dados(arquivo):
    df = pd.read_csv(arquivo, sep=';')
    df.columns = ['tempo_ms', 'X', 'Y', 'Z']
    df['tempo_s'] = df['tempo_ms'] / 1000.0
    return df


def interpolar_100hz(df, fs_alvo=100):
    t_original = df['tempo_s'].values
    t_novo = np.arange(t_original[0], t_original[-1], 1.0/fs_alvo)
    df_interp = pd.DataFrame({'tempo_s': t_novo})
    for eixo in ['X', 'Y', 'Z']:
        f_interp = interp1d(t_original, df[eixo].values,
                            kind='linear', fill_value='extrapolate')
        df_interp[eixo] = f_interp(t_novo)
    return df_interp, fs_alvo


def aplicar_detrend(df, eixos=['X', 'Y', 'Z']):
    df_detrend = df.copy()
    for eixo in eixos:
        df_detrend[eixo] = signal.detrend(df[eixo].values, type='linear')
    return df_detrend


def filtrar_sinal(df, fs, fc=8.0, ordem=4, eixos=['X', 'Y', 'Z']):
    nyq = fs / 2.0
    b, a = signal.butter(ordem, fc/nyq, btype='low')
    df_filtrado = df.copy()
    for eixo in eixos:
        df_filtrado[eixo] = signal.filtfilt(b, a, df[eixo].values)
    return df_filtrado


# ============================================================
# DETECÇÃO DA BASELINE (MENOR VARIABILIDADE)
# ============================================================
def encontrar_baseline(sinal, fs, janela_ms=500):
    """
    Encontra a janela de 500 ms com menor variabilidade (variância).
    Retorna: índice inicial, índice final, média, desvio padrão.
    """
    janela_n = int(fs * janela_ms / 1000.0)  # 50 amostras para 100 Hz
    
    # Calcula variância em janelas deslizantes
    variancias = []
    for i in range(len(sinal) - janela_n):
        variancias.append(np.var(sinal[i:i+janela_n]))
    
    # Encontra a janela com menor variância
    idx_min = np.argmin(variancias)
    
    # Estatísticas da baseline
    baseline = sinal[idx_min:idx_min + janela_n]
    media = np.mean(baseline)
    dp = np.std(baseline)
    
    return idx_min, idx_min + janela_n, media, dp


# ============================================================
# DETECÇÃO DO INÍCIO DA ATIVIDADE
# ============================================================
def detectar_inicio_atividade(sinal, idx_fim_baseline, media, dp, fator=2.0):
    """
    A partir do final da baseline, encontra o primeiro ponto que cruza
    média ± fator*DP.
    """
    limite_sup = media + fator * dp
    limite_inf = media - fator * dp
    
    # Busca a partir do final da baseline
    for i in range(idx_fim_baseline, len(sinal)):
        if sinal[i] > limite_sup or sinal[i] < limite_inf:
            return i, limite_sup, limite_inf
    
    return None, limite_sup, limite_inf


# ============================================================
# INTERFACE
# ============================================================
st.title("🎯 Detecção do Início da Atividade - Sit-to-Stand")
st.markdown("---")

# Upload
arquivo = st.sidebar.file_uploader("📁 Arquivo do acelerômetro", type=['txt', 'csv'])

# Parâmetros
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parâmetros")
janela_ms = st.sidebar.slider("Janela baseline (ms)", 200, 1000, 500, 50)
fator_dp = st.sidebar.slider("Fator × DP", 1.0, 4.0, 2.0, 0.1)
fc = st.sidebar.slider("Frequência de corte (Hz)", 1.0, 20.0, 8.0, 0.5)

if arquivo is not None:
    with st.spinner("Processando..."):
        # Pipeline
        df = carregar_dados(arquivo)
        df_proc, fs = interpolar_100hz(df)
        df_proc = aplicar_detrend(df_proc)
        df_proc = filtrar_sinal(df_proc, fs, fc=fc)
        
        # Eixo Y (vertical)
        sinal = df_proc['Y'].values
        tempo = df_proc['tempo_s'].values
        
        # 1. Encontrar baseline
        idx_ini, idx_fim, media, dp = encontrar_baseline(sinal, fs, janela_ms)
        
        # 2. Detectar início da atividade
        idx_atividade, lim_sup, lim_inf = detectar_inicio_atividade(
            sinal, idx_fim, media, dp, fator=fator_dp
        )
        
        # ========================================================
        # RESULTADOS
        # ========================================================
        st.success("✅ Análise concluída!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📍 Baseline", f"{tempo[idx_ini]:.2f} - {tempo[idx_fim]:.2f} s")
        col2.metric("📊 Média", f"{media:.4f} g")
        col3.metric("📐 Desvio Padrão", f"{dp:.4f} g")
        col4.metric("🎯 Início Atividade", 
                    f"{tempo[idx_atividade]:.2f} s" if idx_atividade else "N/A")
        
        # ========================================================
        # GRÁFICO
        # ========================================================
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sinal completo
        ax.plot(tempo, sinal, 'b-', linewidth=0.8, label='Sinal (Y)')
        
        # Destaque da baseline
        ax.axvspan(tempo[idx_ini], tempo[idx_fim], 
                   color='green', alpha=0.2, label=f'Baseline ({janela_ms} ms)')
        
        # Linhas de média e limites
        ax.axhline(media, color='green', linestyle='--', linewidth=1.5, 
                   label=f'Média = {media:.4f} g')
        ax.axhline(lim_sup, color='red', linestyle='--', linewidth=1.2,
                   label=f'+{fator_dp} DP = {lim_sup:.4f} g')
        ax.axhline(lim_inf, color='red', linestyle='--', linewidth=1.2,
                   label=f'-{fator_dp} DP = {lim_inf:.4f} g')
        
        # Marca o início da atividade
        if idx_atividade is not None:
            ax.axvline(tempo[idx_atividade], color='magenta', 
                       linestyle=':', linewidth=2.5,
                       label=f'Início Atividade = {tempo[idx_atividade]:.2f} s')
            ax.plot(tempo[idx_atividade], sinal[idx_atividade], 
                    'mo', markersize=12, markeredgecolor='black')
        
        ax.set_xlabel('Tempo (s)', fontsize=12)
        ax.set_ylabel('Aceleração Y (g)', fontsize=12)
        ax.set_title('Detecção do Início da Atividade', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        
        # ========================================================
        # ZOOM NA REGIÃO DE INTERESSE
        # ========================================================
        st.subheader("🔍 Zoom na Região de Transição")
        
        # Zoom: 1s antes do início até 2s depois
        if idx_atividade is not None:
            t_zoom_ini = max(tempo[0], tempo[idx_atividade] - 1.0)
            t_zoom_fim = min(tempo[-1], tempo[idx_atividade] + 2.0)
            
            mask = (tempo >= t_zoom_ini) & (tempo <= t_zoom_fim)
            
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(tempo[mask], sinal[mask], 'b-', linewidth=1.2)
            ax2.axhline(media, color='green', linestyle='--')
            ax2.axhline(lim_sup, color='red', linestyle='--')
            ax2.axhline(lim_inf, color='red', linestyle='--')
            ax2.axvline(tempo[idx_atividade], color='magenta', 
                        linestyle=':', linewidth=2.5)
            ax2.plot(tempo[idx_atividade], sinal[idx_atividade], 
                     'mo', markersize=12, markeredgecolor='black')
            ax2.set_xlabel('Tempo (s)')
            ax2.set_ylabel('Aceleração Y (g)')
            ax2.set_title('Zoom na Transição Baseline → Atividade')
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
        
        # ========================================================
        # DOWNLOAD
        # ========================================================
        if idx_atividade is not None:
            resultado = pd.DataFrame({
                'Parametro': ['Baseline inicio (s)', 'Baseline fim (s)',
                              'Media baseline (g)', 'Desvio padrao (g)',
                              'Limite superior (g)', 'Limite inferior (g)',
                              'Inicio atividade (s)'],
                'Valor': [tempo[idx_ini], tempo[idx_fim], media, dp,
                          lim_sup, lim_inf, tempo[idx_atividade]]
            })
            csv = resultado.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Baixar resultados", csv, 
                               "resultados_baseline.csv", "text/csv")

else:
    st.info("👈 Faça upload de um arquivo para iniciar a análise.")
    
    st.markdown("""
    ### 📋 Algoritmo
    
    1. **Processamento do sinal**
       - Interpolação para 100 Hz
       - Detrend linear
       - Filtro passa-baixa Butterworth (8 Hz)
    
    2. **Busca da baseline**
       - Janela deslizante de 500 ms
       - Seleciona a janela com **menor variância**
       - Calcula média (μ) e desvio padrão (σ)
    
    3. **Detecção do início**
       - A partir do final da baseline
       - Primeiro ponto onde: `|sinal - μ| > 2σ`
       - Marca o início da atividade motora
    
    ### 🎯 Interpretação
    
    - 🟢 **Verde**: janela de baseline (repouso)
    - 🔴 **Vermelho tracejado**: limites μ ± 2σ
    - 🟣 **Magenta**: momento do início da atividade
    """)
