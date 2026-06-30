import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

# ============================================================
# 1. CARREGAMENTO DO ARQUIVO
# ============================================================
def carregar_dados(caminho_arquivo):
    """
    Carrega o arquivo CSV do acelerômetro.
    Assume separador ';' e primeira coluna como tempo em ms.
    """
    df = pd.read_csv(caminho_arquivo, sep=';')
    
    # Renomear colunas para facilitar o acesso
    df.columns = ['tempo_ms', 'X', 'Y', 'Z']
    
    # Converter tempo de ms para segundos
    df['tempo_s'] = df['tempo_ms'] / 1000.0
    
    return df

# ============================================================
# 2. INTERPOLAÇÃO PARA 100 Hz
# ============================================================
def interpolar_100hz(df):
    """
    Interpola os dados para uma frequência uniforme de 100 Hz.
    """
    fs_alvo = 100  # Hz
    t_original = df['tempo_s'].values
    t_novo = np.arange(t_original[0], t_original[-1], 1.0/fs_alvo)
    
    df_interp = pd.DataFrame({'tempo_s': t_novo})
    
    for eixo in ['X', 'Y', 'Z']:
        f_interp = interp1d(t_original, df[eixo].values, 
                           kind='linear', fill_value='extrapolate')
        df_interp[eixo] = f_interp(t_novo)
    
    return df_interp, fs_alvo

# ============================================================
# 3. DETREND (Remoção de Tendência Linear)
# ============================================================
def aplicar_detrend(df, eixos=['X', 'Y', 'Z']):
    """
    Remove a tendência linear de cada eixo.
    """
    df_detrend = df.copy()
    for eixo in eixos:
        df_detrend[eixo] = signal.detrend(df[eixo].values, type='linear')
    return df_detrend

# ============================================================
# 4. FILTRO PASSA-BAIXA 8 Hz (Butterworth)
# ============================================================
def filtrar_sinal(df, fs, fc=8.0, ordem=4, eixos=['X', 'Y', 'Z']):
    """
    Aplica filtro Butterworth passa-baixa de 8 Hz.
    """
    nyq = fs / 2.0
    b, a = signal.butter(ordem, fc/nyq, btype='low')
    
    df_filtrado = df.copy()
    for eixo in eixos:
        # filtfilt aplica o filtro em ambas as direções (fase zero)
        df_filtrado[eixo] = signal.filtfilt(b, a, df[eixo].values)
    
    return df_filtrado

# ============================================================
# 5. PIPELINE COMPLETO DE PROCESSAMENTO
# ============================================================
def processar_acelerometro(caminho_arquivo):
    """
    Pipeline completo: carrega -> interpola -> detrend -> filtra
    """
    print("📂 Carregando dados...")
    df = carregar_dados(caminho_arquivo)
    print(f"   - Amostras originais: {len(df)}")
    print(f"   - Duração: {df['tempo_s'].iloc[-1]:.2f} s")
    
    print("🔄 Interpolando para 100 Hz...")
    df_proc, fs = interpolar_100hz(df)
    print(f"   - Amostras após interpolação: {len(df_proc)}")
    
    print("📉 Aplicando detrend...")
    df_proc = aplicar_detrend(df_proc)
    
    print("🔧 Aplicando filtro passa-baixa 8 Hz...")
    df_proc = filtrar_sinal(df_proc, fs, fc=8.0, ordem=4)
    
    print("✅ Processamento concluído!")
    return df_proc, fs

# ============================================================
# 6. VISUALIZAÇÃO COM SELEÇÃO DE EIXO
# ============================================================
def visualizar_eixo(df, fs, eixo='Z'):
    """
    Plota o sinal processado de um eixo específico.
    
    Parâmetros:
    -----------
    df : DataFrame processado
    fs : frequência de amostragem (Hz)
    eixo : 'X', 'Y' ou 'Z'
    """
    if eixo not in ['X', 'Y', 'Z']:
        raise ValueError("Eixo deve ser 'X', 'Y' ou 'Z'")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['tempo_s'], df[eixo], linewidth=0.8, color='steelblue')
    ax.set_xlabel('Tempo (s)', fontsize=12)
    ax.set_ylabel(f'Aceleração - Eixo {eixo} (g)', fontsize=12)
    ax.set_title(f'Sinal Processado - Eixo {eixo} (100 Hz, detrend, LP 8 Hz)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(df['tempo_s'].iloc[0], df['tempo_s'].iloc[-1])
    plt.tight_layout()
    plt.show()

def visualizar_todos_eixos(df, fs):
    """
    Plota os três eixos em subplots separados.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    cores = {'X': 'red', 'Y': 'green', 'Z': 'blue'}
    
    for i, eixo in enumerate(['X', 'Y', 'Z']):
        axes[i].plot(df['tempo_s'], df[eixo], 
                    linewidth=0.8, color=cores[eixo])
        axes[i].set_ylabel(f'Eixo {eixo} (g)', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f'Eixo {eixo}', fontsize=11, fontweight='bold')
    
    axes[-1].set_xlabel('Tempo (s)', fontsize=12)
    fig.suptitle('Acelerômetro - Teste Sentar e Levantar (Processado)', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# ============================================================
# 7. EXECUÇÃO PRINCIPAL
# ============================================================
if __name__ == "__main__":
    # Caminho do arquivo
    arquivo = "Juliana SitStand Mobile .txt"
    
    # Processar dados
    df_proc, fs = processar_acelerometro(arquivo)
    
    # Escolher o eixo para visualização
    print("\n" + "="*50)
    print("SELEÇÃO DO EIXO")
    print("="*50)
    print("O eixo vertical (que registra a gravidade) depende de")
    print("como o celular foi posicionado. Geralmente:")
    print("  - Z: se o celular estava em pé (retrato)")
    print("  - Y: se o celular estava deitado (paisagem)")
    print("  - X: menos comum como eixo vertical")
    print("="*50)
    
    eixo_escolhido = input("\nDigite o eixo para visualizar (X, Y ou Z) [padrão: Z]: ").strip().upper()
    if eixo_escolhido == '':
        eixo_escolhido = 'Z'
    
    # Visualizar o eixo escolhido
    visualizar_eixo(df_proc, fs, eixo=eixo_escolhido)
    
    # Opcional: visualizar todos os eixos juntos
    ver_todos = input("\nDeseja visualizar todos os eixos? (s/n): ").strip().lower()
    if ver_todos == 's':
        visualizar_todos_eixos(df_proc, fs)
