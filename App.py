import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Números que Falam — Brasil", page_icon="📊", layout="wide")

UF_REGIAO = {
    "RO": "Norte", "AC": "Norte", "AM": "Norte", "RR": "Norte", "PA": "Norte", "AP": "Norte", "TO": "Norte",
    "MA": "Nordeste", "PI": "Nordeste", "CE": "Nordeste", "RN": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "AL": "Nordeste", "SE": "Nordeste", "BA": "Nordeste",
    "MG": "Sudeste", "ES": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "SC": "Sul", "RS": "Sul",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste", "GO": "Centro-Oeste", "DF": "Centro-Oeste",
}
RACAS = ["Branca", "Preta", "Parda", "Amarela", "Indígena"]
SEXOS = ["Homem", "Mulher"]
COLUNAS = ["UF", "Regiao", "Ano", "Sexo", "Raca", "Valor"]


@st.cache_data
def gerar_dados_demo() -> pd.DataFrame:
    """Taxa de alfabetização (%) SINTÉTICA por UF, ano, sexo e raça — apenas demonstração."""
    rng = np.random.default_rng(42)
    base_regiao = {"Norte": 90, "Nordeste": 86, "Sudeste": 96, "Sul": 96.5, "Centro-Oeste": 94}
    ajuste_raca = {"Branca": 2.0, "Preta": -2.5, "Parda": -2.0, "Amarela": 2.5, "Indígena": -8.0}
    ajuste_sexo = {"Homem": -0.8, "Mulher": 0.8}
    linhas = []
    for uf, reg in UF_REGIAO.items():
        for ano in (2010, 2022):
            for sexo in SEXOS:
                for raca in RACAS:
                    v = (base_regiao[reg] + ajuste_raca[raca] + ajuste_sexo[sexo]
                         + (2.5 if ano == 2022 else 0) + rng.normal(0, 1.2))
                    linhas.append([uf, reg, ano, sexo, raca, round(min(v, 99.8), 2)])
    return pd.DataFrame(linhas, columns=COLUNAS)


def ler_arquivo(arquivo) -> pd.DataFrame:
    """Lê CSV (tenta ; e ,) ou XLSX enviado pelo usuário."""
    nome = arquivo.name.lower()
    if nome.endswith((".xlsx", ".xls")):
        return pd.read_excel(arquivo)
    bruto = arquivo.getvalue()
    for sep in (";", ","):
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(bruto), sep=sep, encoding=enc)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    raise ValueError("Não consegui ler o arquivo. Verifique separador e codificação.")


def padronizar(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    """Renomeia colunas, converte números (vírgula decimal, '-', '...', 'X') e remove lixo."""
    df = df.rename(columns={v: k for k, v in mapa.items() if v and v != "(nenhuma)"}).copy()
    if "Regiao" not in df.columns and "UF" in df.columns:
        df["Regiao"] = df["UF"].map(UF_REGIAO).fillna("Outra")
    if "Ano" not in df.columns:
        df["Ano"] = 0
    df["Valor"] = (df["Valor"].astype(str).str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False))
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    df = df.dropna(subset=["Valor"])
    for c in ("Sexo", "Raca"):
        df[c] = df[c].astype(str).str.strip().str.capitalize()
    return df[COLUNAS].reset_index(drop=True)

def descritivas(s: pd.Series) -> dict:
    """Média, mediana, moda, quartis, IQR, amplitude, variância, desvio, CV e limites de outlier."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    moda = s.round(0).mode()
    return {
        "n": int(s.size),
        "media": s.mean(),
        "mediana": q2,
        "moda": moda.iloc[0] if not moda.empty else np.nan,
        "q1": q1, "q2": q2, "q3": q3, "iqr": iqr,
        "minimo": s.min(), "maximo": s.max(), "amplitude": s.max() - s.min(),
        "variancia": s.var(ddof=1) if s.size > 1 else 0.0,
        "desvio": s.std(ddof=1) if s.size > 1 else 0.0,
        "cv": (s.std(ddof=1) / s.mean() * 100) if s.size > 1 and s.mean() else np.nan,
        "lim_inf": q1 - 1.5 * iqr, "lim_sup": q3 + 1.5 * iqr,
    }

def outliers_iqr(df: pd.DataFrame, col: str = "Valor") -> pd.DataFrame:
    """Regra 1,5 × IQR."""
    d = descritivas(df[col])
    if not d:
        return df.iloc[0:0]
    return df[(df[col] < d["lim_inf"]) | (df[col] > d["lim_sup"])]

def tabela_por_grupo(df: pd.DataFrame, grupo: str) -> pd.DataFrame:
    """Estatísticas descritivas para cada categoria de `grupo` (Sexo ou Raca)."""
    linhas = []
    for g, parte in df.groupby(grupo):
        d = descritivas(parte["Valor"])
        linhas.append({grupo: g, "N": d["n"], "Média": d["media"], "Mediana": d["mediana"],
                       "Moda": d["moda"], "Q1": d["q1"], "Q3": d["q3"], "IQR": d["iqr"],
                       "Amplitude": d["amplitude"], "Desvio padrão": d["desvio"], "CV (%)": d["cv"]})
    return pd.DataFrame(linhas).round(2)

def lacuna(df: pd.DataFrame, grupo: str, a: str, b: str):
    """Diferença das médias (a − b) em pontos, calculada por UF; devolve (média da lacuna, tabela)."""
    p = df[df[grupo].isin([a, b])].groupby(["UF", grupo])["Valor"].mean().unstack()
    if a not in p or b not in p:
        return np.nan, pd.DataFrame()
    p["Lacuna"] = p[a] - p[b]
    return p["Lacuna"].mean(), p.reset_index().sort_values("Lacuna", ascending=False)

def interpretar(d: dict, rotulo: str) -> str:
    """Texto automático de leitura estatística (média vs mediana, dispersão, outliers)."""
    if not d:
        return "Sem dados para os filtros escolhidos."
    dif = d["media"] - d["mediana"]
    if abs(dif) < 0.05 * abs(d["mediana"] or 1):
        forma = "média e mediana são próximas: distribuição aproximadamente simétrica"
    elif dif < 0:
        forma = "a média está abaixo da mediana: assimetria à esquerda — valores muito baixos puxam a média para baixo"
    else:
        forma = "a média está acima da mediana: assimetria à direita — valores muito altos puxam a média para cima"
    homog = "baixa" if (d["cv"] or 0) < 5 else "moderada" if (d["cv"] or 0) < 15 else "alta"
    return (f"**{rotulo}** — média {d['media']:.2f}, mediana {d['mediana']:.2f}, IQR {d['iqr']:.2f}, "
            f"desvio padrão {d['desvio']:.2f}. Leitura: {forma}. Variação {homog} entre os territórios "
            f"(CV = {d['cv']:.1f}%). Amplitude de {d['amplitude']:.2f} pontos entre o menor e o maior valor.")

PERSONAS = {
    "Secretária de Estado da Mulher (BA/PE/AM/CE)": {
        "pergunta": "Onde a desigualdade entre mulheres e homens é maior no meu estado?",
        "usar": ("Sexo", "Mulher", "Homem"),
    },
    "Governador do Sudeste (MG/SP/ES/RJ)": {
        "pergunta": "Que desigualdades raciais ainda persistem na minha região e como me comparo aos vizinhos?",
        "usar": ("Raca", "Branca", "Preta"),
    },
    "Consórcio de Governadores (MA/PA/AC...)": {
        "pergunta": "Quais UFs do consórcio estão abaixo da mediana nacional e para qual grupo?",
        "usar": ("Raca", "Branca", "Indígena"),
    },
    "Ministra do Trabalho / Sebrae": {
        "pergunta": "Que grupos (sexo x raça) precisam de política de inclusão prioritária?",
        "usar": ("Sexo", "Mulher", "Homem"),
    },
}

st.sidebar.title("📊 Números que Falam")
st.sidebar.caption("Indicador: sexo e raça/cor — IBGE")

arq = st.sidebar.file_uploader("Upload CSV/XLSX (SIDRA/IBGE)", type=["csv", "xlsx", "xls"])
nome_indicador = st.sidebar.text_input("Nome do indicador", "Taxa de alfabetização (%)")

if arq is None:
    dados = gerar_dados_demo()
    st.sidebar.warning("Usando dados SINTÉTICOS de demonstração.")
else:
    try:
        bruto = ler_arquivo(arq)
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()
    st.sidebar.markdown("**Mapeie as colunas do seu arquivo**")
    cols = ["(nenhuma)"] + list(bruto.columns)
    mapa = {}
    for alvo, obrig in [("UF", True), ("Sexo", True), ("Raca", True), ("Valor", True),
                        ("Ano", False), ("Regiao", False)]:
        padrao = next((c for c in bruto.columns if c.lower().startswith(alvo.lower()[:3])), None)
        mapa[alvo] = st.sidebar.selectbox(alvo + (" *" if obrig else ""), cols,
                                          index=cols.index(padrao) if padrao else 0, key=alvo)
    if any(mapa[k] == "(nenhuma)" for k in ("UF", "Sexo", "Raca", "Valor")):
        st.info("Selecione as colunas UF, Sexo, Raça e Valor na barra lateral.")
        st.stop()
    dados = padronizar(bruto, mapa)

anos = sorted(dados["Ano"].unique())
ano = st.sidebar.selectbox("Ano", anos, index=len(anos) - 1)
regioes = st.sidebar.multiselect("Região", sorted(dados["Regiao"].unique()),
                                 default=sorted(dados["Regiao"].unique()))
ufs_disp = sorted(dados[dados["Regiao"].isin(regioes)]["UF"].unique())
ufs = st.sidebar.multiselect("UF", ufs_disp, default=ufs_disp)
sexos = st.sidebar.multiselect("Sexo", sorted(dados["Sexo"].unique()), default=sorted(dados["Sexo"].unique()))
racas = st.sidebar.multiselect("Raça/cor", sorted(dados["Raca"].unique()), default=sorted(dados["Raca"].unique()))

df = dados[(dados["Ano"] == ano) & dados["UF"].isin(ufs) & dados["Sexo"].isin(sexos) & dados["Raca"].isin(racas)]

pagina = st.sidebar.radio("Navegação", ["Visão geral", "Por sexo", "Por raça/cor", "Sexo × Raça",
                                        "Outliers", "Personas (consultoria)"])

if df.empty:
    st.warning("Nenhum dado para os filtros escolhidos.")
    st.stop()

st.title(f"{nome_indicador} — {ano}")
d_geral = descritivas(df["Valor"])

if pagina == "Visão geral":
    c = st.columns(4)
    c[0].metric("Média", f"{d_geral['media']:.2f}")
    c[1].metric("Mediana", f"{d_geral['mediana']:.2f}")
    c[2].metric("IQR", f"{d_geral['iqr']:.2f}")
    c[3].metric("Desvio padrão", f"{d_geral['desvio']:.2f}")
    c = st.columns(4)
    c[0].metric("Moda", f"{d_geral['moda']:.2f}")
    c[1].metric("Amplitude", f"{d_geral['amplitude']:.2f}")
    c[2].metric("Q1 / Q3", f"{d_geral['q1']:.1f} / {d_geral['q3']:.1f}")
    c[3].metric("Nº de observações", d_geral["n"])
    st.plotly_chart(px.histogram(df, x="Valor", nbins=20, title="Histograma geral",
                                 labels={"Valor": nome_indicador}), use_container_width=True)
    st.plotly_chart(px.box(df, y="Valor", points="all", hover_data=["UF", "Sexo", "Raca"],
                           title="Boxplot geral"), use_container_width=True)
    st.markdown(interpretar(d_geral, "Visão geral"))

elif pagina == "Por sexo":
    st.dataframe(tabela_por_grupo(df, "Sexo"), use_container_width=True)
    st.plotly_chart(px.box(df, x="Sexo", y="Valor", color="Sexo", points="all",
                           hover_data=["UF", "Raca"], title="Distribuição por sexo"),
                    use_container_width=True)
    med = df.groupby(["UF", "Sexo"])["Valor"].mean().reset_index()
    st.plotly_chart(px.bar(med, x="UF", y="Valor", color="Sexo", barmode="group",
                           title="Média por UF e sexo"), use_container_width=True)
    if {"Mulher", "Homem"} <= set(df["Sexo"]):
        m, tab = lacuna(df, "Sexo", "Mulher", "Homem")
        st.metric("Lacuna média Mulher − Homem (pontos)", f"{m:.2f}")
    for s_, p_ in df.groupby("Sexo"):
        st.markdown(interpretar(descritivas(p_["Valor"]), s_))

elif pagina == "Por raça/cor":
    st.dataframe(tabela_por_grupo(df, "Raca"), use_container_width=True)
    st.plotly_chart(px.box(df, x="Raca", y="Valor", color="Raca", points="all",
                           hover_data=["UF", "Sexo"], title="Distribuição por raça/cor"),
                    use_container_width=True)
    rk = df.groupby("Raca")["Valor"].mean().sort_values().reset_index()
    st.plotly_chart(px.bar(rk, x="Valor", y="Raca", orientation="h", title="Ranking de médias por raça/cor"),
                    use_container_width=True)
    if {"Branca", "Preta"} <= set(df["Raca"]):
        m, _ = lacuna(df, "Raca", "Branca", "Preta")
        st.metric("Lacuna média Branca − Preta (pontos)", f"{m:.2f}")
    for r_, p_ in df.groupby("Raca"):
        st.markdown(interpretar(descritivas(p_["Valor"]), r_))

elif pagina == "Sexo × Raça":
    cruz = df.groupby(["Raca", "Sexo"])["Valor"].mean().unstack().round(2)
    st.subheader("Média do indicador por raça/cor e sexo")
    st.dataframe(cruz, use_container_width=True)
    st.plotly_chart(px.imshow(cruz, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                              title="Mapa de calor Sexo × Raça"), use_container_width=True)
    st.plotly_chart(px.box(df, x="Raca", y="Valor", color="Sexo", title="Boxplot Raça × Sexo"),
                    use_container_width=True)
    pior = cruz.stack().idxmin()
    melhor = cruz.stack().idxmax()
    st.info(f"Menor média: **{pior[1]} {pior[0]}** ({cruz.stack().min():.2f}). "
            f"Maior média: **{melhor[1]} {melhor[0]}** ({cruz.stack().max():.2f}). "
            f"Distância entre extremos: {cruz.stack().max() - cruz.stack().min():.2f} pontos.")

elif pagina == "Outliers":
    out = outliers_iqr(df)
    st.write(f"Limites pela regra 1,5×IQR: **{d_geral['lim_inf']:.2f}** a **{d_geral['lim_sup']:.2f}**")
    st.metric("Outliers encontrados", len(out))
    if len(out):
        st.dataframe(out.sort_values("Valor"), use_container_width=True)
        sem = df.drop(out.index)
        st.write(f"Média com outliers: **{d_geral['media']:.2f}** | sem outliers: "
                 f"**{sem['Valor'].mean():.2f}** | mediana (não muda muito): **{d_geral['mediana']:.2f}**")
        st.caption("A mediana é robusta a valores extremos; a média não.")
    else:
        st.success("Nenhum outlier pelos critérios atuais.")
    st.plotly_chart(px.strip(df, x="Raca", y="Valor", color="Sexo", hover_data=["UF"],
                             title="Pontos individuais (outliers aparecem afastados)"),
                    use_container_width=True)

else:
    escolha = st.selectbox("Cliente (persona)", list(PERSONAS))
    p = PERSONAS[escolha]
    st.subheader("Pergunta de negócio")
    st.write(p["pergunta"])
    grupo, a, b = p["usar"]
    if {a, b} <= set(df[grupo].unique()):
        m, tab = lacuna(df, grupo, a, b)
        st.metric(f"Lacuna média {a} − {b} (pontos)", f"{m:.2f}")
        st.write("**Onde a lacuna é maior (top 5 UFs):**")
        st.dataframe(tab.head(5).round(2), use_container_width=True)
        st.plotly_chart(px.bar(tab, x="UF", y="Lacuna", title=f"Lacuna {a} − {b} por UF"),
                        use_container_width=True)
    else:
        st.info(f"Inclua {a} e {b} nos filtros para ver a lacuna.")
    st.markdown(interpretar(d_geral, "Resumo para a decisão"))
    st.caption("Roteiro: contexto → número-chave → o que ele esconde (média × mediana, outliers) → recomendação.")
