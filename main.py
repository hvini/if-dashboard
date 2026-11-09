import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords", quiet=True)
nltk.download("rslp", quiet=True)

OUT = Path("plots")
OUT.mkdir(exist_ok=True)

# ---------- vocabulário ----------
INFORMAL = {"pra", "pro", "pros", "tá", "ta", "tava", "tô", "to", "q", "vc", "pq", "tb", "tbm", "msm", "dps",
            "né", "ne", "aí", "ai", "lá", "la", "ja", "so", "ate", "nao", "entao", "então", "tudo", "todo", "toda",
            "todos", "vez", "vezes", "coisa", "gente", "kkkk", "kkk"}
# palavras presentes em quase todo testemunho de igreja: não diferenciam nada
DOMINIO = {"deus", "senhor", "jesus", "cristo", "glória", "gloria", "amém", "amem", "aleluia", "igreja", "irmãos",
           "irmao", "testemunho", "testemunhar", "compartilhar", "agradecer", "agradeço", "obrigado", "obrigada",
           "quero", "queria", "gostaria", "fiel", "abençoe"}
# tempo e verbos genéricos
FILLER = {"hoje", "ano", "anos", "mês", "mes", "meses", "dia", "dias", "semana", "agora", "ainda", "aqui", "faz",
          "fiz", "deu", "consegui", "nunca", "sempre", "depois", "desde", "outro", "outra", "bem", "passado",
          "passada", "vida", "mim", "nada", "cada", "ia", "dar", "ter", "ser", "fazer", "vai", "vou"}
STOP = set(stopwords.words("portuguese")) | INFORMAL | DOMINIO | FILLER

# categorias pastorais (dicionário de palavras-chave; um testemunho pode ter várias)
CATEGORIAS = {
    "Saúde física": r"cirurg|exame|médic|medic|hospital|\buti\b|câncer|quimio|internad|diabet|pressão|\bavc\b|infart|"
                    r"fratur|engravid|grávida|prematur|dengue|covid|tireoide|hepatite|catarata|endometriose|convuls|"
                    r"pneumonia|biópsia|nódulo|stent|próstata|hérnia|leucemia|fisioterap|fêmur|oncolog|cardiolog|"
                    r"glicada|plaqueta|neurolog",
    "Saúde emocional": r"depress|ansiedade|pânico|panico|insônia|burnout|terapia|psicólog|psicolog|psiquiatr|trauma|"
                       r"luto|mágoa|esgotad|sozinh|vontade de viver|me cortava|ninho vazio|tdah|abuso|vingança|"
                       r"aconselhamento|dúvida|sem chão|perdi (?:minha|meu) (?:mãe|pai|marido|esposa|filh)|cabeça|tranquil|consolad",
    "Vícios": r"álcool|alcool|\bbebia\b|droga|cocaína|crack|cigarro|\bfum(?:ei|ava|ar)\b|\bjogo\b|\bbets?\b|"
              r"pornografia|vici|recaí|limpo desde|bulimia|recuperação|tráfico|bebida|sóbri",
    "Finanças": r"dinheiro|dívid|devendo|aluguel|despej|parcel|nome sujo|nome limpo|conta de luz|cesta|vaquinha|"
                r"vakinha|\bpix\b|herança|auxílio|financ|bolsa de|doou|sem condição|não tinha como pagar|negativad|\brenda\b|receber",
    "Trabalho e estudos": r"emprego|desempregad|demitid|contratad|promovid|concurso|\bvaga\b|currículo|entrevista|"
                          r"estágio|salário|carteira assinada|registrado|aposent|\bmei\b|licitação|faculdade|"
                          r"vestibular|\benem\b|mestrado|aprovad|aprendiz|client|oficina|aplicativo|contratos?\b",
    "Família": r"casamento|casad|brigad|divórcio|separ(?:ar|ados|ação)|reconcil|sem (?:se )?falar|briga|perdão|pazes|"
               r"relacionamento|abusiv|pai ausente|renovamos|reconheci|aceitar|fugido|cortou o contato|pai saiu|me chamou de vó|alzheimer",
    "Fé e serviço": r"batiz|convert|\bservi(?:r|ndo)\b|\bsirvo\b|ministério|missionári|\bfé\b|louvor|obreiro|berçário|bíblia|\borar\b",
}
# o que ajudou: indicador de quais frentes da igreja (e de fora) aparecem nos testemunhos
AJUDA = {
    "Oração": r"\bor(?:ei|ação|ações|aram|ou|ando|ar)\b",
    "Jejum / campanha": r"jejum|jejuei|jejuado|propósito|campanha|quartas? da virada",
    "Célula / grupos": r"célula|grupo de|discipul|\blíder|ministério de|retiro",
    "Pastor / aconselhamento": r"pastor|aconselhamento|sessão",
    "Ajuda material dos irmãos": r"cesta|vaquinha|vakinha|doou|doação|ação social|de graça|\bpix\b|casa de recuperação",
    "Cursos da igreja": r"curso de (?:casais|finanças)|curso.*igreja|igreja indicou",
    "Culto / Bíblia": r"\bculto|bíblia|palavra de deus|louvor",
    "Saúde / serviço público": r"médic|terapia|psicólog|psiquiatr|\bsus\b|\bcaps\b|posto de saúde|\binss\b|hospital|"
                               r"justiça|assistente social|senai|tratamento|samu|fisioterap",
}
CAT_ORIGINAL = {"Cura Física": "Saúde física", "Paz e Saúde Mental": "Saúde emocional", "Libertação": "Vícios",
                "Provisão Financeira": "Finanças", "Emprego e Carreira": "Trabalho e estudos",
                "Restauração Familiar": "Família"}

# ---------- estilo (paleta de referência da skill dataviz, modo claro) ----------
AZUL, LARANJA = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10,
})

df = pd.read_csv("testemunhos.csv", encoding="utf-8-sig", parse_dates=["data_registro"])
texto = df["testemunho_texto"].str.lower()
df["faixa"] = pd.cut(df["idade"], [0, 29, 44, 59, 200], labels=["até 29", "30–44", "45–59", "60+"])

# ---------- TF-IDF: stopwords + radical (RSLP) + bigramas ----------
stemmer = RSLPStemmer()
TOKEN = re.compile(r"(?u)\b[^\W\d_]{2,}\b")
exemplos = defaultdict(Counter)  # radical -> palavras originais, para exibir algo legível


def analisar(t):
    radicais = []
    for w in TOKEN.findall(t.lower()):
        if w in STOP or re.fullmatch(r"k+", w):
            continue
        r = stemmer.stem(w)
        exemplos[r][w] += 1
        radicais.append(r)
    return radicais + [f"{a} {b}" for a, b in zip(radicais, radicais[1:])]


vec = TfidfVectorizer(analyzer=analisar, min_df=2)  # min_df=2: termo precisa aparecer em 2+ testemunhos
X = vec.fit_transform(df["testemunho_texto"])
legivel = [" ".join(exemplos[r].most_common(1)[0][0] for r in t.split()) for t in vec.get_feature_names_out()]
tfidf = pd.DataFrame(X.toarray(), columns=legivel)

geral = tfidf.mean().groupby(level=0).sum().sort_values(ascending=False).rename("tfidf_medio")
geral.rename_axis("termo").to_csv("tfidf.csv")
print("Top 20 termos (TF-IDF médio no corpus):")
print(geral.head(20).round(4).to_string(), "\n")


def distintivos(mascara, n=8):
    """Termos com TF-IDF médio maior no grupo do que no resto."""
    diff = tfidf[mascara.values].mean() - tfidf[~mascara.values].mean()
    return diff.groupby(level=0).sum().nlargest(n)


# ---------- categorias e "o que ajudou" por dicionário ----------
def marcar(dicionario):
    return pd.DataFrame({k: texto.str.contains(p, regex=True) for k, p in dicionario.items()})


cats, ajudas = marcar(CATEGORIAS), marcar(AJUDA)
sem_cat = ~cats.any(axis=1)
print(f"Testemunhos sem categoria: {sem_cat.sum()} de {len(df)}")
for t in df.loc[sem_cat, "testemunho_texto"]:
    print("  -", t[:90])

# conferência (não usada na análise): a categoria do formulário aparece entre as detectadas?
alvo = df["categoria_inferida"].map(CAT_ORIGINAL)
ok = [c in cats.columns and cats.at[i, c] for i, c in alvo.dropna().items()]
print(f"\nConcordância com categoria_inferida: {sum(ok)}/{len(ok)} ({sum(ok) / len(ok):.0%})\n")


# ---------- gráficos ----------
def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(OUT / nome, dpi=150)
    plt.close(fig)


def barh_ax(ax, serie, fmt="{:.0f}", cor=AZUL):
    serie = serie.sort_values()
    ax.barh(serie.index, serie.values, color=cor, height=0.7)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(serie.values):
        ax.text(v, y, " " + fmt.format(v), va="center", color=INK2, fontsize=9)


def barh(serie, titulo, nome, fmt="{:.0f}", xlabel=""):
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(serie) + 1.2))
    barh_ax(ax, serie, fmt)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    salvar(fig, nome)


def heatmap(tab, titulo, nome, xlabel=""):
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * tab.shape[1] + 3), 0.55 * tab.shape[0] + 1.5))
    ax.imshow(tab.values, cmap=SEQ, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), tab.index)
    for (i, j), v in pd.DataFrame(tab.values).stack().items():
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="#ffffff" if v > tab.values.max() / 2 else INK)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    salvar(fig, nome)


def por_grupo(grupos, titulo, nome):
    """Small multiples: termos distintivos de cada grupo vs o restante."""
    fig, axes = plt.subplots(1, len(grupos), figsize=(4 * len(grupos), 3.8))
    for ax, (rotulo, mascara) in zip(axes, grupos.items()):
        barh_ax(ax, distintivos(mascara), fmt="{:.3f}")
        ax.set_title(f"{rotulo} (n={mascara.sum()})", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(titulo, x=0.01, ha="left", fontweight="bold")
    salvar(fig, nome)


# 1. termos mais relevantes no geral
barh(geral.head(20), "Termos mais relevantes (TF-IDF médio)", "01_tfidf_geral.png", "{:.3f}")

# 2. testemunhos por mês (inclui meses sem registro)
mes = df.set_index("data_registro").resample("MS").size()
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(mes.index.strftime("%Y-%m"), mes.values, color=AZUL, width=0.7)
ax.grid(axis="x", visible=False)
ax.axhline(mes.mean(), color=MUTED, lw=1, ls="--")
ax.text(len(mes) - 0.5, mes.mean(), f" média {mes.mean():.1f}", color=INK2, va="bottom", ha="right", fontsize=9)
ax.set_title("Testemunhos por mês")
ax.set_ylabel("testemunhos")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
salvar(fig, "02_testemunhos_por_mes.png")

# 3. categorias mais abordadas (um testemunho pode contar em mais de uma)
barh(cats.sum(), "Assuntos mais abordados (um testemunho pode ter vários)", "03_categorias.png", xlabel="testemunhos")

# 4–6. categorias por faixa etária, sexo e trimestre
heatmap(pd.concat({f: cats[df["faixa"] == f].sum() for f in df["faixa"].cat.categories}, axis=1),
        "Assunto por faixa etária", "04_categoria_faixa_etaria.png", "faixa etária")

sx = pd.concat({s: cats[df["sexo"] == s].sum() for s in ["F", "M"]}, axis=1).sort_values("F")
fig, ax = plt.subplots(figsize=(8, 4.5))
y = range(len(sx))
ax.barh([i + 0.2 for i in y], sx["F"], height=0.38, color=LARANJA, label="Feminino")
ax.barh([i - 0.2 for i in y], sx["M"], height=0.38, color=AZUL, label="Masculino")
ax.set_yticks(list(y), sx.index)
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Assunto por sexo")
ax.set_xlabel("testemunhos")
salvar(fig, "05_categoria_sexo.png")

tri = df["data_registro"].dt.to_period("Q").astype(str)
heatmap(pd.concat({q: cats[tri == q].sum() for q in sorted(tri.unique())}, axis=1),
        "Assunto por trimestre", "06_categoria_trimestre.png")

# 7. perfil etário
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["idade"], bins=range(10, 85, 5), color=AZUL, rwidth=0.9)
ax.grid(axis="x", visible=False)
ax.axvline(df["idade"].median(), color=MUTED, ls="--", lw=1)
ax.text(df["idade"].median(), ax.get_ylim()[1] * 0.95, f" mediana {df['idade'].median():.0f}", color=INK2, fontsize=9)
ax.set_title("Idade de quem testemunha")
ax.set_xlabel("idade")
ax.set_ylabel("testemunhos")
salvar(fig, "07_idade.png")

# 8. o que ajudou
barh(ajudas.sum(), "O que é citado como ajuda", "08_o_que_ajudou.png", xlabel="testemunhos que citam")
heatmap(pd.concat({c: ajudas[cats[c]].sum() for c in cats.columns}, axis=1),
        "O que ajudou, por assunto", "09_ajuda_por_assunto.png")

# 10–12. termos que distinguem cada grupo
por_grupo({"Feminino": df["sexo"] == "F", "Masculino": df["sexo"] == "M"},
          "Termos que distinguem cada sexo", "10_termos_sexo.png")
por_grupo({f: df["faixa"] == f for f in df["faixa"].cat.categories},
          "Termos que distinguem cada faixa etária", "11_termos_faixa.png")
corte = df["data_registro"].max() - pd.DateOffset(months=6)
por_grupo({"Antes": df["data_registro"] <= corte, "Últimos 6 meses": df["data_registro"] > corte},
          "Termos que distinguem o período recente", "12_termos_periodo.png")

# ---------- dados do dashboard (dashboard/dados.js) ----------
# só atributos e índices por testemunho, sem o texto: o navegador refaz as contas a cada filtro
termos_df = tfidf.T.groupby(level=0).sum().T
dados = {
    "assuntos": list(cats.columns),
    "ajudas": list(ajudas.columns),
    "termos": list(termos_df.columns),
    "testemunhos": [{
        "data": d.strftime("%Y-%m-%d"), "idade": int(i), "faixa": f, "sexo": s,
        "assuntos": [k for k, v in enumerate(c) if v],
        "ajudas": [k for k, v in enumerate(a) if v],
        "termos": [[k, round(float(v), 4)] for k, v in enumerate(t) if v > 0],
    } for d, i, f, s, c, a, t in zip(df["data_registro"], df["idade"], df["faixa"].astype(str), df["sexo"],
                                     cats.values, ajudas.values, termos_df.values)],
}
Path("dashboard").mkdir(exist_ok=True)
Path("dashboard/dados.js").write_text("const DADOS = " + json.dumps(dados, ensure_ascii=False) + ";\n", encoding="utf-8")

# self-check
assert len(dados["testemunhos"]) == len(df)
assert not (set(geral.index[:20]) & STOP), "stopword no ranking"
assert len(tfidf) == len(cats) == len(ajudas) == len(df)
print(f"Gráficos salvos em {OUT.resolve()}")
