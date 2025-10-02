# -*- coding: utf-8 -*-
"""
Chatbot Escolar — Colégio Cleoracy
Cole e execute no Google Colab (ou em qualquer Python).
"""

from __future__ import annotations
import re
import unicodedata
from typing import Dict, Optional, Tuple, List

# tenta sklearn; se não houver, usa fallback mais conservador com difflib
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN = True
except Exception:
    import difflib
    SKLEARN = False

# ---------- normalização / tokenização ----------
def normalizar(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def simple_stem(tokens: List[str]) -> List[str]:
    out = []
    for tok in tokens:
        if len(tok) > 3 and tok.endswith("s"):
            out.append(tok[:-1])
        else:
            out.append(tok)
    return out

def tokenize_for_vector(text: str) -> List[str]:
    t = normalizar(text)
    toks = [w for w in t.split() if len(w) > 1]
    return simple_stem(toks)

# ---------- base de conhecimento (unificada) ----------
FAQ_RAW: Dict[str, str] = {
    "qual é o horário de atendimento?": "O horário de atendimento da secretaria é de segunda a sexta, das 13h às 23h.",
    "qual é o valor da mensalidade?": "Não possui mensalidade",
    "como faço a matrícula?": "A matrícula pode ser feita presencialmente na secretaria ou pelo site da escola.",
    "quais documentos preciso para matrícula?": "É necessário RG, CPF, comprovante de residência e histórico escolar.",
    "a escola oferece alimentação?": "Sim, temos merenda escolar inclusa para todos os alunos.",
    "a escola oferece transporte?": "Sim, a escola possui transporte escolar com rotas definidas.",
    "quais cursos a escola oferece?": "Oferecemos ensino médio e o Curso Técnico em Desenvolvimento de Sistemas.",
    "qual é o telefone da escola?": "Nosso telefone é (44) 99879-8508",
    "qual é o endereço da escola?": "Estamos localizados na Rua Francisco Barroso, 280, centro de Douradina-PR",
    "a escola tem atividades extracurriculares?": "Sim, oferecemos esportes, música, teatro e clube de ciências.",
    "quais são os documentos necessários para matrícula?": "Para matrícula: RG, CPF (quando aplicável), certidão de nascimento, comprovante de residência, histórico escolar (se transferência), carteira de vacinação atualizada e 1 foto 3x4.",
    "qual é o valor da mensalidade e formas de pagamento?": "O valor varia por série; a secretaria fornece a tabela atualizada. Aceitamos boleto, cartão, débito em conta e PIX, conforme contrato.",
    "há taxa de matrícula ou material didático?": "Há taxa de matrícula anual. Material didático pode ser adquirido na escola ou em livrarias parceiras, conforme série.",
    "qual é o horário de entrada e saída?": "Horários são organizados por série: Infantil e Anos Iniciais têm horários específicos; consulte a secretaria para horários por série.",
    "a escola oferece período integral?": "Oferecemos opção de contraturno/período estendido conforme vagas, com atividades pedagógicas e acompanhamento de tarefas.",
    "há transporte escolar credenciado?": "Indicamos transportadores autorizados; o contrato é firmado diretamente entre família e prestador.",
    "como funciona a alimentação na escola?": "O lanche é gratuito e segue cardápio criado e acompanhado por nutricionistas do Estado.",
    "quais atividades extracurriculares são oferecidas?": "Oferecemos inglês, música, judô, futebol, balé, robótica, entre outras atividades, conforme semestre e turmas.",
    "qual é a formação dos professores?": "Nossa equipe docente é formada por profissionais graduados; vários possuem pós-graduação e formação continuada.",
    "como a escola lida com bullying e conflitos?": "Temos programa de convivência, acompanhamento psicopedagógico, mediação de conflitos e comunicação com famílias em casos relevantes.",
    "quando começam e terminam as aulas?": "O calendário letivo normalmente inicia em fevereiro e termina em dezembro, com recesso em julho.",
    "vai haver aula na véspera de feriado?": "Em geral mantemos aulas até a véspera; alterações são sempre comunicadas oficialmente.",
    "qual o período de férias escolares?": "Férias de julho (cerca de 15 dias) e recesso de fim de ano entre dezembro e janeiro; consulte o calendário anual.",
    "a escola oferece colônia de férias?": "Sim — geralmente em julho e janeiro, com vagas limitadas e programação recreativa/pedagógica.",
    "como funcionam reposições em feriados prolongados?": "Reposições são agendadas em sábados letivos ou por calendário suplementar, conforme necessidade.",
    "há programação em datas comemorativas?": "Sim: realizamos apresentações e eventos (Dia das Mães, Páscoa, Natal etc.) com convite às famílias.",
    "como a escola age em tragédias ou emergências?": "Ativamos protocolos de segurança, comunicamos pais e autoridades e adaptamos atividades conforme necessidade.",
    "como são feitas as comunicações urgentes?": "Via aplicativo oficial da escola, e-mail, WhatsApp institucional e publicações no site.",
    "qual a orientação em caso de doença contagiosa?": "Solicitamos atestado médico; seguimos orientações da Vigilância Sanitária e políticas internas para evitar contágio.",
    "a escola devolve mensalidade em caso de suspensão prolongada?": "Questões financeiras são tratadas conforme contrato e legislação; oferecemos atividades on-line e reposições quando aplicável.",
    "quem é a diretora da escola?": "A diretora é a Francys Paula Otilio Mota Espolador.",
    "quem é a pedagoga responsável?": "A pedagoga responsável é Maria Sônia.",
    "qual o endereço do colégio?": "Rua Francisco Barroso, 280, centro de Douradina-PR",
    "qual o período de aulas?": "O período de aulas é Noturno.",
    "como é o lanche da escola?": "O lanche é gratuito e segue cardápio criado e acompanhado por nutricionistas do Estado.",
    "qual o grau de violência do colégio?": "Nenhuma ocorrência registrada nos últimos três anos.",
    "quantos alunos tem no colégio?": "O colégio possui aproximadamente 500 alunos matriculados.",
    "como faço a matrícula do meu filho?": "A matrícula pode ser feita presencialmente na secretaria ou online pelo site da escola.",
    "tem uniforme escolar obrigatório?": "Sim, o uso do uniforme é obrigatório durante as aulas e atividades escolares.",
    "onde posso comprar o uniforme escolar?": "O uniforme pode ser adquirido na loja parceira indicada pela escola.",
    "qual o telefone da secretaria?": "O telefone da secretaria é (11) 1234-5678.",
    "como funciona a cantina da escola?": "A cantina funciona no intervalo das aulas e oferece lanches saudáveis.",
    "a escola possui ensino integral?": "Sim, temos opção de ensino integral até o 9º ano.",
    "qual é o calendário escolar deste ano?": "O calendário está disponível no site da escola ou na secretaria.",
    "como justifico a falta do meu filho?": "A justificativa deve ser enviada por e-mail ou entregue na secretaria.",
    "como acompanho as notas do meu filho?": "As notas podem ser acompanhadas pelo portal online da escola.",
    "qual é a filosofia pedagógica da escola?": "Seguimos uma proposta construtivista, incentivando o aprendizado ativo do aluno.",
    "há reuniões de pais e mestres?": "Sim, as reuniões são trimestrais. As datas são informadas com antecedência.",
    "a escola oferece reforço escolar?": "Sim, temos aulas de reforço no contraturno para alunos que precisarem.",
    "como faço para falar com um professor?": "O contato pode ser feito por meio da agenda escolar ou pelo e-mail institucional do professor.",
    "a escola possui biblioteca?": "Sim, temos uma biblioteca com amplo acervo de livros e espaço de estudo.",
    "quantas aulas tem por dia?": "São 5 aulas por dia.",
    "quais são as materias que tem no curso?": "O curso segue a grade curricular do Curso Técnico em Desenvolvimento de Sistemas do Paraná.",
    "qual e o metodo de ensino": "É centrado na formação integral do estudante, com foco na abordagem histórico-cultural, buscando o desenvolvimento cognitivo, social, físico, cultural e emocional dos alunos.",
    "Há acessibilidade para alunos com deficiência": "O colégio possui recursos de acessibilidade física, como rampas e banheiros adaptados. Para outras necessidades específicas (visuais, auditivas ou de aprendizagem), orientamos que a família entre em contato com a secretaria para avaliarmos caso a caso.",
    "Como é a segurança dentro e ao redor da escola": "O colégio conta com dois monitores de pátio responsáveis pela supervisão dos alunos, sistema de câmeras de segurança em funcionamento e apoio externo da Polícia Militar, que realiza rondas periódicas na região.",
    "A escola participa de olimpíadas ou feiras de conhecimento": "Sim, o colégio participa de olimpíadas em geral (Robótica, Matemática, IA, Programação)",
    "Quantos dias de aulas por semana?": "As aulas são de segunda à sexta das 18h:50 às 23h",
    "possui água potável no colégio?": "Sim. O colégio possui água potável disponível em bebedouros com filtro, acessíveis a todos os alunos.",
    "As salas são climatizadas?": "Sim, todas as salas possui Ar Condicionado.",
    "O colégio possui gerador elétrico ?": "Não possui",
    "O colégio possui plano de evacuação?": "Sim. O colégio conta com um plano de evacuação e realiza treinamentos periódicos ao longo do ano para orientar alunos e funcionários sobre como agir em situações de emergência."
}

FAQ_KEYS = list(FAQ_RAW.keys())
FAQ_NORMALIZED = [normalizar(k) for k in FAQ_KEYS]

# ---------- vetor TF-IDF (quando disponível) ----------
VECTORIZER = None
TFIDF_MATRIX = None
if SKLEARN:
    VECTORIZER = TfidfVectorizer(tokenizer=tokenize_for_vector, lowercase=False, ngram_range=(1,2))
    TFIDF_MATRIX = VECTORIZER.fit_transform(FAQ_NORMALIZED)

# ---------- padrões fortes (regex) para capturar perguntas curtas sem ambiguidade ----------
KEYWORD_PATTERNS = {
    r"\bquantos alunos\b": "quantos alunos tem no colégio?",
    r"\baluno(s)?\b": "quantos alunos tem no colégio?",
    r"\bquantas aulas\b": "quantas aulas tem por dia?",
    r"\baulas?\b": "qual o período de aulas?",
    r"\bmensalidade\b": "qual é o valor da mensalidade?",
    r"\bendereçob|onde fica\b": "qual é o endereço da escola?",
    r"\btelefone\b": "qual é o telefone da escola?",
    r"\bmatr(í|i)cula|matricula\b": "como faço a matrícula?",
    r"\bdocumento(s)?\b": "quais são os documentos necessários para matrícula?",
    r"\blanche|lanche\b": "como funciona a alimentação na escola?",
    r"\bdiretora|diretor\b": "quem é a diretora da escola?",
    r"\bférias\b": "qual o período de férias escolares?",
    r"\baulas por semana\b": "Quantos dias de aulas por semana?",
    r"\bmerenda\b": "como é o lanche da escola?",

}


# ---------- parâmetros de decisão (seguro) ----------
SIMILARITY_THRESHOLD = 0.50     # cosseno mínimo para aceitar automaticamente
TOKEN_OVERLAP_THRESHOLD = 0.25 # sobreposição mínima de tokens
HIGH_SCORE = 0.80              # se >= isso, aceita mesmo se overlap baixo
FOLLOWUP_TOKEN_LIMIT = 3       # apenas inputs curtos (<=) serão considerados follow-ups
CONTEXT_SCORE_DELTA = 0.15     # aumento mínimo p/ aceitar contexto

# ---------- utilitários de matching ----------
def token_overlap(a_tokens: List[str], b_tokens: List[str]) -> float:
    sa = set(a_tokens)
    sb = set(b_tokens)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)

def is_followup_candidate(user_norm: str) -> bool:
    toks = [t for t in user_norm.split() if len(t) > 0]
    return len(toks) <= FOLLOWUP_TOKEN_LIMIT

def find_best_match_simple(user_text: str) -> Tuple[Optional[str], float, float]:
    """Retorna (faq_key, similarity_score, token_overlap) sem contexto."""
    user_norm = normalizar(user_text)
    user_tokens = simple_stem([t for t in user_norm.split() if len(t) > 1])
    if not user_tokens:
        return None, 0.0, 0.0

    # 1) checar padrões fortes (regex) primeiro
    for pat, faq_key in KEYWORD_PATTERNS.items():
        if re.search(pat, user_norm):
            faq_tok = simple_stem(normalizar(faq_key).split())
            overlap = token_overlap(user_tokens, faq_tok)
            return faq_key, 1.0, overlap

    # 2) TF-IDF + cosine (se disponível)
    if SKLEARN and VECTORIZER is not None:
        q_vec = VECTORIZER.transform([user_norm])
        sims = cosine_similarity(q_vec, TFIDF_MATRIX)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        faq_tok = simple_stem(FAQ_NORMALIZED[best_idx].split())
        overlap = token_overlap(user_tokens, faq_tok)
        return FAQ_KEYS[best_idx], best_score, overlap

    # 3) Fallback com difflib (muito conservador)
    import difflib as _d
    matches = _d.get_close_matches(user_norm, FAQ_NORMALIZED, n=1, cutoff=0.75)
    if matches:
        idx = FAQ_NORMALIZED.index(matches[0])
        faq_tok = simple_stem(FAQ_NORMALIZED[idx].split())
        overlap = token_overlap(user_tokens, faq_tok)
        return FAQ_KEYS[idx], 0.85, overlap

    return None, 0.0, 0.0

def find_best_match_with_context(user_text: str, last_key: Optional[str]) -> Tuple[Optional[str], float, float]:
    """Tenta combinar contexto (last_key) + user_text quando apropriado."""
    # primeiro sem contexto
    faq_key, score, overlap = find_best_match_simple(user_text)
    if not last_key:
        return faq_key, score, overlap

    user_norm = normalizar(user_text)
    combined = normalizar(last_key) + " " + user_norm
    # calcular com TF-IDF se possível
    if SKLEARN and VECTORIZER is not None:
        q_vec = VECTORIZER.transform([combined])
        sims = cosine_similarity(q_vec, TFIDF_MATRIX)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        faq_tok = simple_stem(FAQ_NORMALIZED[best_idx].split())
        combined_tokens = simple_stem([t for t in combined.split() if len(t) > 1])
        overlap2 = token_overlap(combined_tokens, faq_tok)
        return faq_key, score, overlap if (score >= best_score) else (FAQ_KEYS[best_idx], best_score, overlap2)
    else:
        # fallback: sem contexto efetivo
        return faq_key, score, overlap

# ---------- loop de conversação ----------
def chatbot_loop():
    last_matched: Optional[str] = None
    print("🤖 Chatbot Escolar — modo conversa (digite 'sair' para encerrar)\n")
    while True:
        try:
            user = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando. Até mais!")
            break
        if not user:
            continue
        if user.lower() == "sair":
            print("Chatbot: Até logo! 👋")
            break

        # 1) melhor match sem contexto
        faq_key, score, overlap = find_best_match_simple(user)

        # 2) decisão segura: responder apenas se confiável
        reliable = False
        if faq_key:
            if (score >= HIGH_SCORE):
                reliable = True
            elif (score >= SIMILARITY_THRESHOLD and overlap >= TOKEN_OVERLAP_THRESHOLD):
                reliable = True

        # 3) se não confiável, tentar contexto somente se input for follow-up curto
        used_context = False
        if not reliable and last_matched and is_followup_candidate(normalizar(user)):
            # tenta com contexto; aceita só se melhora concretamente
            faq_ctx, score_ctx, overlap_ctx = find_best_match_with_context(user, last_matched)
            # só aceitar contexto se houver ganho significativo
            if faq_ctx and (score_ctx - score >= CONTEXT_SCORE_DELTA) and (overlap_ctx >= TOKEN_OVERLAP_THRESHOLD):
                faq_key, score, overlap = faq_ctx, score_ctx, overlap_ctx
                reliable = True
                used_context = True

        # 4) responder ou pedir reformulação
        if reliable and faq_key:
            resp = FAQ_RAW.get(faq_key, "Desculpe — informação não encontrada.")
            print("Chatbot:", resp)
            last_matched = faq_key  # atualiza contexto
            continue

        # 5) Não confiável -> não responder errado; mostrar sugestões
        print("Chatbot: Desculpe — não tenho certeza da resposta. Pode reformular?")
        # compor sugestões (top 3)
        suggestions = []
        if SKLEARN and VECTORIZER is not None:
            qv = VECTORIZER.transform([normalizar(user)])
            sims = cosine_similarity(qv, TFIDF_MATRIX)[0]
            top_idx = sims.argsort()[::-1][:3]
            for idx in top_idx:
                suggestions.append((FAQ_KEYS[int(idx)], float(sims[int(idx)])))
        else:
            # fallback com difflib (menos preciso)
            import difflib as _d
            matches = _d.get_close_matches(normalizar(user), FAQ_NORMALIZED, n=3, cutoff=0.0)
            for m in matches:
                i = FAQ_NORMALIZED.index(m)
                suggestions.append((FAQ_KEYS[i], 0.0))

        if suggestions:
            print("Sugestões (perguntas próximas):")
            for s, sc in suggestions:
                print(" -", s)
        # não atualizamos last_matched para evitar contaminação por respostas incorretas

# ---------- executar ----------
if __name__ == "__main__":
    chatbot_loop()
