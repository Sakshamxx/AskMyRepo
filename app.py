"""Streamlit front end for AskMyRepo: ask a question, see each claim with its file and line.
Run `streamlit run app.py` from this folder after building the index (see README.md)."""

import os
import time
from pathlib import Path

import streamlit as st

from askmyrepo import agent_tools, answer, keyword_index, vector_index
from askmyrepo.answer import ContractBroken
from askmyrepo.database import connect_read_only
from askmyrepo.llm import GeminiClient, ModelError, cost_usd
from askmyrepo.search import SETUPS, Searcher

st.set_page_config(page_title="AskMyRepo", page_icon="◦", layout="centered")

MODEL = os.environ.get("ASKMYREPO_MODEL", "gemini-3.5-flash-lite")
DATABASE = os.environ.get("ASKMYREPO_DATABASE", "../shoplite/instance/shoplite.db")
NAMESPACE = "syntax"

EXAMPLES = [
    "How much does shipping cost, and when is it free?",
    "How many orders are in each status?",
    "How does checkout apply coupons?",
    "How are customer passwords stored?",
    "Which active product has the least sellable stock, and does the low-stock report list it?",
    "Where is inv_reservation_ledger written?",
]

# name: (accent, soft background of the accent, subtle ink for chips)
PALETTES = {
    "Indigo": ("#4f46e5", "#eef2ff", "#3730a3"),
    "Emerald": ("#059669", "#ecfdf5", "#065f46"),
    "Amber": ("#d97706", "#fffbeb", "#92400e"),
    "Rose": ("#e11d48", "#fff1f2", "#9f1239"),
    "Slate": ("#334155", "#f1f5f9", "#1e293b"),
}


# Lets the app run without exporting keys by hand: reads .env if present.
def load_dotenv(path: str = ".env") -> None:
    """Read KEY=value lines from .env into the environment, if the shell didn't already."""
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


# All the page styling in one place: one accent colour on a plain white page.
def style(accent: str, soft: str, ink: str) -> None:
    """A quiet page: lots of white space, one accent colour, hairline borders."""
    st.markdown(
        f"""
<style>
  #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
  .block-container {{ padding-top: 2.2rem; max-width: 760px; }}
  html, body, [class*="css"] {{ font-family: Inter, -apple-system, "Segoe UI", sans-serif; }}
  h1.title {{ font-weight: 600; letter-spacing: -0.02em; margin: 0; font-size: 1.9rem; }}
  h1.title span {{ color: {accent}; }}
  p.sub {{ color: #6b7280; margin: 0.15rem 0 1.4rem 0; font-size: 0.95rem; }}
  .claim {{ border-left: 3px solid {accent}; padding: 0.15rem 0 0.15rem 0.9rem; margin: 0.9rem 0; }}
  .claim .text {{ font-size: 1rem; line-height: 1.55; }}
  .chip {{ display: inline-block; background: {soft}; color: {ink}; border-radius: 999px;
          padding: 0.12rem 0.6rem; font-size: 0.76rem; margin: 0.35rem 0.35rem 0 0;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace; word-break: break-all; }}
  .refusal {{ background: {soft}; color: {ink}; border-radius: 10px; padding: 0.8rem 1rem; }}
  .meta {{ color: #9ca3af; font-size: 0.78rem; margin-top: 0.8rem; }}
  .stButton > button {{ border-radius: 999px; border: 1px solid #e5e7eb; background: transparent;
                       font-size: 0.82rem; padding: 0.15rem 0.8rem; }}
  .stButton > button:hover {{ border-color: {accent}; color: {accent}; }}
  [data-testid="stSidebar"] {{ border-right: 1px solid #f0f0f0; }}
  [data-testid="stChatMessage"] {{ background: transparent; }}
  a {{ color: {accent}; }}
</style>
""",
        unsafe_allow_html=True,
    )


# Built once per server (cached) because loading the models is the slow part.
@st.cache_resource(show_spinner="Loading the search models (first time only)…")
def get_searcher() -> Searcher:
    """Pinecone, the keyword index, and (on first use) the local models."""
    index = vector_index.open_index(vector_index.connect())
    return Searcher(index, keyword_index.KeywordIndex(NAMESPACE), NAMESPACE)


# One shared Gemini client, created from GEMINI_API_KEY.
@st.cache_resource
def get_client() -> GeminiClient:
    """One Gemini client for the session."""
    return GeminiClient(os.environ["GEMINI_API_KEY"], MODEL)


# Opens ShopLite's SQLite file read-only so the agent can never change data.
@st.cache_resource
def get_database():
    """ShopLite's database, read-only."""
    return connect_read_only(Path(DATABASE))


# Translates the sidebar's repository choice (and 'newest only') into search filters.
def filters_for(version_choice: str, repo_choice: str) -> dict:
    """The sidebar choices in the filter format the search uses."""
    filters: dict = {}
    if repo_choice != "Both repositories":
        filters["repo"] = repo_choice
    if version_choice == "Newest only":
        filters["latest"] = True
    elif version_choice != "All versions":
        filters["version"] = version_choice
    return filters


# The only place the UI calls the agent; returns plain data the page can draw.
def ask(question: str, mode: str, setup: str, filters: dict) -> dict:
    """Answer one question; return the claims as plain data, plus steps, usage and cost."""
    client, searcher = get_client(), get_searcher()
    steps: list[str] = []
    started = time.perf_counter()
    if mode == "Agent":
        response = agent_tools.ask_agent(client, searcher, get_database(), question, filters,
                                         log=steps.append)
        result, usage, log = response["answer"], response["usage"], response["source_log"]
        describe = log.describe
        sources = response["sources"]
    else:
        hits = searcher.search(question, setup, filters, 5)
        result, usage = answer.answer_question(client, question, hits)
        sources = [{"location": h.location, "version": h.version, "text": h.text} for h in hits]

        # Pipeline mode: the file and lines of passage number `number`.
        def describe(number: int) -> str:
            """Where passage number `number` came from (file and lines)."""
            hit = hits[number - 1]
            return f"{answer.link(hit)}  (lines {hit.start_line}-{hit.end_line})"

    claims = ([{"statement": c.statement, "source": c.passage, "where": describe(c.passage),
                "text": sources[c.passage - 1].get("text", "")} for c in result.claims]
              if result.answered else [])
    return {"answered": result.answered, "refusal": result.refusal, "claims": claims,
            "steps": steps, "tokens_in": usage.input_tokens, "tokens_out": usage.output_tokens,
            "cost": cost_usd(MODEL, usage), "seconds": time.perf_counter() - started,
            "mode": mode, "shown": len(sources)}


# Draws one reply: claims with source chips, a refusal box, or an error.
def render(reply: dict) -> None:
    """One answer: claim by claim with its source chip, a refusal if there is no answer."""
    if reply.get("error"):
        st.error(reply["error"])
        return
    if not reply["answered"]:
        st.markdown(f'<div class="refusal">Not in the code or data I searched: '
                    f'{reply["refusal"]}</div>', unsafe_allow_html=True)
    for claim in reply["claims"]:
        where = claim["where"].replace("<", "&lt;")
        st.markdown(f'<div class="claim"><div class="text">{claim["statement"]}</div>'
                    f'<span class="chip">[{claim["source"]}] {where}</span></div>',
                    unsafe_allow_html=True)
    st.markdown(f'<div class="meta">${reply["cost"]:.4f} · {reply["seconds"]:.1f} s</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    repo_choice = st.selectbox("Repository", ["Both repositories", "shoplite",
                                              "shoplite-warehouse"])
    st.markdown("---")
    st.caption("Try one")
    for example in EXAMPLES:
        if st.button(example, key=example, use_container_width=True):
            st.session_state["queued"] = example
    st.markdown("---")
    total = sum(m["reply"].get("cost", 0) for m in st.session_state.get("history", [])
                if m["role"] == "assistant")
    st.caption(f"This session: ${total:.4f}")

style(*PALETTES["Emerald"])
st.markdown('<h1 class="title">Ask<span>My</span>Repo</h1>'
            '<p class="sub">Questions about ShopLite’s code and data, with the file and line '
            '(or the SQL) under every claim.</p>', unsafe_allow_html=True)

missing = [k for k in ("GEMINI_API_KEY", "PINECONE_API_KEY") if not os.environ.get(k)]
if missing:
    st.error(f"Missing in .env: {', '.join(missing)}")
    st.stop()

history = st.session_state.setdefault("history", [])
for message in history:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["text"])
        else:
            render(message["reply"])

question = st.chat_input("Ask about the code or the database…") or st.session_state.pop(
    "queued", None)
if question:
    history.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("The agent is working…"):
            try:
                reply = ask(question, "Agent", SETUPS[-1], filters_for("Newest only", repo_choice))
            except ContractBroken as exc:
                reply = {"error": f"The answer broke the contract: {exc}"}
            except ModelError as exc:
                reply = {"error": f"The model call failed ({exc}). Wait a minute and try again."}
            except Exception as exc:  # show anything else instead of a stack trace
                reply = {"error": f"{type(exc).__name__}: {exc}"}
        render(reply)
    history.append({"role": "assistant", "reply": reply})
