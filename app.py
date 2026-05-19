"""Streamlit app for human validation of LLM-as-a-judge cultural labels.

Annotators independently re-label a sample of MMLU-Pro-X items as
culturally specific or universal. Their labels are later compared against
the Gemini-2.5-Flash judge to measure human-vs-AI agreement (see README).

Two logical pages, switched via st.session_state:
  * landing / login page
  * annotation page  (-> thank-you page once the batch is complete)

Storage backend: a Google Sheet with three worksheets (batches,
assignments, annotations) accessed through gspread.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BATCH_SIZE = 50
NUM_OPTIONS = 10
OPTION_LETTERS = [chr(ord("A") + i) for i in range(NUM_OPTIONS)]   # A..J

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
POOL_CSV = "cultural_eval_pool.csv"

WS_BATCHES = "batches"
WS_ASSIGNMENTS = "assignments"
WS_ANNOTATIONS = "annotations"

HEADER_BATCHES = ["batch_id", "item_id", "position_in_batch", "is_overlap"]
HEADER_ASSIGNMENTS = ["annotator_id", "batch_id", "assigned_at"]
HEADER_ANNOTATIONS = [
    "annotator_id", "batch_id", "item_id", "is_culturally_specific",
    "specificity_type", "region", "confidence", "explanation",
    "position_in_batch", "submitted_at",
]

SPECIFICITY_TYPES = [
    "Region/Country", "Religion/Philosophy", "Language-internal",
    "Named-entity", "Social-convention", "Other",
]
CONFIDENCE_LEVELS = ["High", "Medium", "Low"]

INSTRUCTIONS_MD = """\
# Task

Your task is to decide whether a multiple-choice question requires
knowledge tied to a specific culture, region, religion, or language —
or whether the answer is universal across cultures.

# When to label "Culturally Specific"

- The question references a specific country's laws, history, geography,
  or institutions (e.g., U.S. tax law, the Indian Constitution).
- The question references a specific religious or philosophical
  tradition's beliefs.
- Answering correctly requires understanding a specific language's
  grammar, idiom, or wordplay.
- The question references region-specific named entities (specific
  authors, cuisines, athletes, brands, local conventions).
- The question assumes social conventions or value systems tied to a
  particular culture.

# When to label "Universal"

- Pure math, hard sciences (physics, chemistry, biology, generic
  programming).
- Concepts that hold the same across cultures.

# Specificity types (if culturally specific)

- **Region/Country**: tied to a specific country's laws, history, or
  institutions.
- **Religion/Philosophy**: tied to a religious or philosophical tradition.
- **Language-internal**: requires understanding of a specific language's
  grammar or wordplay.
- **Named-entity**: references a culturally-embedded named person, work,
  brand, etc.
- **Social-convention**: tied to social customs or value systems.
- **Other**: doesn't fit the above but is still culturally specific.

# Hard cases

- A hard-science question that *mentions* a country by name is still
  UNIVERSAL — the science doesn't change.
- A programming question is UNIVERSAL even when it mentions a specific
  language like Python or Java.
- If genuinely undecided, use "Low" confidence and pick your best guess.

# Examples

- "What is the derivative of x²?" → Universal.
- "Which amendment established the direct election of U.S. senators?"
  → Culturally specific (Region/Country, USA).
- "According to the Buddhist concept of anatta, what is the nature of
  the self?" → Culturally specific (Religion/Philosophy).
- "Which of Hemingway's novels is set during the Spanish Civil War?"
  → Culturally specific (Named-entity).
- "What does the term 'inflation' mean in economics?" → Universal.
"""

st.set_page_config(
    page_title="Cultural Annotation",
    page_icon="🌍",
    layout="wide",  # MCQs with 10 options need horizontal room
    initial_sidebar_state="expanded",  # instructions visible by default
)


# --------------------------------------------------------------------------
# Cached resources / data
# --------------------------------------------------------------------------
@st.cache_resource
def get_sheet():
    """Authorized gspread Spreadsheet handle (cached for the app's life)."""
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["GSPREAD_SHEET_ID"])


@st.cache_data
def load_pool() -> pd.DataFrame:
    """Load the joined MCQ pool, indexed by item_id (read as strings)."""
    df = pd.read_csv(POOL_CSV, dtype=str).fillna("")
    return df.set_index("item_id")


@st.cache_resource(show_spinner=False)
def _ensure_worksheet(name: str, header_key: tuple):
    """Worksheet handle with row 1 guaranteed to hold exactly the header.

    Cached: the handle is reused and the header check runs only once per
    app process, so reads/writes don't pay for it on every rerun.
    """
    header = list(header_key)
    sh = get_sheet()
    try:
        ws = sh.worksheet(name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=2000, cols=max(len(header), 10))
    first_row = ws.row_values(1)
    if not first_row:                                  # empty sheet
        ws.update(values=[header], range_name="A1")    # header deterministically at A1
    elif [c.strip() for c in first_row][:len(header)] != header:
        ws.insert_row(header, index=1)                 # header missing -> prepend it
    return ws


def get_worksheet(name: str, header: list[str]):
    """Cached worksheet handle (see _ensure_worksheet)."""
    return _ensure_worksheet(name, tuple(header))


def read_records(name: str, header: list[str]) -> list[dict]:
    """Fresh read of a worksheet's data rows as a list of dict rows.

    Rows are parsed positionally against `header` (our own column-order
    constant) rather than the sheet's header row, so the read cannot fail
    on stray content in row 1. Blank rows are skipped.
    """
    ws = get_worksheet(name, header)
    out: list[dict] = []
    for row in ws.get_all_values()[1:]:                # drop the header row
        if not any(str(c).strip() for c in row):
            continue
        padded = list(row) + [""] * len(header)        # pad trimmed trailing cells
        out.append(dict(zip(header, padded)))
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Batch / assignment / progress logic
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_batches_records() -> list[dict]:
    """Cached read of the static 'batches' worksheet.

    'batches' is written once by prepare_batches.py and never changes
    during collection, so caching it is safe and removes a Sheets round-
    trip from every rerun. (assignments/annotations are NOT cached.)
    """
    return read_records(WS_BATCHES, HEADER_BATCHES)


def all_batch_ids() -> list[int]:
    """Distinct batch_ids defined in the 'batches' worksheet."""
    return sorted({
        int(r["batch_id"]) for r in load_batches_records()
        if str(r["batch_id"]).strip()
    })


def batch_items(batch_id: int) -> list[tuple[int, str]]:
    """Items of a batch as a list of (position_in_batch, item_id), ordered."""
    items = [
        (int(r["position_in_batch"]), str(r["item_id"]))
        for r in load_batches_records()
        if str(r["batch_id"]).strip() and int(r["batch_id"]) == batch_id
    ]
    return sorted(items)


def get_or_assign_batch(annotator_id: str):
    """Return (batch_id, status).

    status is one of: 'returning', 'new', 'all_assigned'.
    A returning annotator keeps their existing batch; a new one is given
    the smallest unassigned batch_id.
    """
    records = read_records(WS_ASSIGNMENTS, HEADER_ASSIGNMENTS)

    for r in records:
        if str(r["annotator_id"]) == annotator_id:
            return int(r["batch_id"]), "returning"

    assigned = {
        int(r["batch_id"]) for r in records if str(r["batch_id"]).strip()
    }
    free = [b for b in all_batch_ids() if b not in assigned]
    if not free:
        return None, "all_assigned"

    batch_id = min(free)
    get_worksheet(WS_ASSIGNMENTS, HEADER_ASSIGNMENTS).append_row(
        [annotator_id, batch_id, now_iso()]
    )
    return batch_id, "new"


def load_batch_progress(annotator_id: str, batch_id: int) -> set[str]:
    """item_ids this annotator has already submitted for this batch.

    Read from the 'annotations' worksheet once when the annotation page is
    first shown (to find the resume point). Afterwards progress is tracked
    in st.session_state, so button clicks don't trigger a Sheets read.
    """
    rows = read_records(WS_ANNOTATIONS, HEADER_ANNOTATIONS)
    return {
        str(r["item_id"])
        for r in rows
        if str(r["annotator_id"]) == annotator_id
        and str(r["batch_id"]) == str(batch_id)
    }


def record_annotation(row: dict) -> None:
    """Append one annotation row in HEADER_ANNOTATIONS order."""
    ws = get_worksheet(WS_ANNOTATIONS, HEADER_ANNOTATIONS)
    ws.append_row([row[col] for col in HEADER_ANNOTATIONS])


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def render_landing() -> None:
    st.title("Cultural Annotation Task 🌍")
    st.markdown(
        "Help us check an AI judge. You will see multiple-choice questions "
        "and decide, for each one, whether answering it correctly requires "
        "knowledge tied to a **specific culture, region, religion, or "
        "language**, or whether the question is **universal** across "
        "cultures. Each annotator labels a batch of "
        f"**{BATCH_SIZE} questions**."
    )
    st.info(
        "Your annotations will be compared to an AI judge's labels on the "
        "same items. We are measuring how often the AI agrees with humans."
    )

    with st.expander(
        "📖 Annotation instructions — please read before you start",
        expanded=True,
    ):
        st.markdown(INSTRUCTIONS_MD)

    st.warning(
        "Use a unique username. If you've annotated before, use the **same "
        "username** to resume where you left off."
    )

    username = st.text_input("Username", key="username_input").strip()
    agreed = st.checkbox(
        "I have read the instructions and understood the task",
        key="agreed_checkbox",
    )
    if st.button("Start annotating", type="primary", disabled=not agreed):
        if not username:
            st.error("Please enter a username.")
            return
        batch_id, status = get_or_assign_batch(username)
        if status == "all_assigned":
            st.error(
                "All batches are currently in use. Please contact the "
                "researcher to assign you a batch."
            )
            return
        st.session_state.annotator_id = username
        st.session_state.batch_id = batch_id
        st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"**Annotator:** {st.session_state.annotator_id}")
        st.markdown(f"**Batch:** {st.session_state.batch_id}")
        if st.button("Sign out"):
            for key in ("annotator_id", "batch_id", "done_items"):
                st.session_state.pop(key, None)
            st.rerun()
        st.divider()
        st.info("📋 **Instructions** — reference; keep this open while you "
                "annotate.")
        st.markdown(INSTRUCTIONS_MD)


def render_thank_you() -> None:
    st.title("Thank you! 🎉")
    st.success(
        f"You've completed all {BATCH_SIZE} items in your batch. Your labels "
        "will help us validate the AI judge used in our paper. You can close "
        "this tab."
    )


def render_question(item: pd.Series, item_id: str) -> None:
    """Display the subject and the full-size question stem.

    The 10 options A-J are hidden behind a "Show options (A-J)" toggle,
    collapsed by default: most items can be decided from the stem alone,
    and the annotator opens the options only when the stem is ambiguous.
    The toggle is keyed on item_id, so it resets to collapsed for every
    new item.

    Question/option text is HTML-escaped and rendered in styled <div>s so
    that special characters in the MCQ text are shown verbatim and the
    correct answer is highlighted in green without relying on bracket-
    fragile markdown.
    """
    st.markdown(f"**Subject:** {item['subject_category']}")
    st.markdown(
        f"<div style='font-family:Georgia,serif;font-size:1.35rem;"
        f"line-height:1.5;margin:0.5rem 0 1rem 0;'>"
        f"{html.escape(str(item['question_text']))}</div>",
        unsafe_allow_html=True,
    )
    show_options = st.toggle("Show options (A–J)", key=f"options_{item_id}")
    if not show_options:
        return

    correct = str(item["correct_answer_letter"]).strip().upper()
    for letter in OPTION_LETTERS:
        text = str(item.get(f"option_{letter.lower()}", ""))
        if text == "":
            continue
        safe = html.escape(text)
        if letter == correct:
            st.markdown(
                f"<div style='color:#1a7f37;font-weight:700;margin:2px 0;'>"
                f"{letter}. {safe} &nbsp;&#10003; (correct answer)</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='margin:2px 0;'><b>{letter}.</b> {safe}</div>",
                unsafe_allow_html=True,
            )


def render_annotation() -> None:
    render_sidebar()

    annotator_id = st.session_state.annotator_id
    batch_id = st.session_state.batch_id

    items = batch_items(batch_id)
    if not items:
        st.error(
            f"Batch {batch_id} has no items. The 'batches' worksheet may not "
            "have been populated — contact the researcher."
        )
        return

    # Read the 'annotations' sheet only once per session to find the resume
    # point; thereafter progress is tracked in session_state, so button
    # clicks (incl. the Yes/No toggle) rerun with no Sheets round-trip.
    if "done_items" not in st.session_state:
        st.session_state.done_items = load_batch_progress(annotator_id, batch_id)
    done = st.session_state.done_items

    remaining = [(pos, iid) for pos, iid in items if iid not in done]

    if not remaining:
        render_thank_you()
        return

    position, item_id = remaining[0]
    n_done = len(items) - len(remaining)

    st.title(f"Annotator: {annotator_id}")
    st.progress(
        n_done / len(items),
        text=f"Item {n_done + 1} of {len(items)} "
             f"({round(100 * n_done / len(items))}%)",
    )

    pool = load_pool()
    if item_id not in pool.index:
        st.error(
            f"Item '{item_id}' is missing from {POOL_CSV}. "
            "Re-run prepare_batches.py — contact the researcher."
        )
        return
    item = pool.loc[item_id]
    render_question(item, item_id)

    st.divider()

    # The Yes/No radio lives OUTSIDE st.form so the specificity_type /
    # region fields can be shown conditionally (form widgets don't trigger
    # a rerun until submit). Widget keys are suffixed with item_id, so when
    # the next item loads every input resets to its default (all blank).
    is_cs = st.radio(
        "Is this item culturally specific?",
        ["Yes", "No"],
        index=None,
        horizontal=True,
        key=f"is_cs_{item_id}",
    )

    with st.form(f"form_{item_id}", border=False):
        if is_cs == "Yes":
            specificity_type = st.selectbox(
                "Specificity type",
                SPECIFICITY_TYPES,
                index=None,                       # start blank, no default
                placeholder="Choose a specificity type…",
                key=f"spec_{item_id}",
            )
            region = st.text_input(
                "Which region or country is this item most related to? "
                "(optional)",
                key=f"region_{item_id}",
            )
        else:
            specificity_type = None
            region = ""
            if is_cs is None:
                st.caption("Select Yes above to choose a specificity type.")

        confidence = st.radio(
            "Confidence",
            CONFIDENCE_LEVELS,
            index=None,                           # no default; must be chosen
            horizontal=True,
            key=f"conf_{item_id}",
        )
        explanation = st.text_area(
            "Explanation (optional) — why did you label it this way?",
            key=f"explanation_{item_id}",
        )
        submitted = st.form_submit_button("Submit annotation", type="primary")

    if submitted:
        errors = []
        if is_cs is None:
            errors.append("Please answer: is this item culturally specific?")
        if is_cs == "Yes" and specificity_type is None:
            errors.append("Please choose a specificity type.")
        if confidence is None:
            errors.append("Please choose your confidence level.")
        if errors:
            for msg in errors:
                st.error(msg)
            return
        record_annotation({
            "annotator_id": annotator_id,
            "batch_id": batch_id,
            "item_id": item_id,
            "is_culturally_specific": is_cs,
            "specificity_type": specificity_type or "",
            "region": region.strip(),
            "confidence": confidence,
            "explanation": explanation.strip(),
            "position_in_batch": position,
            "submitted_at": now_iso(),
        })
        st.session_state.done_items.add(item_id)
        st.rerun()


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def main() -> None:
    if "annotator_id" in st.session_state:
        render_annotation()
    else:
        render_landing()


if __name__ == "__main__":
    main()
