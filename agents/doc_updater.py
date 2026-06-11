import sys
from pathlib import Path
from typing import TypedDict
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from agents.supabase_store import fetch_latest_doc, upload_doc, save_version, save_agent_log
from app.config import Config

MAX_ITERATIONS = 5


# AzureChatOpenAI reads Azure-specific params
llm = AzureChatOpenAI(
    azure_deployment=Config.CHAT_DEPLOYMENT,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    api_key=Config.AZURE_OPENAI_API_KEY,
    api_version=Config.AZURE_OPENAI_API_VERSION,
    temperature=0.0,
)


class DocState(TypedDict):
    code_diff: str
    current_docs: str
    file_contents: str
    current_version: int
    commit_sha: str
    trigger_source: str
    updated_docs: str
    review_feedback: str
    iteration: int
    logs: list[dict]

# The modifier agent updates the documentation based on the current docs, code, diff, and any reviewer feedback. The reviewer agent checks the updated docs against the actual code and provides feedback. The graph orchestrates these agents until the docs are approved or max iterations are reached.
def modifier_agent(state: DocState) -> DocState:
    iteration = state["iteration"] + 1
    log_msg = f"Iteration {iteration}: {'Addressing reviewer feedback and rewriting docs.' if state['review_feedback'] else 'Reading code diff and writing updated documentation.'}"
    print(f"[Modifier] {log_msg}")

    feedback_section = f"\n\nREVIEWER FEEDBACK TO ADDRESS:\n{state['review_feedback']}" if state["review_feedback"] else ""

    prompt = f"""You are a technical documentation writer.

    Generate markdown documentation that accurately reflects the provided codebase.

    CURRENT DOCUMENTATION:
    {state['current_docs']}

    SOURCE CODE:
    {state['file_contents']}

    GIT DIFF:
    {state['code_diff']}

    {feedback_section}

    Requirements:
    - Document only functionality present in the source code.
    - Address all reviewer feedback.
    - Include any undocumented modules found in the codebase.
    - Do not invent APIs, environment variables, setup steps, or features.

    Required sections:

    # Project Overview
    # Architecture
    # Application Modules
    # Agent Modules
    # API Endpoints
    # Setup Instructions
    # Environment Variables
    # Data Flow
    # Key Workflows

    For every module include:
    - Purpose
    - Key functions/classes
    - Responsibilities

    For agent modules also include:
    - Role in the documentation workflow

    Output only the complete markdown document.
    """
    
    updated = llm.invoke(prompt).content.strip()
    logs = state["logs"] + [{"agent": "modifier", "iteration": iteration, "message": log_msg}]
    return {**state, "updated_docs": updated, "iteration": iteration, "logs": logs}


def reviewer_agent(state: DocState) -> DocState:
    log_msg = f"Iteration {state['iteration']}: Reviewing documentation against source code..."

    print(f"[Reviewer] {log_msg}")

    prompt = f"""You are a senior engineer reviewing documentation accuracy.

    DOCUMENTATION:
    {state['updated_docs']}

    SOURCE CODE:
    {state['file_contents']}

    GIT DIFF:
    {state['code_diff']}

    Review only factual accuracy.

    Verify:
    1. API endpoints
    2. Module descriptions
    3. Setup instructions
    4. Environment variables
    5. Architecture
    6. Data flow
    7. Changes introduced by the git diff
    8. Documentation coverage for all app/ modules
    9. Documentation coverage for all agents/ modules

    Ignore:
    - Style
    - Wording
    - Formatting
    - Additional examples
    - Nice-to-have improvements

    Respond exactly with:

    APPROVED

    or

    REVISION NEEDED

    followed by a numbered list of factual issues.
    """

    feedback = llm.invoke(prompt).content.strip()
    print("\n=== Reviewer Feedback ===")
    print(feedback)
    print("========================\n")
    approved = feedback.upper().startswith("APPROVED")
    result_msg = f"Iteration {state['iteration']}: {'Approved.' if approved else 'Revision needed - ' + feedback[:150] + '...'}"
    logs = state["logs"] + [{"agent": "reviewer", "iteration": state["iteration"], "message": result_msg}]
    print(
    f"[Reviewer] "
    f"{'Approved.' if approved else 'Revision requested.'}")
    if approved:
        print(f"[Graph] Approved after {state['iteration']} iterations.")
    return {**state, "review_feedback": "" if approved else feedback, "logs": logs}


def should_continue(state: DocState) -> str:
    if not state["review_feedback"]: return "finish"
    if state["iteration"] >= MAX_ITERATIONS:
        print("[Graph] Max iterations reached.")
        return "finish"
    return "revise"


def build_graph():
    g = StateGraph(DocState)
    g.add_node("modifier", modifier_agent)
    g.add_node("reviewer", reviewer_agent)
    g.set_entry_point("modifier")
    g.add_edge("modifier", "reviewer")
    g.add_conditional_edges("reviewer", should_continue, {"revise": "modifier", "finish": END})
    return g.compile()


def run(code_diff: str, commit_sha: str = "manual", trigger_source: str = "manual"):
    current_docs, current_version = fetch_latest_doc()
    new_version = current_version + 1
    print(f"[Store] Current v{current_version} -> creating v{new_version}")

    parts = []
    for folder in ["app", "agents"]:
        for f in sorted(Path(folder).glob("**/*.py")):
            parts.append(
                f"=== {f} ===\n{f.read_text(encoding='utf-8')}"
            )
    file_contents = "\n\n".join(parts)

    initial: DocState = {
        "code_diff": code_diff, "current_docs": current_docs,
        "file_contents": file_contents, "current_version": current_version,
        "commit_sha": commit_sha, "trigger_source": trigger_source,
        "updated_docs": "", "review_feedback": "", "iteration": 0, "logs": [],
    }

    print("\n=== Starting agentic documentation update ===")
    final = build_graph().invoke(initial)

    # Save latest documentation to repository
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    (Path("docs/latest.md")).write_text(
        final["updated_docs"],
        encoding="utf-8"
    )

    print("[Store] Updated docs/latest.md")

    path = upload_doc(final["updated_docs"], new_version)
    version_id = save_version(new_version, path, commit_sha, trigger_source)
    for log in final["logs"]:
        save_agent_log(version_id, log["agent"], log["iteration"], log["message"])

    print(f"\n=== Done. v{new_version} live. {len(final['logs'])} agent steps logged. ===")
    return final["updated_docs"]


if __name__ == "__main__":
    run(
        sys.argv[1] if len(sys.argv) > 1 else "Manual trigger.",
        sys.argv[2] if len(sys.argv) > 2 else "manual",
        sys.argv[3] if len(sys.argv) > 3 else "manual",
    )