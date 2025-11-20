from llm.gemini_interface import generate_answer
from langchain.schema import Document

def process_question_with_clauses(question: str, clauses: list[Document]) -> str:
    # Additional filters or rule-based logic can go here
    return generate_answer(question, clauses)
