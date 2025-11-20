def rank_clauses_by_relevance(chunks, question):
    # Add rule-based scoring or keyword checks
    return sorted(chunks, key=lambda x: len(x.page_content), reverse=True)
