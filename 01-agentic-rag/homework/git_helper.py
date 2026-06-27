from gitsource import GithubRepositoryDataReader, chunk_documents

def load_lessons() -> list[dict[str, str]]:
    # Keeping all values hardcoded since we are working with a specific commit
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()

    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)

    return documents

    # chunks = chunk_documents(documents, size=2000, step=1000)

    # return chunks

