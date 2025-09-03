import asyncio

def retrieval_agent(vectorstore, query, k=2):
    """
    Retrieves top k relevant documents from Milvus for a given query.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)


async def async_retrieve_agent(vectorstore, query, k=2):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, retrieval_agent, vectorstore, query, k)


