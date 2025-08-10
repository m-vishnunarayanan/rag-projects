import os
from dotenv import load_dotenv

import openai
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

openai.api_key = os.environ['OPENAI_API_KEY']
user_query = "Who is the most honoured actor?"


llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

embedding_function = OpenAIEmbeddings()
str_output_parser = StrOutputParser()



# RAG Flow
# Fetch Documents -> Split Documents for storing in Vector DB -> Retreive \
# with Context -> Generate Response

# Retreive a Wikipedia page
retriever = WikipediaRetriever()
docs = retriever.invoke("71st_National_Film_Awards")
print(docs)

# Split
# Though RecursiveCharacterTextSplitter works well for smaller document, 
# we have used Semantic based chunking
text_splitter = SemanticChunker(embedding_function) 
splits = text_splitter.split_documents(docs)


# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embedding_function)

retriever = vectorstore.as_retriever()



prompt = hub.pull("jclemens24/rag-prompt")



if __name__ == "__main__":
    # Chain it all together with LangChain
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | str_output_parser
    )


    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)



    # Question - run the chain
    result = rag_chain_with_source.invoke(user_query)
    print(f"Question asked {user_query}")
    print(f"Result : {result['answer']}")