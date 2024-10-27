import openai
import configparser
from termcolor import cprint
from indexer import Indexer # local
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# config
config = configparser.ConfigParser()
config.read('conf.ini')
openai.api_key = config['OPENAI'].get('APIKEY')

# where do we read data from?
DOC_DIR = "./virtual-network"

# initialize and run the indexer
indexer = Indexer(
    doc_dir=DOC_DIR, 
    chroma_path_prefix=config['CHROMADB'].get('PREFIX'),
    openai_api_key=openai.api_key, 
    rpm=int(config['OPENAI'].get('RPM', 3))
    )

# the chroma db will be in memory for the run
db_chroma = indexer.index_documents()

# ----- retrieval and generation process -----
cprint("####################################################################################", "blue")
cprint(f"WORKING WITH THE FOLLOWING INDEX: {DOC_DIR}", "black", "on_cyan", attrs=["bold"])
cprint("####################################################################################", "blue")
query = 'can virtual networks across tenants be peered?'
cprint(f"QUERY: {query}", "green")
cprint("####################################################################################", "blue")
docs_chroma = db_chroma.similarity_search_with_score(query, k=10)

# filter by minimum similarity score (e.g., 0.8)
min_similarity_threshold = 0.22
filtered_docs = [(doc, score) for doc, score in docs_chroma if score >= min_similarity_threshold]

# prepare prompt based on the availability of relevant documents
if filtered_docs:
    # documents are relevant, so use them in the prompt
    print("\n--- retrieved documents and sources ---\n")
    context_texts = []
    for doc, score in filtered_docs:
        print(f"content: {doc.page_content[:200]}...")
        cprint(f"source: {doc.metadata.get('source', 'unknown source')}", "black", "on_cyan", attrs=["bold"])
        print(f"score: {score}\n")
        context_texts.append(doc.page_content)

    # combine retrieved contexts for the prompt
    context_text = "\n\n".join(context_texts)

    # use the standard prompt template
    PROMPT_TEMPLATE = """
    answer the question based only on the following context:
    {context}
    answer the question based on the above context: {question}.
    provide a detailed answer.
    don't justify your answers.
    don't give information not mentioned in the context information.
    do not say "according to the context" or "mentioned in the context" or similar.
    """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
else:
    # no relevant documents found, so instruct the model to use its own knowledge
    cprint("no highly relevant documents found in local data. proceeding with llm general knowledge.", "red")
    fallback_prompt_template = """
    the local indexed data did not contain any information directly related to the question.
    answer the question based on your general knowledge:
    {question}.
    """
    prompt_template = ChatPromptTemplate.from_template(fallback_prompt_template)
    prompt = prompt_template.format(question=query)

# generate response
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    openai_api_key=openai.api_key
)
response_text = model.invoke(prompt)

# display the final response
response_content = response_text.content
cprint("####################################################################################", "blue")
cprint("\n--- MODEL RESPONSE ---\n", "green")
print(response_content)
cprint("####################################################################################", "blue")
