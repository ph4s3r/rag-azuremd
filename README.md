# RAG / Azure

what it do>?

 - you fire a query related to azure, let's say data is indexed already to the vector DB (if not, it will do):
 - the query gets transformed to a vector using the same embedding model used in the indexing phase
 - this query vector is then matched against all vectors in the vector database to find the most similar ones (e.g., using the Euclidean distance metric) that might contain the answer to the user’s question. This step is about identifying relevant knowledge chunks.
 - gpt takes the user’s question and the relevant information retrieved from the vectordb to create a response. This process combines the question with the identified data to generate an answer.

TODO:
 - can the context extended? like there will be only one big conversation - ever growing and it remembers everything
 - if not: the point would be: first the user asks a question (need to maintain sessions or conversations - like with yago, need to be able to select an older thread or open new one)
 - new indexer: error loading virtual-network\what-is-ip-address-168-63-129-16.md: 'lxml.etree._ProcessingInstruction' object has no attribute 'is_phrasing' 