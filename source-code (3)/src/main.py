import os
import tiktoken
import hashlib
from functools import reduce
from apify import Actor
from tqdm.auto import tqdm
from uuid import uuid4
from getpass import getpass
from pinecone import ServerlessSpec
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient


def get_nested_value(data_dict, keys_str):
    keys = keys_str.split('.')
    result = data_dict

    for key in keys:
        if key in result:
            result = result[key]
        else:
            # If any of the keys are not found, return None
            return None

    return result


async def main():
    async with Actor:

        # Get the value of the actor input
        actor_input = await Actor.get_input() or {}

        print(actor_input)

        os.environ['OPENAI_API_KEY'] = actor_input.get('openai_token')


        fields = actor_input.get('fields') or []
        metadata_fields = actor_input.get('metadata_fields') or {}
        metadata_values = actor_input.get('metadata_values') or {}

        PINECONE_API_KEY = actor_input.get('pinecone_token')
        PINECONE_ENV = actor_input.get('pinecone_env')
        OPENAI_API_KEY = actor_input.get('openai_token')

        print("Loading dataset")

        # Iterator over metadata fields
        for field in metadata_fields:
            metadata_fields[field] = get_nested_value(actor_input.get('resource'), metadata_fields[field])
        
        # If you want to process data from Apify dataset before sending it to pinecone, do it here inside iterator function
        def document_iterator(dataset_item):
            m = hashlib.sha256()
            m.update(dataset_item['url'].encode('utf-8'))
            uid = m.hexdigest()[:12]
            return Document(
                page_content=dataset_item['text'],
                metadata={"source": dataset_item['url'], "id": uid}
            )

        loader = ApifyDatasetLoader(
            dataset_id=actor_input.get('resource')['defaultDatasetId'],
            dataset_mapping_function=document_iterator
        )
                
        # Cleaning data before intializing pinecone
        tiktoken_model_name = 'gpt-3.5-turbo'
        tiktoken.encoding_for_model(tiktoken_model_name)
        tokenizer = tiktoken.get_encoding('cl100k_base')

        # create the length function
        def tiktoken_len(text):
            tokens = tokenizer.encode(
                text,
                disallowed_special=()
            )
            return len(tokens)

        # Create text splitter based on length function
        text_splitter1 = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=25,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )

        text_splitter2 = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10,
            length_function=tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )

        print("Loading documents ")

        # load documents from Apify
        documents = loader.load()

        # Split documents into chunks
        parent_child_documents = []
        for doc in documents:
            # First, split document into parent chunks
            parent_chunks = text_splitter1.split_text(doc.page_content)
            for parent_index, parent_chunk in enumerate(parent_chunks):
                # For each parent chunk, split further into child chunks
                child_chunks = text_splitter2.split_text(parent_chunk)
                for child_chunk in child_chunks:
                    # Append parent chunk (larger) and child chunk (smaller) together with metadata
                    parent_child_documents.append(
                        {
                            "parent_content": parent_chunk,
                            "child_content": child_chunk,
                            "metadata": {
                                "source": doc.metadata["source"],
                                "parent_id": f"{doc.metadata['id']}-{parent_index}",
                                "id": doc.metadata["id"]  # Keeping original ID for parent linkage
                            }
                        }
                    )

        print("Embedding child documents and upserting to Pinecone")

        # Revised print statements to match the Document object structure

        # above loading is equivalent to following
        #    chunks = text_splitter.split_text(doc['page_content'])  # get page content from 'page_content' key
        #    for i, chunk in enumerate(chunks):
        #        documents.append({
        #            'id': f'{uid}-{i}',
        #            'text': chunk,
        #            'source': url
        #})

        print("Initializing pinecone")
        pc = PineconeClient(api_key=PINECONE_API_KEY)
        print("Pinecone initialized")
        print(pc)

        index_name = actor_input.get("index_name")
        namespace_name = actor_input.get("namespace_name")

        embeddings = OpenAIEmbeddings(
            model='text-embedding-3-small',
            openai_api_key=OPENAI_API_KEY
        )
        print(embeddings)

        spec = ServerlessSpec(cloud='aws', region='us-west-2')

        # Check if our index already exists. If it doesn't, we create it
        if index_name not in pc.list_indexes().names():
            print("Creating index")
            # Create a new index
            pc.create_index(
                index_name,
                dimension=1536,
                metric='cosine',
                spec=spec
            )
            # Wait for index to be initialized
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            print("Index created!")
        else:
            print("Index already exists, updating index.")

        index = pc.Index(index_name)


        print("generating embeddings:", embeddings)
        # Generate embeddings and prepare documents for upserting
        async def generate_and_upsert_documents(parent_child_documents, index, embeddings):
            batch_size = 100
            document_batches = [parent_child_documents[i:i + batch_size] for i in range(0, len(parent_child_documents), batch_size)]

            for batch in document_batches:
                # Prepare data for embedding, using only child content
                texts = [doc["child_content"] for doc in batch]
                try:
                    # Use the correct method for batch embedding
                    vectors = embeddings.embed_documents(texts)

                    # Prepare upsert data
                    upsert_data = [
                        {
                            "id": doc["metadata"].get("parent_id"),  # Use parent ID for linkage
                            "values": vectors[i],
                            "metadata": {
                                "source": doc["metadata"].get("source"),
                                "parent_content": doc["parent_content"],  # Include parent content in metadata
                                "child_content": doc["child_content"]  # Include child content in metadata
                            }
                        } for i, doc in enumerate(batch)
                    ]

                    # Upsert batch
                    index.upsert(vectors=upsert_data, namespace=namespace_name)
                except Exception as e:
                    print(f"Error during embedding or upserting: {e}")

        # Upsert documents
        await generate_and_upsert_documents(parent_child_documents, index, embeddings)
        print("Documents successfully upserted to Pinecone index")