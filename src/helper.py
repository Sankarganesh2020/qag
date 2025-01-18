import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.prompt import *
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
class LLMPipeline:
    def __init__(self):
        load_dotenv()

        # Set Azure OpenAI credentials
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        os.environ["AZURE_OPENAI_ENDPOINT"] = self.azure_openai_endpoint

        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        os.environ["AZURE_OPENAI_API_KEY"] = self.azure_openai_api_key

        # Initialize AzureChatOpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        # Create embeddings 
        self.embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")
        # # self.vector_store = FAISS("", self.embeddings) 
        # index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        # self.vector_store = FAISS(
        #     embedding_function=self.embeddings,
        #     index=index,
        #     docstore= InMemoryDocstore(),
        #     index_to_docstore_id={}
        # )

    def chunk_documents(self, content):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        documents = splitter.split_documents(content)
        return documents

    def update_vector_store(self, content):
        self.vector_store = FAISS.from_documents(content, self.embeddings)
        self.vector_store.save_local("faiss_store")
        

    def get_vector_store(self):        
        self.vector_store = FAISS.load_local("faiss_store", self.embeddings, allow_dangerous_deserialization=True)
                      

    def file_processing_qa_gen(self, file_path):
        """
        Process the input PDF file to generate documents for question and answer generation.
        """
        # Load data from PDF
        loader = PyPDFLoader(file_path)
        data = loader.load()

        question_gen = ''
        for page in data:
            question_gen += page.page_content

        # Split content into chunks
        splitter1 = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        chunks1 = splitter1.split_text(question_gen)
        documents1 = [Document(page_content=t) for t in chunks1]

        splitter2 = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        documents2 = splitter2.split_documents(documents1)

        return documents1, documents2

    def qa_chain(self):
        """
        Create a question-answering chain using the provided vector store.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        answer_generation_chain = create_retrieval_chain(self.vector_store.as_retriever(), question_answer_chain)
        return answer_generation_chain

    def generate_qa_gen_pipeline(self, file_path):
        """
        Complete pipeline for processing a PDF file and creating a question-answering chain.
        """
        document_ques_gen, document_answer_gen = self.file_processing_qa_gen(file_path)

        # Define prompts for question generation
        prompt_questions = PromptTemplate(template=prompt_template, input_variables=["text"])
        refine_prompt_questions = PromptTemplate(
            input_variables=["existing_answer", "text"],
            template=refine_template,
        )

        # Generate questions using the summarize chain
        ques_gen_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            verbose=True,
            question_prompt=prompt_questions,
            refine_prompt=refine_prompt_questions,
        )
        ques = ques_gen_chain.invoke(document_ques_gen)

        # Create vector store
        self.update_vector_store(document_answer_gen)

        # Filter the generated questions
        ques_list = ques.get("output_text").split("\n")
        filtered_ques_list = [
            element for element in ques_list if element.endswith('?') or element.endswith('.')
        ]

        # Create the answer generation chain
        answer_generation_chain = self.qa_chain()

        return answer_generation_chain, filtered_ques_list


    def create_website_knowledge_base(self, urls):
        loaders = UnstructuredURLLoader(urls=urls)
        data = loaders.load()
        documents = self.chunk_documents(data)
        # Create vector store
        self.update_vector_store(documents) 
        website_qa_chain = self.qa_chain()
        return website_qa_chain



def llm_qa_gen_pipeline(file_path):
    pipeline = LLMPipeline()
    answer_chain, questions = pipeline.generate_qa_gen_pipeline(file_path)
    return answer_chain, questions



def llm_website_content_pipeline(url):
    URLs=[]
    URLs.append(url)
    pipeline = LLMPipeline()
    pipeline.create_website_knowledge_base(URLs)
    return "Ok"


def get_website_chatbot_response():
    pipeline = LLMPipeline()
    pipeline.get_vector_store()
    website_qa_chain = pipeline.qa_chain()
    return website_qa_chain