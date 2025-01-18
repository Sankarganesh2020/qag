import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.prompt import *

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

    def qa_chain(self, vector_store):
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
        answer_generation_chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)
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

        # Create embeddings and vector store
        embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_documents(document_answer_gen, embeddings)

        # Filter the generated questions
        ques_list = ques.get("output_text").split("\n")
        filtered_ques_list = [
            element for element in ques_list if element.endswith('?') or element.endswith('.')
        ]

        # Create the answer generation chain
        answer_generation_chain = self.qa_chain(vector_store)

        return answer_generation_chain, filtered_ques_list


def llm_qa_gen_pipeline(file_path):
    pipeline = LLMPipeline()
    answer_chain, questions = pipeline.generate_qa_gen_pipeline(file_path)
    return answer_chain, questions