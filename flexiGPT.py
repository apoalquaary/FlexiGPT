
#################################################

#       Done by: apoalquaary - 9/9/2023

#################################################

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
import transformers
from torch import cuda, bfloat16
import time
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
import os

from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

import argparse

parser = argparse.ArgumentParser(
                    prog='FlexiGPT',
                    description='This program gives you the ability to talk with your files using the embeddings and LLM that you want from huggingface. You can also use 4-bit or 8-bit quantization on models so your local device can handle running those models locally',
                    epilog='check the paper of this project or the github repo "FlexiGPT" to get more info')


# command-line arguments
parser.add_argument('--llm_model', type=str, default='meta-llama/Llama-2-7b-chat-hf', 
	help='You can add any llm model you want from HF (default model: meta-llama/Llama-2-7b-chat-hf).')

parser.add_argument('--embeddings_model', type=str, default='BAAI/bge-base-en', 
	help='You can add any embeddings model you want from HF (default model: BAAI/bge-base-en).')

parser.add_argument('--dir_path', type=str, default='docs', 
	help='You can add the directory that has your files. (files can be txt, docx, pdf).')

parser.add_argument('--embedding_device', type=str, default='cuda', 
	help='add the appropriate device for your machine. (it expects one of the following: cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone).')

parser.add_argument('--retriever_k', type=int, default=5, 
	help='the number of retriever chunks to get information from.')

parser.add_argument('--loading_bit', type=str, default='4bit', 
	help='You can choose how to quantize the model. (loading bit can be 4bit or 8bit).')

parser.add_argument('--source_documents', action='store_true', help='show the original data that were found by the similarity search')

args = parser.parse_args()


documents = []
for file in os.listdir(args.dir_path):
    if file.endswith('.pdf'):
        pdf_path = './docs/' + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = './docs/' + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = './docs/' + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())


# splitting the text into chunks

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
texts = text_splitter.split_documents(documents)

# Text Embedding


model_name = args.embeddings_model
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': args.embedding_device},
    encode_kwargs=encode_kwargs,
    cache_folder='embedding_models/',
)

persist_directory = 'db'
embedding = model_norm
vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

retriever = vectordb.as_retriever(search_kwargs={"k": args.retriever_k})

# llm model
bnb_config_4 = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

bnb_config_8 = transformers.BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

if args.loading_bit == "4bit":
    bnb_config = bnb_config_4
else:
    bnb_config = bnb_config_8

model_id = args.llm_model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir='models/' , quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True) # cache_dir


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    pad_token_id = 50256,
    repetition_penalty=1.15,
)

local_llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  chain_type_kwargs=chain_type_kwargs)

def process_llm_response(query, llm_response):
    
    print(f"\n\n> Question:\n{query}\n\n")
    print(f"> Answer:\n{llm_response['result']}")
    
    if args.source_documents:
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source) # .metadata['source']



while 1:
    query = input("\nEnter a query: ")
    if query.lower() == "exit":
        break
        # query = "what is bertopic?"
    llm_response = qa_chain(query)
    process_llm_response(query, llm_response)


