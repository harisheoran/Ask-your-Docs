from dotenv import load_dotenv
import streamlit as ui
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


def extract_text_from_pdf(my_pdf):
    text = ""
    reader = PdfReader(my_pdf)
    for page in reader.pages:
        text += page.extract_text()
    
    return text

def main():
    load_dotenv()
    ui.set_page_config(page_title="Ask your PDFs")
    ui.header("Ask your pdfs")
    my_pdf = ui.file_uploader("Upload your PDF", type="pdf")
    
    if my_pdf is not None:
    #extraxt text from pdf
        my_text = extract_text_from_pdf(my_pdf)
        #ui.write(my_text)
        
    # 02.create chunks
        splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    
        chunks = splitter.split_text(my_text)
        
        # 03. save chunks into vector db as embeddings/vector
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_texts(chunks, embeddings)
        
        user_query = ui.text_input("Ask a question from your PDF.")
        if user_query:
            found_doc = db.similarity_search(user_query)
            llm = OpenAI()
            chain = load_qa_chain(llm ,chain_type="stuff")
            res = chain.run(input_documents=found_doc, question=user_query)
            ui.write(res)
    
      
if __name__ == "__main__":
    main()
