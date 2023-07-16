from typing import Union
from app.ProcessamentoTexto import TextPreprocessor
from fastapi import FastAPI
from fastapi.responses import JSONResponse 
app = FastAPI(
    title="Processamento de Texto",
    description="Uma API que faz processamento de texto, conhecido tambem como Processamento de Linguagem Natural",
    version="1.0.0"
)
ppt  = TextPreprocessor()

@app.get("/Tokenização/{text}",summary="Separa as partes mais importantes do texto", description="Retorna uma lista TOKENIZADA,  do texto que você vai colocar.")
def Tokenizacao(text):
    tokens = ppt.preprocess_text(text)
    return JSONResponse(content={"tokens": tokens})

