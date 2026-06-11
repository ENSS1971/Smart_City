FROM python:3.11-slim

# Cria o diretório de trabalho dentro do servidor
WORKDIR /code

# Copia e instala as bibliotecas
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copia o restante dos arquivos do projeto
COPY . .

# Comando de inicialização exigido pelo Hugging Face (Porta padrão 7860)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]