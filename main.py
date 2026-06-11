import os
import cv2
import numpy as np
import tensorflow as tf
import gdown
import requests
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form, RuntimeError
from pymongo import MongoClient

# 1. CONFIGURAÇÕES INICIAIS E CONSTANTES
NOME_MODELO = "modelo_defesa_maxima_cifar10.keras"
# Link direto do arquivo compartilhado (Acesso Geral: Qualquer pessoa com o link)
LINK_GOOGLE_DRIVE = "https://drive.google.com/uc?id=1Bnd90SLLO5DOSF6DDwyheAvVIlevR5Qm"

# Classes do CIFAR-10 adaptadas para o contexto da Smart City
CLASSES_IA = {
    0: "Aviao/Asa-Delta",
    1: "Automovel/Caminhao",  # Mapeado no seu teste com 100%
    2: "Passaro/Fauna",
    3: "Gato/Animal-Solto",
    4: "Veado/Animal-Silvestre",
    5: "Cao/Animal-Abandonado",
    6: "Sapo/Anfibio-Praga",
    7: "Cavalo/Tracao-Animal",
    8: "Navio/Embarcacao",
    9: "Caminhao/Carga-Irregular"
}

# 2. DOWNLOAD E CARREGAMENTO SEGURO DO MODELO DE IA
if not os.path.exists(NOME_MODELO):
    print(f"📥 O cérebro da IA ({NOME_MODELO}) não foi encontrado localmente.")
    print("Iniciando download seguro do modelo direto do Google Drive...")
    try:
        gdown.download(LINK_GOOGLE_DRIVE, NOME_MODELO, quiet=False)
        print("✅ Download do cérebro da IA concluído com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro crítico ao buscar o modelo no Google Drive: {e}")

# Carrega o modelo na memória RAM estável do Hugging Face
modelo_smart_city = tf.keras.models.load_model(NOME_MODELO)

# 3. CONEXÃO COM O BANCO DE DADOS NOSQL (MONGODB ATLAS)
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Variável de ambiente MONGO_URI não foi configurada no Hugging Face!")

client = MongoClient(MONGO_URI)
db = client["smart_city_database"]       # Seu Banco de Dados
colecao = db["chamados_prefeitura"]       # Sua Coleção

# 4. INICIALIZAÇÃO DA API FASTAPI
app = FastAPI(
    title="UrbeIA - Smart City API",
    description="API de monitoramento urbano inteligente usando Visão Computacional e Inteligência Artificial.",
    version="1.0.0"
)

# 5. ENDPOINT PRINCIPAL: REGISTRO DE OCORRÊNCIAS
@app.post("/registrar_ocorrencia/", summary="Registra ocorrência com GPS manual ou automático por IP")
async def registrar_ocorrencia(
    request: Request,
    latitude: float = Form(0.0),
    longitude: float = Form(0.0),
    foto: UploadFile = File(...)
):
    # 📌 PASSO 1: SISTEMA INTELIGENTE DE GEOLOCALIZAÇÃO POR IP (CORRIGIDO PARA HUGGING FACE)
    if latitude == 0.0 and longitude == 0.0:
        print("🔍 Coordenadas zeradas. Tentando detectar localização por IP do usuário...")
        
        # Lê o cabeçalho HTTP injetado pelo roteador do Hugging Face
        x_forwarded_for = request.headers.get("x-forwarded-for")
        
        if x_forwarded_for:
            # Captura o primeiro IP da lista (IP público real do computador/celular do cidadão)
            ip_real = x_forwarded_for.split(",")[0].strip()
            print(f"🌐 IP real detectado atrás do Proxy: {ip_real}")
        else:
            ip_real = request.client.host
            print(f"💻 Teste local ou direto detectado. IP: {ip_real}")
        
        # Consulta uma API externa de geolocalização passando o IP real coletado
        try:
            if ip_real not in ["127.0.0.1", "localhost", "0.0.0.0", "10.16.25.137"]:
                resposta_gps = requests.get(f"http://ip-api.com/json/{ip_real}").json()
                if resposta_gps.get("status") == "success":
                    latitude = resposta_gps.get("lat", 0.0)
                    longitude = resposta_gps.get("lon", 0.0)
                    print(f"🛰️ Geolocalização por IP aplicada! Lat: {latitude}, Lon: {longitude}")
        except Exception as erro_gps:
            print(f"⚠️ Falha na geolocalização por IP: {erro_gps}")