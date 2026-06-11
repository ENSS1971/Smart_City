import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from pymongo import MongoClient
import requests
import gdown  # Biblioteca especialista em downloads do Google Drive

# 1. CARREGA AS VARIÁVEIS DE AMBIENTE SECRETAS (.env)
load_dotenv()

# 2. CONFIGURAÇÃO DO ARQUIVO DO MODELO DE IA
NOME_MODELO = "modelo_defesa_maxima_cifar10.keras"

# ⚠️ SUBSTITUA O LINK ABAIXO PELO LINK DO SEU ARQUIVO DO GOOGLE DRIVE
LINK_GOOGLE_DRIVE = "https://drive.google.com/file/d/1Bnd90SLLO5DOSF6DDwyheAvVIlevR5Qm/view?usp=drive_link"

# 3. MECANISMO AUTO-DOWNLOAD (Garante o deploy leve na nuvem)
if not os.path.exists(NOME_MODELO):
    print(f"📥 O cérebro da IA ({NOME_MODELO}) não foi encontrado localmente.")
    print("Iniciando download seguro do modelo direto do Google Drive...")
    try:
        gdown.download(LINK_GOOGLE_DRIVE, NOME_MODELO, quiet=False)
        print("✅ Download do cérebro da IA concluído com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro crítico ao buscar o modelo no Google Drive: {e}")

# 4. INICIALIZAÇÃO DA API E CARREGAMENTO DO MODELO
app = FastAPI(
    title="UrbeIA - Sistema Inteligente de Monitoramento Urbano",
    description="API robusta de IA integrada ao MongoDB Atlas com deploy automatizado.",
    version="1.2.0"
)

modelo_smart_city = tf.keras.models.load_model(NOME_MODELO)

# Pasta física local para salvar as evidências visuais
PASTA_IMAGENS = "imagens_ocorrencias"
if not os.path.exists(PASTA_IMAGENS):
    os.makedirs(PASTA_IMAGENS)

# 5. CONEXÃO SEGURA COM O MONGODB ATLAS
string_conexao_secreta = os.getenv("MONGO_URI")
if not string_conexao_secreta:
    raise ValueError("ERRO CRÍTICO: A variável MONGO_URI não foi encontrada no ambiente!")

cliente = MongoClient(string_conexao_secreta)
db = cliente["urbeia_db"]
colecao = db["chamados"]

# 6. LOGÍSTICA URBANA (Roteamento de Secretarias)
def rotear_classe_cifar(indice_classe):
    if indice_classe == 2: 
        return "Passaro (Fauna Silvestre)", "Policia Militar Ambiental - Comando de Combate ao Trafico", "CRITICO - Investigacao de Cativeiro"
    elif indice_classe in [1, 9]: 
        return "Automovel/Caminhao", "Secretaria de Mobilidade Urbana e Transito", "Aberto - Aguardando Fiscalizacao"
    elif indice_classe in [3, 5]: 
        return "Gato/Cachorro", "Secretaria de Bem-Estar Animal (Zoonoses)", "Aberto - Triagem Veterinaria"
    elif indice_classe in [4, 7]: 
        return "Cervo/Cavalo", "Guarda Ambiental / Defesa Civil", "URGENTE - Animal de Grande Porte na Via"
    return "Nao Identificado", "Secretaria de Servicos Urbanos", "Aberto - Triagem Geral"

# 7. ENDPOINT PRINCIPAL: REGISTRO DE OCORRÊNCIAS IMEDIATO
@app.post("/registrar_ocorrencia/", summary="Registra ocorrência com GPS manual ou automático por IP")
async def registrar_ocorrencia(
    foto: UploadFile = File(...),
    latitude: float | None = Form(None, description="Opcional. Se omitido, usará o IP do computador"),
    longitude: float | None = Form(None, description="Opcional. Se omitido, usará o IP do computador")
):
    # SE O GPS VIER VAZIO, CAPTURA AUTOMATICAMENTE O IP DO COMPUTADOR
    if latitude is None or longitude is None:
        print("📍 Coordenadas não enviadas. Ativando rastreamento automático por IP...")
        try:
            resposta_geo = requests.get("https://ipapi.co/json/", timeout=5).json()
            latitude = float(resposta_geo.get("latitude", 0.0))
            longitude = float(resposta_geo.get("longitude", 0.0))
            print(f"🌍 Localização aproximada encontrada: {resposta_geo.get('city')} - {resposta_geo.get('region')}")
        except Exception as e:
            print(f"⚠️ Falha ao rastrear IP, aplicando coordenadas padrão (0,0): {e}")
            latitude = 0.0
            longitude = 0.0

    # Processamento da imagem recebida
    conteudo_foto = await foto.read()
    nparr = np.frombuffer(conteudo_foto, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Prepara a imagem para o padrão exigido pela rede neural (224x224)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_redimensionada = tf.image.resize(img_rgb, (224, 224))
    img_batch = tf.expand_dims(img_redimensionada, axis=0)
    img_pronta = preprocess_input(img_batch)
    
    # Executa a inteligência artificial
    predicoes = modelo_smart_city.predict(img_pronta, verbose=0)
    indice_classe = np.argmax(predicoes[0])
    confianca = float(predicoes[0][indice_classe] * 100)
    
    # Aplica as regras de negócio da cidade
    classe_str, secretaria, status_inicial = rotear_classe_cifar(indice_classe)
    protocolo_id = datetime.now().strftime("%Y%m%d%H%M%S")
    horario_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Salva a imagem fisicamente na pasta de auditoria
    caminho_foto = os.path.join(PASTA_IMAGENS, f"{protocolo_id}.jpg")
    cv2.imwrite(caminho_foto, frame)
    
    # Estrutura o documento NoSQL para persistência eterna
    documento_chamado = {
        "protocolo": protocolo_id,
        "data_hora": str(horario_atual),
        "classe_detectada": classe_str,
        "confianca_ia": round(confianca, 2),
        "secretaria_destino": secretaria,
        "status": status_inicial,
        "geolocalizacao": {
            "latitude": latitude,
            "longitude": longitude
        },
        "caminho_imagem": caminho_foto
    }
    
    # Envia para a nuvem do MongoDB
    resultado_banco = colecao.insert_one(documento_chamado)
    
    return {
        "status_requisicao": "Sucesso",
        "id_banco": str(resultado_banco.inserted_id),
        "protocolo": protocolo_id,
        "ia_insights": {
            "alvo": classe_str,
            "certeza": f"{confianca:.2f}%"
        },
        "encaminhado_para": secretaria,
        "geolocalizacao_gravada": {
            "latitude": latitude,
            "longitude": longitude
        },
        "status_chamado": status_inicial
    }