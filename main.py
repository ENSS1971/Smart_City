import requests
import os 
from dotenv import load_dotenv 
from fastapi import FastAPI, UploadFile, File, Form
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
from pymongo import MongoClient
from datetime import datetime

# 1. CARREGA AS VARIÁVEIS DE AMBIENTE SECRETAS
load_dotenv()

app = FastAPI(
    title="UrbeIA - Sistema Inteligente de Monitoramento Urbano",
    description="API robusta e segura integrada ao MongoDB Atlas.",
    version="1.1.1"
)

# Carrega o modelo de IA
modelo_smart_city = tf.keras.models.load_model('modelo_defesa_maxima_cifar10.keras')

# Pasta física local para salvar as evidências visuais
PASTA_IMAGENS = "imagens_ocorrencias"
if not os.path.exists(PASTA_IMAGENS):
    os.makedirs(PASTA_IMAGENS)

# Captura a string de conexão do arquivo .env
string_conexao_secreta = os.getenv("MONGO_URI")
if not string_conexao_secreta:
    raise ValueError("ERRO CRÍTICO: A variável MONGO_URI não foi encontrada no arquivo .env!")

# Conecta ao MongoDB
cliente = MongoClient(string_conexao_secreta)
db = cliente["urbeia_db"]
colecao = db["chamados"]

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
            # Consulta um serviço de geolocalização baseado no IP público da sua rede
            resposta_geo = requests.get("https://ipapi.co/json/", timeout=5).json()
            latitude = float(resposta_geo.get("latitude"))
            longitude = float(resposta_geo.get("longitude"))
            print(f"🌍 Localização aproximada encontrada: {resposta_geo.get('city')} - {resposta_geo.get('region')}")
        except Exception as e:
            print(f"⚠️ Falha ao rastrear IP, aplicando coordenadas padrão (0,0): {e}")
            latitude = 0.0
            longitude = 0.0

    # 1. Transforma o arquivo de imagem enviado
    conteudo_foto = await foto.read()
    nparr = np.frombuffer(conteudo_foto, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Prepara a imagem para a sua ResNet50
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_redimensionada = tf.image.resize(img_rgb, (224, 224))
    img_batch = tf.expand_dims(img_redimensionada, axis=0)
    img_pronta = preprocess_input(img_batch)
    
    # 3. Executa a predição do Modelo
    predicoes = modelo_smart_city.predict(img_pronta, verbose=0)
    indice_classe = np.argmax(predicoes[0])
    confianca = float(predicoes[0][indice_classe] * 100)
    
    # 4. Define o destino inteligente e gera os IDs temporais
    classe_str, secretaria, status_inicial = rotear_classe_cifar(indice_classe)
    protocolo_id = datetime.now().strftime("%Y%m%d%H%M%S")
    horario_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 5. Salva a foto física na pasta local
    caminho_foto = os.path.join(PASTA_IMAGENS, f"{protocolo_id}.jpg")
    cv2.imwrite(caminho_foto, frame)
    
    # 6. Estrutura o documento JSON perfeito para o MongoDB NoSQL
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
    
    # 7. Dispara o salvamento para a nuvem!
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