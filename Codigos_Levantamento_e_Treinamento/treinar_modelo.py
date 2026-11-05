# -*- coding: utf-8 -*-
"""
Arquivo: treinar_modelo.py
Objetivo: Realizar o treinamento supervisionado do modelo YOLOv26n no dataset
          consolidado de EPIs (unified_tcc - 1.814 imagens e 4 classes).
"""

import os
import argparse
import yaml
import torch
from ultralytics import YOLO


def treinar_modelo(
    caminho_yaml,
    epocas=100,
    tamanho_imagem=640,
    tamanho_lote=16,
    taxa_aprendizado=0.01,
    decaimento_peso=0.0005,
    mosaico=1.0,
    paciencia=15,
    dispositivo=None,
    pasta_saida=None,
    apenas_validar=False
):
    """
    Configura e inicia o treinamento do modelo YOLOv26n com os parâmetros
    definidos na metodologia do TCC.
    """
    pasta_atual = os.path.dirname(os.path.abspath(__file__))

    if pasta_saida is None:
        pasta_saida = os.path.abspath(os.path.join(pasta_atual, "../models"))

    # Localiza o arquivo de arquitetura (.yaml) e os pesos iniciais (.pt)
    arquivo_arquitetura = os.path.join(pasta_atual, "yolo26n.yaml")
    pesos_iniciais = os.path.join(pasta_atual, "yolo26n.pt")

    if not os.path.exists(arquivo_arquitetura):
        arquivo_arquitetura = os.path.join(pasta_atual, "yolov26n.yaml")
    if not os.path.exists(pesos_iniciais):
        pesos_iniciais = os.path.join(pasta_atual, "yolov26_weights.pt")

    # Seleciona GPU (0) se o CUDA estiver disponível, caso contrário usa CPU
    if dispositivo is None:
        dispositivo = 0 if torch.cuda.is_available() else "cpu"

    print("\n--- Iniciando Configuração do Treinamento (YOLOv26n) ---")
    print(f"Arquitetura : {arquivo_arquitetura}")
    print(f"Pesos base  : {pesos_iniciais}")
    print(f"Dataset     : {caminho_yaml}")
    print(f"Dispositivo : {dispositivo} (GPU ativa: {torch.cuda.is_available()})")
    print(f"Parâmetros  : epocas={epocas}, lote={tamanho_lote}, img={tamanho_imagem}, paciencia={paciencia}\n")

    # Carrega a estrutura da rede e transfere os pesos pré-treinados
    modelo = YOLO(arquivo_arquitetura).load(pesos_iniciais)
    modelo.info(detailed=False, verbose=True)

    # Modo de validação rápida (Dry-Run) apenas para testar se está tudo pronto
    if apenas_validar:
        with open(caminho_yaml, "r", encoding="utf-8") as f:
            dados_yaml = yaml.safe_load(f)
        print("\n[OK] Validação rápida (Dry-Run) concluída com sucesso!")
        print(f" -> Pasta do banco : {dados_yaml.get('path')}")
        print(f" -> Total classes  : {dados_yaml.get('nc')}")
        print(f" -> Classes        : {dados_yaml.get('names')}\n")
        return modelo

    # Inicia o treinamento supervisionado
    resultados = modelo.train(
        data=caminho_yaml,
        epochs=epocas,
        imgsz=tamanho_imagem,
        batch=tamanho_lote,
        lr0=taxa_aprendizado,
        weight_decay=decaimento_peso,
        mosaic=mosaico,
        patience=paciencia,
        save=True,
        project=pasta_saida,
        name="ppe_detection_model",
        exist_ok=True,
        device=dispositivo
    )

    caminho_melhor_peso = os.path.join(pasta_saida, "ppe_detection_model", "weights", "best.pt")
    print(f"\n[OK] Treinamento finalizado! Melhor modelo salvo em: {caminho_melhor_peso}")
    return resultados


# Alias para compatibilidade com chamadas anteriores
train_model = treinar_modelo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de treinamento do modelo YOLOv26n para EPIs")
    parser.add_argument("--data", type=str, default="", help="Caminho para o arquivo data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Quantidade total de épocas")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem (padrão: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do lote de imagens por iteração")
    parser.add_argument("--lr0", type=float, default=0.01, help="Taxa de aprendizado inicial")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Decaimento de peso (regularização L2)")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Aumentação de dados em mosaico (0.0 a 1.0)")
    parser.add_argument("--patience", type=int, default=15, help="Épocas sem melhoria para parada antecipada")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo ('0' para GPU ou 'cpu')")
    parser.add_argument("--dry-run", action="store_true", help="Apenas testa o carregamento sem treinar")
    args = parser.parse_args()

    pasta_script = os.path.dirname(os.path.abspath(__file__))
    caminhos_possiveis = [
        args.data,
        os.path.abspath(os.path.join(pasta_script, "../datasets/processed/unified_tcc/data.yaml")),
        os.path.abspath(os.path.join(pasta_script, "../datasets/raw/construction-ppe/data.yaml")),
    ]

    caminho_encontrado = next((p for p in caminhos_possiveis if p and os.path.exists(p)), None)

    if caminho_encontrado:
        treinar_modelo(
            caminho_yaml=caminho_encontrado,
            epocas=args.epochs,
            tamanho_imagem=args.imgsz,
            tamanho_lote=args.batch,
            taxa_aprendizado=args.lr0,
            decaimento_peso=args.weight_decay,
            mosaico=args.mosaic,
            paciencia=args.patience,
            dispositivo=args.device,
            apenas_validar=args.dry_run
        )
    else:
        print("[Erro] Arquivo data.yaml não encontrado. Execute antes o script gerenciar_bases.py.")
