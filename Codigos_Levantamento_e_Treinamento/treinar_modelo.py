# -*- coding: utf-8 -*-
"""
Arquivo: treinar_modelo.py
Objetivo: Realizar o treinamento supervisionado do modelo YOLOv26n no dataset
          consolidado de EPIs (unified_tcc - 1.814 imagens e 4 classes).

Uso no terminal:
    1. Validação rápida pré-treino (Dry-Run):
       python "Base e treinamento/Codigos_Levantamento_e_Treinamento/treinar_modelo.py" --dry-run
    2. Treinamento supervisionado completo:
       python "Base e treinamento/Codigos_Levantamento_e_Treinamento/treinar_modelo.py" --epochs 100 --imgsz 640 --batch 16
"""

import os
import sys
import glob
import traceback
import argparse


def pausar_se_interativo(forcar_sem_pausa=False, houve_erro=False):
    """
    Impede que a janela do terminal se feche sozinha ao ocorrer um erro ou ao
    executar o script por duplo clique no Windows Explorer.
    """
    if forcar_sem_pausa:
        return
    try:
        if houve_erro or (sys.stdin and sys.stdin.isatty()):
            print("\n" + "-" * 80)
            input("  [PAUSA] Pressione ENTER para finalizar e fechar esta janela...")
    except (EOFError, KeyboardInterrupt):
        pass


try:
    import yaml
    import torch
    from ultralytics import YOLO
except ImportError as exc_imp:
    print("\n" + "!" * 80)
    print("  [ERRO DE DEPENDÊNCIA EM treinar_modelo.py]")
    print("!" * 80)
    print(f"  -> Biblioteca ausente: {exc_imp}")
    print("  -> Como resolver: Execute no terminal:")
    print("     pip install ultralytics torch torchvision pyyaml opencv-python")
    print("!" * 80)
    pausar_se_interativo(forcar_sem_pausa=False, houve_erro=True)
    raise SystemExit(1) from exc_imp


def validar_e_corrigir_caminho_yaml(caminho_yaml, pasta_atual):
    """
    Lê o arquivo data.yaml e verifica se a chave 'path' aponta para um diretório existente
    neste computador. Caso aponte para um caminho antigo (ex.: 'D:/ProjetoFinal2/...'),
    atualiza automaticamente para o caminho real de 'unified_tcc' na máquina atual.
    """
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        dados_yaml = yaml.safe_load(f) or {}

    caminho_configurado = dados_yaml.get("path", "")
    if not caminho_configurado or not os.path.exists(caminho_configurado):
        # Procura o diretório real do dataset unified_tcc neste computador
        base_dir = os.path.dirname(pasta_atual)
        raiz_ws = os.path.dirname(base_dir)
        candidatos_dataset = [
            os.path.join(base_dir, "datasets", "processed", "unified_tcc"),
            os.path.join(raiz_ws, "Projeto_EPI", "datasets", "processed", "unified_tcc"),
            os.path.dirname(os.path.abspath(caminho_yaml)),
        ]
        for cand in candidatos_dataset:
            if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "images", "train")):
                novo_path = os.path.abspath(cand).replace("\\", "/")
                print(f"  [*] Ajustando caminho raiz em '{os.path.basename(caminho_yaml)}' para:\n      {novo_path}")
                dados_yaml["path"] = novo_path
                with open(caminho_yaml, "w", encoding="utf-8") as fw:
                    yaml.dump(dados_yaml, fw, sort_keys=False, allow_unicode=True)
                break

    return dados_yaml


def resolver_dispositivo(dispositivo):
    """
    Verifica de forma segura se o CUDA está realmente disponível antes de alocar GPU,
    evitando crash caso o usuário passe '--device 0' em um ambiente PyTorch somente CPU.
    """
    cuda_ativo = torch.cuda.is_available()
    if dispositivo is None or str(dispositivo).strip() == "":
        return (0 if cuda_ativo else "cpu"), cuda_ativo

    disp_str = str(dispositivo).strip().lower()
    if disp_str != "cpu" and not cuda_ativo:
        print(f"  [!] AVISO: Foi solicitado dispositivo '{dispositivo}', mas o PyTorch atual não possui GPU CUDA ativa.")
        print("      Alternando automaticamente para 'cpu' para evitar interrupção do script.")
        return "cpu", False

    return dispositivo, cuda_ativo


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
    nome_experimento="ppe_detection_model",
    apenas_validar=False,
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

    if not os.path.exists(arquivo_arquitetura):
        raise FileNotFoundError(f"Arquivo de arquitetura não encontrado em: {arquivo_arquitetura}")
    if not os.path.exists(pesos_iniciais):
        raise FileNotFoundError(f"Arquivo de pesos iniciais não encontrado em: {pesos_iniciais}")

    # Valida e corrige se necessário o caminho interno do data.yaml
    dados_yaml = validar_e_corrigir_caminho_yaml(caminho_yaml, pasta_atual)
    pasta_banco = dados_yaml.get("path", "")

    # Conta imagens das partições para exibir diagnóstico claro antes de treinar
    n_tr = len(glob.glob(os.path.join(pasta_banco, "images", "train", "*.*"))) if os.path.isdir(pasta_banco) else 0
    n_va = len(glob.glob(os.path.join(pasta_banco, "images", "val", "*.*"))) if os.path.isdir(pasta_banco) else 0
    n_te = len(glob.glob(os.path.join(pasta_banco, "images", "test", "*.*"))) if os.path.isdir(pasta_banco) else 0

    # Seleciona GPU (0) se o CUDA estiver disponível, caso contrário usa CPU com segurança
    dispositivo_final, cuda_disponivel = resolver_dispositivo(dispositivo)

    print("\n" + "=" * 80)
    print("  CONFIGURAÇÃO E EXECUÇÃO DO TREINAMENTO SUPERVISIONADO (YOLOv26n)")
    print("=" * 80)
    print(f"  -> Arquitetura : {arquivo_arquitetura}")
    print(f"  -> Pesos Base  : {pesos_iniciais}")
    print(f"  -> Manifesto   : {caminho_yaml}")
    print(f"  -> Pasta Dados : {pasta_banco}")
    print(f"  -> Imagens     : {n_tr + n_va + n_te} totais (Treino: {n_tr} | Val: {n_va} | Teste: {n_te})")
    print(f"  -> Dispositivo : {dispositivo_final} (GPU CUDA ativa: {cuda_disponivel})")
    print(f"  -> Parâmetros  : epochs={epocas}, batch={tamanho_lote}, imgsz={tamanho_imagem}, lr0={taxa_aprendizado}, patience={paciencia}")
    print("-" * 80 + "\n")

    if (n_tr + n_va) == 0:
        raise RuntimeError(
            f"Nenhuma imagem de treino/validação foi encontrada em '{pasta_banco}'.\n"
            "Execute primeiro: python gerenciar_bases.py --action prepare --apply-clahe"
        )

    # Carrega a estrutura da rede e transfere os pesos pré-treinados
    modelo = YOLO(arquivo_arquitetura).load(pesos_iniciais)
    modelo.info(detailed=False, verbose=True)

    # Modo de validação rápida (Dry-Run) apenas para testar se está tudo pronto
    if apenas_validar:
        print("\n" + "=" * 80)
        print("  [OK] VALIDAÇÃO PRÉ-TREINO (DRY-RUN) CONCLUÍDA COM SUCESSO!")
        print("=" * 80)
        print(f"  -> Pasta do banco : {pasta_banco}")
        print(f"  -> Partições      : Treino={n_tr} | Validação={n_va} | Teste={n_te} (Total={n_tr + n_va + n_te} imagens)")
        print(f"  -> Total classes  : {dados_yaml.get('nc')}")
        print(f"  -> Classes        : {dados_yaml.get('names')}")
        print("=" * 80 + "\n")
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
        name=nome_experimento,
        exist_ok=True,
        device=dispositivo_final,
    )

    caminho_melhor_peso = os.path.join(pasta_saida, nome_experimento, "weights", "best.pt")
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
    parser.add_argument("--project", type=str, default=None, help="Pasta de saída dos resultados do treinamento")
    parser.add_argument("--name", type=str, default="ppe_detection_model", help="Nome do experimento")
    parser.add_argument("--dry-run", action="store_true", help="Apenas testa o carregamento e valida as pastas sem treinar")
    parser.add_argument("--no-pause", action="store_true", help="Não pausa o terminal ao finalizar")
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        pasta_script = os.path.dirname(os.path.abspath(__file__))
        raiz_workspace = os.path.dirname(os.path.dirname(pasta_script))

        caminhos_possiveis = [
            args.data,
            os.path.abspath(os.path.join(pasta_script, "../datasets/processed/unified_tcc/data.yaml")),
            os.path.abspath(os.path.join(pasta_script, "data.yaml")),
            os.path.abspath(os.path.join(raiz_workspace, "Projeto_EPI/datasets/processed/unified_tcc/data.yaml")),
            os.path.abspath(os.path.join(pasta_script, "../datasets/raw/construction-ppe/data.yaml")),
        ]

        caminho_encontrado = next((p for p in caminhos_possiveis if p and os.path.exists(p)), None)

        # Se ainda não existir o dataset processado unified_tcc, constrói automaticamente para evitar erro!
        dataset_unified_dir = os.path.abspath(os.path.join(pasta_script, "../datasets/processed/unified_tcc"))
        if not os.path.isdir(os.path.join(dataset_unified_dir, "images", "train")):
            print("  [*] O dataset consolidado 'unified_tcc' ainda não estava preparado nesta pasta.")
            print("  [*] Executando preparação automática (preparar_dados.py) antes de iniciar...")
            from preparar_dados import build_unified_dataset
            caminho_encontrado = build_unified_dataset(use_clahe=True, clip_limit=3.0)

        if caminho_encontrado and os.path.exists(caminho_encontrado):
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
                pasta_saida=args.project,
                nome_experimento=args.name,
                apenas_validar=args.dry_run,
            )
        else:
            erro_ocorreu = True
            print("\n" + "!" * 80)
            print("  [ERRO] Arquivo 'data.yaml' não encontrado nos caminhos pesquisados:")
            for p in caminhos_possiveis:
                if p:
                    print(f"    - {p}")
            print("\n  [Como resolver]: Execute antes o script:")
            print('    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/gerenciar_bases.py" --action all --apply-clahe')
            print("!" * 80)
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 80)
        print("  [ERRO DETECTADO EM treinar_modelo.py]")
        print("!" * 80)
        print(f"  -> Tipo do erro : {type(exc).__name__}")
        print(f"  -> Mensagem     : {exc}")
        print("\n  [Detalhes técnicos (Traceback)]:")
        traceback.print_exc()
        print("!" * 80)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
