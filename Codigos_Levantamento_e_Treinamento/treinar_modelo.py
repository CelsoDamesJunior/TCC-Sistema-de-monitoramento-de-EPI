# -*- coding: utf-8 -*-
"""
Módulo: treinar_modelo.py
Executa o treinamento supervisionado da arquitetura YOLOv26n sobre o dataset
consolidado de EPIs (unified_tcc - 4 bases nativas YOLO .txt e 4 classes NR-06 / NR-18).

Otimizações de Convergência (Google Colab / GPU):
  - Carregamento direto dos pesos pré-treinados oficiais (yolo26n.pt com suporte a yolo26n.yaml).
  - Desativação de aumentações destrutivas para objetos pequenos (Safety-Shoes / Hardhat):
    erasing=0.0 (evita apagar botas/capacetes aleatoriamente) e auto_augment=None.
  - Agendador de taxa de aprendizado cosseno (cos_lr=True) e refinamento espacial (close_mosaic=20).

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
    """Aguarda confirmação do usuário ao final de execuções interativas em terminal."""
    if forcar_sem_pausa:
        return
    try:
        if houve_erro or (sys.stdin and sys.stdin.isatty()):
            print("\n" + "-" * 80)
            input("  [PAUSA] Pressione ENTER para finalizar...")
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
    print("  -> Instale os pacotes necessários executando:")
    print("     pip install ultralytics torch torchvision pyyaml opencv-python")
    print("!" * 80)
    pausar_se_interativo(forcar_sem_pausa=False, houve_erro=True)
    raise SystemExit(1) from exc_imp


def validar_e_corrigir_caminho_yaml(caminho_yaml, pasta_atual):
    """
    Lê o manifesto data.yaml e verifica se a chave 'path' aponta para um diretório válido
    no ambiente de execução atual (Windows local ou Linux no Google Colab),
    atualizando-o automaticamente caso necessário.
    """
    with open(caminho_yaml, "r", encoding="utf-8") as f:
        dados_yaml = yaml.safe_load(f) or {}

    caminho_configurado = dados_yaml.get("path", "")
    if not caminho_configurado or not os.path.exists(caminho_configurado):
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
                print(f"  [*] Atualizando caminho raiz em '{os.path.basename(caminho_yaml)}' para:\n      {novo_path}")
                dados_yaml["path"] = novo_path
                with open(caminho_yaml, "w", encoding="utf-8") as fw:
                    yaml.dump(dados_yaml, fw, sort_keys=False, allow_unicode=True)
                break

    return dados_yaml


def resolver_dispositivo(dispositivo):
    """
    Detecta a disponibilidade de acelerador gráfico CUDA e seleciona o dispositivo
    de processamento apropriado (GPU ou CPU).
    """
    cuda_ativo = torch.cuda.is_available()
    if dispositivo is None or str(dispositivo).strip() == "":
        return (0 if cuda_ativo else "cpu"), cuda_ativo

    disp_str = str(dispositivo).strip().lower()
    if disp_str != "cpu" and not cuda_ativo:
        print(f"  [!] Dispositivo '{dispositivo}' solicitado sem suporte CUDA ativo. Alternando para 'cpu'.")
        return "cpu", False

    return dispositivo, cuda_ativo


def treinar_modelo(
    caminho_yaml,
    epocas=100,
    tamanho_imagem=800,
    tamanho_lote=16,
    otimizador="auto",
    taxa_aprendizado=0.01,
    fator_lr_final=0.01,
    momento=0.937,
    decaimento_peso=0.0005,
    mosaico=1.0,
    fechar_mosaico=20,
    copiar_colar=0.3,
    mistura_imagens=0.1,
    apagamento_aleatorio=0.0,
    cosseno_lr=True,
    paciencia=30,
    dispositivo=None,
    pasta_saida=None,
    nome_experimento="ppe_detection_model_v2",
    apenas_validar=False,
):
    """
    Configura e executa o treinamento supervisionado do modelo YOLOv26n
    conforme os hiperparâmetros estabelecidos na metodologia experimental.
    """
    pasta_atual = os.path.dirname(os.path.abspath(__file__))

    if pasta_saida is None:
        pasta_saida = os.path.abspath(os.path.join(pasta_atual, "../models"))

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

    dados_yaml = validar_e_corrigir_caminho_yaml(caminho_yaml, pasta_atual)
    pasta_banco = dados_yaml.get("path", "")

    n_tr = len(glob.glob(os.path.join(pasta_banco, "images", "train", "*.*"))) if os.path.isdir(pasta_banco) else 0
    n_va = len(glob.glob(os.path.join(pasta_banco, "images", "val", "*.*"))) if os.path.isdir(pasta_banco) else 0
    n_te = len(glob.glob(os.path.join(pasta_banco, "images", "test", "*.*"))) if os.path.isdir(pasta_banco) else 0

    dispositivo_final, cuda_disponivel = resolver_dispositivo(dispositivo)

    print("\n" + "=" * 84)
    print("  CONFIGURAÇÃO E EXECUÇÃO DO TREINAMENTO SUPERVISIONADO (YOLOv26n)")
    print("=" * 84)
    print(f"  -> Arquitetura : {arquivo_arquitetura}")
    print(f"  -> Pesos Base  : {pesos_iniciais}")
    print(f"  -> Manifesto   : {caminho_yaml}")
    print(f"  -> Pasta Dados : {pasta_banco}")
    print(f"  -> Imagens     : {n_tr + n_va + n_te} totais (Treino: {n_tr} | Val: {n_va} | Teste: {n_te})")
    print(f"  -> Dispositivo : {dispositivo_final} (GPU CUDA ativa: {cuda_disponivel})")
    print(f"  -> Experimento : {nome_experimento} (em {pasta_saida})")
    print(f"  -> Parâmetros  : epochs={epocas}, batch={tamanho_lote}, imgsz={tamanho_imagem}, optimizer={otimizador}, cos_lr={cosseno_lr}, copy_paste={copiar_colar}, mixup={mistura_imagens}, erasing={apagamento_aleatorio}, patience={paciencia}")
    print("-" * 84 + "\n")

    if (n_tr + n_va) == 0:
        raise RuntimeError(
            f"Nenhuma imagem de treino ou validação localizada em '{pasta_banco}'.\n"
            "Execute previamente: python gerenciar_bases.py --action prepare --apply-clahe"
        )

    # Carrega o checkpoint pré-treinado preservando ckpt_path e a topologia oficial yolo26n.yaml
    modelo = YOLO(pesos_iniciais)
    modelo.info(detailed=False, verbose=True)

    if apenas_validar:
        print("\n" + "=" * 84)
        print("  [OK] VALIDAÇÃO PRÉ-TREINO (DRY-RUN) CONCLUÍDA COM SUCESSO")
        print("=" * 84)
        print(f"  -> Pasta do banco : {pasta_banco}")
        print(f"  -> Partições      : Treino={n_tr} | Validação={n_va} | Teste={n_te} (Total={n_tr + n_va + n_te} imagens)")
        print(f"  -> Total classes  : {dados_yaml.get('nc')}")
        print(f"  -> Classes        : {dados_yaml.get('names')}")
        print("=" * 84 + "\n")
        return modelo

    resultados = modelo.train(
        data=caminho_yaml,
        epochs=epocas,
        imgsz=tamanho_imagem,
        batch=tamanho_lote,
        optimizer=otimizador,
        lr0=taxa_aprendizado,
        lrf=fator_lr_final,
        momentum=momento,
        weight_decay=decaimento_peso,
        cos_lr=cosseno_lr,
        mosaic=mosaico,
        close_mosaic=fechar_mosaico,
        copy_paste=copiar_colar,
        mixup=mistura_imagens,
        erasing=apagamento_aleatorio,
        auto_augment=None,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        patience=paciencia,
        save=True,
        project=pasta_saida,
        name=nome_experimento,
        exist_ok=True,
        device=dispositivo_final,
    )

    caminho_melhor_peso = os.path.join(pasta_saida, nome_experimento, "weights", "best.pt")
    print(f"\n[OK] Treinamento concluído. Pesos otimizados salvos em: {caminho_melhor_peso}")
    return resultados


train_model = treinar_modelo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento supervisionado do modelo YOLOv26n para detecção de EPIs")
    parser.add_argument("--data", type=str, default="", help="Caminho para o arquivo data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Quantidade total de épocas")
    parser.add_argument("--imgsz", type=int, default=800, help="Dimensão espacial da imagem (padrão: 800 para objetos pequenos; use 640 para baseline)")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do lote por iteração")
    parser.add_argument("--optimizer", type=str, default="auto", help="Otimizador ('auto', 'MuSGD', 'AdamW' ou 'SGD')")
    parser.add_argument("--lr0", type=float, default=0.01, help="Taxa de aprendizado inicial")
    parser.add_argument("--lrf", type=float, default=0.01, help="Fração final da taxa de aprendizado")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momento do otimizador")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Decaimento de peso (regularização L2)")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Probabilidade da aumentação Mosaic (0.0 a 1.0)")
    parser.add_argument("--close-mosaic", type=int, default=20, help="Épocas finais com Mosaic desativado para refinamento")
    parser.add_argument("--copy-paste", type=float, default=0.3, help="Probabilidade de Copy-Paste (padrão: 0.3 para reforçar objetos pequenos como Safety-Shoes)")
    parser.add_argument("--mixup", type=float, default=0.1, help="Probabilidade de MixUp (padrão: 0.1)")
    parser.add_argument("--erasing", type=float, default=0.0, help="Probabilidade de apagamento aleatório (padrão: 0.0 para preservar Safety-Shoes)")
    parser.add_argument("--cos-lr", action="store_true", default=True, help="Ativa agendador cosseno de taxa de aprendizado (ativo por padrão)")
    parser.add_argument("--patience", type=int, default=30, help="Limite de épocas sem melhoria para parada antecipada")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo de execução ('0' para GPU ou 'cpu')")
    parser.add_argument("--project", type=str, default=None, help="Diretório de saída dos artefatos de treinamento")
    parser.add_argument("--name", type=str, default="ppe_detection_model_v2", help="Identificador do experimento (padrão: ppe_detection_model_v2 para preservar o baseline)")
    parser.add_argument("--dry-run", action="store_true", help="Valida a arquitetura e o dataset sem iniciar o treinamento")
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

        dataset_unified_dir = os.path.abspath(os.path.join(pasta_script, "../datasets/processed/unified_tcc"))
        if not os.path.isdir(os.path.join(dataset_unified_dir, "images", "train")):
            print("  [*] Dataset consolidado 'unified_tcc' não localizado. Iniciando pipeline de preparação...")
            from preparar_dados import build_unified_dataset
            caminho_encontrado = build_unified_dataset(use_clahe=True, clip_limit=1.5, target_resolution=args.imgsz)

        if caminho_encontrado and os.path.exists(caminho_encontrado):
            treinar_modelo(
                caminho_yaml=caminho_encontrado,
                epocas=args.epochs,
                tamanho_imagem=args.imgsz,
                tamanho_lote=args.batch,
                otimizador=args.optimizer,
                taxa_aprendizado=args.lr0,
                fator_lr_final=args.lrf,
                momento=args.momentum,
                decaimento_peso=args.weight_decay,
                mosaico=args.mosaic,
                fechar_mosaico=args.close_mosaic,
                copiar_colar=args.copy_paste,
                mistura_imagens=args.mixup,
                apagamento_aleatorio=args.erasing,
                cosseno_lr=args.cos_lr,
                paciencia=args.patience,
                dispositivo=args.device,
                pasta_saida=args.project,
                nome_experimento=args.name,
                apenas_validar=args.dry_run,
            )
        else:
            erro_ocorreu = True
            print("\n" + "!" * 80)
            print("  [ERRO] Manifesto 'data.yaml' não localizado nos diretórios verificados:")
            for p in caminhos_possiveis:
                if p:
                    print(f"    - {p}")
            print("\n  Execute previamente o gerenciador de bases:")
            print('    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/gerenciar_bases.py" --action all --apply-clahe')
            print("!" * 80)
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 80)
        print("  [ERRO EM treinar_modelo.py]")
        print("!" * 80)
        print(f"  -> Tipo     : {type(exc).__name__}")
        print(f"  -> Mensagem : {exc}")
        traceback.print_exc()
        print("!" * 80)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
