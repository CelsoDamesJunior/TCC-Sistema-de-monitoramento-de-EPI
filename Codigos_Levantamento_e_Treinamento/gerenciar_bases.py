# -*- coding: utf-8 -*-
"""
Módulo: gerenciar_bases.py
Orquestrador de Linha de Comando (CLI) para Gerenciamento e Auditoria das Bases de Dados de EPI.
Permite verificar o status detalhado (--action status), adquirir as bases (--action download),
construir o dataset consolidado unified_tcc com CLAHE (--action prepare --apply-clahe) ou
executar o pipeline completo (--action all --apply-clahe).

Uso no terminal:
    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/gerenciar_bases.py" --action status
    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/gerenciar_bases.py" --action all --apply-clahe
"""

import os
import sys
import glob
import traceback
import argparse

from preparar_dados import (
    download_single_dataset,
    ensure_roboflow_dataset,
    build_unified_dataset,
    corrigir_pasta_aninhada,
    pausar_se_interativo,
    CONSTRUCTION_PPE_URL,
    ROBOFLOW_RELEASE_URL,
    CONSTRUCTION_PPE_DOCS,
    ROBOFLOW_SAFETY_URL,
)
from baixar_roboflow import download_dataset as download_roboflow_dataset


def contar_caixas_pasta(labels_root, splits):
    """Contabiliza as caixas delimitadoras por classe (0..3) nas subpastas de anotações."""
    contagem = {0: 0, 1: 0, 2: 0, 3: 0}
    for sp in splits:
        pasta_sp = os.path.join(labels_root, sp)
        if not os.path.isdir(pasta_sp):
            continue
        for arq in glob.glob(os.path.join(pasta_sp, "*.txt")):
            try:
                with open(arq, "r", encoding="utf-8") as f:
                    for linha in f:
                        partes = linha.strip().split()
                        if len(partes) == 5:
                            cid = int(partes[0])
                            if cid in contagem:
                                contagem[cid] += 1
            except Exception:
                pass
    return contagem


def print_summary():
    """Exibe o relatório consolidado de integridade das bases de dados no padrão YOLO (.txt)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'datasets')

    print("\n" + "=" * 84)
    print("       RELATÓRIO DE STATUS DAS BASES DE DADOS DE EPI (PADRÃO YOLO TXT)")
    print("=" * 84)
    print(f"  Documentação Base 1  : {CONSTRUCTION_PPE_DOCS}")
    print(f"  Release Base 1 (.zip): {CONSTRUCTION_PPE_URL}")
    print(f"  Documentação Base 2  : {ROBOFLOW_SAFETY_URL}")
    print(f"  Release Base 2 (.zip): {ROBOFLOW_RELEASE_URL}")
    print("-" * 84)

    # 1. Base Principal Bruta (Ultralytics Construction-PPE)
    raw_ppe = os.path.join(datasets_dir, 'raw', 'construction-ppe')
    corrigir_pasta_aninhada(raw_ppe)
    if os.path.exists(raw_ppe):
        tr_r = len(glob.glob(os.path.join(raw_ppe, 'images', 'train', '*.*')))
        va_r = len(glob.glob(os.path.join(raw_ppe, 'images', 'val', '*.*')))
        te_r = len(glob.glob(os.path.join(raw_ppe, 'images', 'test', '*.*')))
        tot_r = tr_r + va_r + te_r
        status_b1 = "[OK]" if tot_r == 1416 else "[!]"
        print(f"{status_b1} Base 1 - Ultralytics Construction-PPE (Bruta YOLO TXT - 11 Classes):")
        print(f"     - Localização: {raw_ppe}")
        print(f"     - Imagens:     {tot_r} (Treino: {tr_r} | Val: {va_r} | Teste: {te_r}) [Esperado: 1.416]")
    else:
        print("[ ] Base 1 - Ultralytics Construction-PPE (Bruta): Não localizada.")

    # 2. Base Complementar Bruta (Roboflow Construction Site Safety)
    raw_rf = os.path.join(datasets_dir, 'raw', 'roboflow', 'Construction-Site-Safety-1')
    if os.path.exists(raw_rf):
        rf_tr = len(glob.glob(os.path.join(raw_rf, 'train', 'images', '*.*')))
        rf_va = len(glob.glob(os.path.join(raw_rf, 'valid', 'images', '*.*')))
        rf_te = len(glob.glob(os.path.join(raw_rf, 'test', 'images', '*.*')))
        tot_rf = rf_tr + rf_va + rf_te
        status_b2 = "[OK]" if tot_rf == 398 else "[!]"
        print(f"{status_b2} Base 2 - Roboflow Construction Site Safety (Bruta YOLO TXT - 15 Classes):")
        print(f"     - Localização: {raw_rf}")
        print(f"     - Imagens:     {tot_rf} (Treino: {rf_tr} | Val: {rf_va} | Teste: {rf_te}) [Esperado: 398]")
    else:
        print("[ ] Base 2 - Roboflow Construction Site Safety (Bruta): Não localizada.")

    # 3. Dataset Consolidado e Padronizado (unified_tcc - 4 Classes NR-06 / NR-18)
    unified_dir = os.path.join(datasets_dir, 'processed', 'unified_tcc')
    if os.path.exists(unified_dir):
        u_tr = len(glob.glob(os.path.join(unified_dir, 'images', 'train', '*.*')))
        u_va = len(glob.glob(os.path.join(unified_dir, 'images', 'val', '*.*')))
        u_te = len(glob.glob(os.path.join(unified_dir, 'images', 'test', '*.*')))
        tot_u = u_tr + u_va + u_te
        u_yaml = os.path.join(unified_dir, 'data.yaml')
        caixas = contar_caixas_pasta(os.path.join(unified_dir, 'labels'), ['train', 'val', 'test'])
        tot_caixas = sum(caixas.values())
        meta_str = "ATINGIDA (1.814 imgs / 7.774 caixas)" if (tot_u == 1814 and tot_caixas == 7774) else f"Parcial ({tot_u} imgs / {tot_caixas} caixas)"
        print("[OK] Dataset Consolidado (unified_tcc - 4 Classes NR-06 / NR-18):")
        print(f"     - Localização: {unified_dir}")
        print(f"     - Imagens:     {tot_u} (Treino: {u_tr} | Val: {u_va} | Teste: {u_te}) [Meta >= 1.500: {meta_str}]")
        print(f"     - Anotações:   {tot_caixas} caixas (0-Hardhat: {caixas[0]} | 1-Safety-Vest: {caixas[1]} | 2-Person: {caixas[2]} | 3-Safety-Shoes: {caixas[3]})")
        print(f"     - Manifesto:   {'Presente (' + u_yaml + ')' if os.path.exists(u_yaml) else 'Ausente'}")
    else:
        print("[ ] Dataset Consolidado (unified_tcc): Não construído.")
        print("     -> Execute 'python gerenciar_bases.py --action prepare --apply-clahe' para construí-lo.")

    print("=" * 84 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gerenciador Central das Bases de EPI (Construction-PPE + Roboflow)")
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        choices=["download", "prepare", "status", "all"],
        help="Ação a ser executada: download, prepare, status ou all",
    )
    parser.add_argument(
        "--apply-clahe",
        action="store_true",
        default=True,
        help="Aplica redimensionamento 640x640 (INTER_AREA) e CLAHE no canal L (CIE LAB) (ativo por padrão)",
    )
    parser.add_argument(
        "--no-clahe",
        action="store_true",
        help="Desativa a aplicação do filtro CLAHE",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=3.0,
        help="Limite de corte (clipLimit) do CLAHE (padrão: 3.0)",
    )
    parser.add_argument(
        "--api-key",
        "--api_key",
        dest="api_key",
        type=str,
        default="",
        help="Chave de API do Roboflow (opcional)",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Não pausa o terminal ao finalizar",
    )
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if args.action in ["download", "all"]:
            download_single_dataset(base_dir)
            ensure_roboflow_dataset(base_dir)
            if args.api_key:
                download_roboflow_dataset(api_key=args.api_key)
        if args.action in ["prepare", "all"]:
            usar_clahe = False if args.no_clahe else args.apply_clahe
            build_unified_dataset(use_clahe=usar_clahe, clip_limit=args.clip_limit)
        print_summary()
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 84)
        print("  [ERRO EM gerenciar_bases.py]")
        print("!" * 84)
        print(f"  -> Tipo     : {type(exc).__name__}")
        print(f"  -> Mensagem : {exc}")
        traceback.print_exc()
        print("!" * 84)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
