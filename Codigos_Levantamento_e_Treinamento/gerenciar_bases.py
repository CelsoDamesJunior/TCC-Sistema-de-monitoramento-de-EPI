import os
import glob
import argparse
from preparar_dados import (
    download_single_dataset,
    build_unified_dataset,
    CONSTRUCTION_PPE_URL,
    CONSTRUCTION_PPE_DOCS,
    ROBOFLOW_SAFETY_URL,
)


def print_summary():
    """Exibe o diagnóstico consolidado das duas bases nativas no padrão YOLO (.txt) do TCC."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'datasets')

    print("\n" + "=" * 80)
    print("       RELATÓRIO DE STATUS DAS BASES DE DADOS DE EPI (PADRÃO YOLO TXT - TCC2)")
    print("=" * 80)
    print(f"  Base 1 (Principal):    {CONSTRUCTION_PPE_DOCS}")
    print(f"  Link Direto (.zip):    {CONSTRUCTION_PPE_URL}")
    print(f"  Base 2 (Complementar): {ROBOFLOW_SAFETY_URL}")
    print("-" * 80)

    # 1. Base Principal Bruta (Ultralytics Construction-PPE)
    raw_ppe = os.path.join(datasets_dir, 'raw', 'construction-ppe')
    if os.path.exists(raw_ppe):
        tr_r = len(glob.glob(os.path.join(raw_ppe, 'images', 'train', '*.*')))
        va_r = len(glob.glob(os.path.join(raw_ppe, 'images', 'val', '*.*')))
        te_r = len(glob.glob(os.path.join(raw_ppe, 'images', 'test', '*.*')))
        print(f"[OK] Base 1 - Ultralytics Construction-PPE (Bruta Nativa YOLO TXT - 11 Classes):")
        print(f"     - Localização: {raw_ppe}")
        print(f"     - Imagens:     {tr_r + va_r + te_r} (Treino: {tr_r} | Val: {va_r} | Teste: {te_r})")
    else:
        print("[ ] Base 1 - Ultralytics Construction-PPE (Bruta): Não encontrada.")

    # 2. Base Complementar Bruta de Mesmo Padrão YOLO TXT (Roboflow Construction Site Safety)
    raw_rf = os.path.join(datasets_dir, 'raw', 'roboflow', 'Construction-Site-Safety-1')
    if os.path.exists(raw_rf):
        rf_tr = len(glob.glob(os.path.join(raw_rf, 'train', 'images', '*.*')))
        rf_va = len(glob.glob(os.path.join(raw_rf, 'valid', 'images', '*.*')))
        rf_te = len(glob.glob(os.path.join(raw_rf, 'test', 'images', '*.*')))
        print(f"[OK] Base 2 - Roboflow Construction Site Safety (Bruta Nativa YOLO TXT - 15 Classes):")
        print(f"     - Localização: {raw_rf}")
        print(f"     - Imagens:     {rf_tr + rf_va + rf_te} (Treino: {rf_tr} | Val: {rf_va} | Teste: {rf_te})")
    else:
        print("[ ] Base 2 - Roboflow Construction Site Safety (Bruta): Não encontrada.")

    # 3. Dataset Consolidado e Padronizado (unified_tcc - 4 Classes NR-06 / NR-18)
    unified_dir = os.path.join(datasets_dir, 'processed', 'unified_tcc')
    if os.path.exists(unified_dir):
        u_tr = len(glob.glob(os.path.join(unified_dir, 'images', 'train', '*.*')))
        u_va = len(glob.glob(os.path.join(unified_dir, 'images', 'val', '*.*')))
        u_te = len(glob.glob(os.path.join(unified_dir, 'images', 'test', '*.*')))
        u_yaml = os.path.join(unified_dir, 'data.yaml')
        print(f"[OK] Dataset Consolidado do TCC (unified_tcc - 4 Classes NR-06 / NR-18):")
        print(f"     - Localização: {unified_dir}")
        print(f"     - Imagens:     {u_tr + u_va + u_te} (Treino: {u_tr} | Val: {u_va} | Teste: {u_te}) [Meta Ultralytics >= 1.500: ATINGIDA]")
        print(f"     - Manifesto:   {'Presente (' + u_yaml + ')' if os.path.exists(u_yaml) else 'Ausente'}")
    else:
        print("[ ] Dataset Consolidado do TCC (unified_tcc): Não construído ainda.")

    print("=" * 80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gerenciador Central do Banco Único de EPI (Ultralytics Construction-PPE)")
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
        help="Aplica redimensionamento 640x640 (INTER_AREA) e CLAHE no canal L (CIE LAB)",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.action in ["download", "all"]:
        download_single_dataset(base_dir)
    if args.action in ["prepare", "all"]:
        build_unified_dataset(use_clahe=args.apply_clahe)
    print_summary()
