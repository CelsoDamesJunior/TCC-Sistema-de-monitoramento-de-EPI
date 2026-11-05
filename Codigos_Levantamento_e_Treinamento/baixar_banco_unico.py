# -*- coding: utf-8 -*-
"""
Módulo: baixar_banco_unico.py
Script utilitário dedicado para baixar o pacote oficial 'construction-ppe.zip' (170,15 MB)
diretamente do repositório oficial da Ultralytics caso o projeto seja clonado em outro
computador (já que o arquivo .zip específico excede o limite de 100 MB do GitHub).

Uso no terminal:
    python Projeto_EPI/src/baixar_banco_unico.py
"""

import os
import zipfile
import urllib.request
import argparse

CONSTRUCTION_PPE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"


def baixar_arquivo_zip(extrair=True):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "datasets", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_dest = os.path.join(raw_dir, "construction-ppe.zip")
    extracted_dir = os.path.join(raw_dir, "construction-ppe")

    print("=" * 75)
    print("  DOWNLOAD DO PACOTE OFICIAL ULTRALYTICS CONSTRUCTION-PPE (.ZIP)")
    print("=" * 75)
    print(f"  -> URL Oficial : {CONSTRUCTION_PPE_URL}")
    print(f"  -> Destino     : {zip_dest}")
    print("-" * 75)

    def _progresso(blocos, tam_bloco, tam_total):
        if tam_total > 0:
            baixado_mb = (blocos * tam_bloco) / (1024 * 1024)
            total_mb = tam_total / (1024 * 1024)
            pct = min(100.0, (baixado_mb / total_mb) * 100.0)
            print(f"\r  [*] Baixando: {baixado_mb:6.1f} MB / {total_mb:6.1f} MB ({pct:5.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(CONSTRUCTION_PPE_URL, zip_dest, reporthook=_progresso)
    print(f"\n[OK] Arquivo 'construction-ppe.zip' baixado com sucesso em:\n     {zip_dest}")

    if extrair:
        print(f"[*] Extraindo conteúdo para: {extracted_dir} ...")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(extracted_dir)
        print(f"[OK] Extração concluída com sucesso!")
    print("=" * 75)
    return zip_dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa o arquivo construction-ppe.zip em outro computador")
    parser.add_argument("--apenas-zip", action="store_true", help="Apenas baixa o .zip sem descompactar novamente")
    args = parser.parse_args()
    baixar_arquivo_zip(extrair=not args.apenas_zip)
