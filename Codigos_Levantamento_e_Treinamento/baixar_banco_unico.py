# -*- coding: utf-8 -*-
"""
Módulo: baixar_banco_unico.py
Script utilitário para aquisição e validação da Base 1 (Ultralytics Construction-PPE),
composta por 1.416 imagens e 7.194 caixas delimitadoras válidas nas 4 classes do projeto.

Fonte Oficial de Distribuição:
  - GitHub Releases: https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/construction-ppe.zip

Uso no terminal:
    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/baixar_banco_unico.py"
"""

import os
import sys
import glob
import shutil
import zipfile
import traceback
import urllib.request
import argparse

CONSTRUCTION_PPE_URL = "https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/construction-ppe.zip"
URL_RELEASE_TCC = CONSTRUCTION_PPE_URL


def pausar_se_interativo(forcar_sem_pausa=False, houve_erro=False):
    """Aguarda confirmação do usuário ao final de execuções interativas em terminal."""
    if forcar_sem_pausa:
        return
    try:
        if houve_erro or (sys.stdin and sys.stdin.isatty()):
            print("\n" + "-" * 75)
            input("  [PAUSA] Pressione ENTER para finalizar...")
    except (EOFError, KeyboardInterrupt):
        pass


def corrigir_pasta_aninhada(extracted_dir):
    """Normaliza a estrutura de diretórios caso o arquivo compactado possua subpasta raiz duplicada."""
    subpasta_interna = os.path.join(extracted_dir, "construction-ppe")
    if os.path.isdir(subpasta_interna) and os.path.isdir(os.path.join(subpasta_interna, "images")):
        for item in os.listdir(subpasta_interna):
            origem = os.path.join(subpasta_interna, item)
            destino = os.path.join(extracted_dir, item)
            if os.path.exists(destino):
                if os.path.isdir(destino):
                    shutil.rmtree(destino)
                else:
                    os.remove(destino)
            shutil.move(origem, destino)
        shutil.rmtree(subpasta_interna, ignore_errors=True)


def contar_imagens_extraidas(extracted_dir):
    """Retorna a contagem (total, train, val, test) de imagens no diretório extraído."""
    if not os.path.isdir(extracted_dir):
        return 0, 0, 0, 0
    corrigir_pasta_aninhada(extracted_dir)
    tr = len(glob.glob(os.path.join(extracted_dir, "images", "train", "*.*")))
    va = len(glob.glob(os.path.join(extracted_dir, "images", "val", "*.*")))
    te = len(glob.glob(os.path.join(extracted_dir, "images", "test", "*.*")))
    return (tr + va + te), tr, va, te


def tentar_copiar_de_pasta_local(extracted_dir, zip_dest):
    """Verifica a existência de cache local do pacote compactado ou da estrutura de diretórios no workspace."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raiz_ws = os.path.dirname(base_dir)
    candidato_dir = os.path.join(raiz_ws, "Projeto_EPI", "datasets", "raw", "construction-ppe")
    if os.path.isdir(candidato_dir):
        tot, _, _, _ = contar_imagens_extraidas(candidato_dir)
        if tot == 1416:
            print(f"  [*] Sincronizando Base 1 (1.416 imagens) a partir do diretório local: {candidato_dir}")
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
            shutil.copytree(candidato_dir, extracted_dir)
            return True
    return False


def baixar_arquivo_zip(extrair=True, forcar_download=False):
    """Realiza o download, a extração e a verificação de integridade da Base 1 (Construction-PPE)."""
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(pasta_script)
    raw_dir = os.path.join(base_dir, "datasets", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_dest = os.path.join(raw_dir, "construction-ppe.zip")
    extracted_dir = os.path.join(raw_dir, "construction-ppe")

    print("\n" + "=" * 75)
    print("  ETAPA 1: AQUISIÇÃO E VALIDAÇÃO DA BASE 1 (CONSTRUCTION-PPE - 1.416 IMAGENS)")
    print("=" * 75)
    print(f"  -> Fonte Oficial : {URL_RELEASE_TCC}")
    print(f"  -> Arquivo Alvo  : {zip_dest}")
    print(f"  -> Pasta Alvo    : {extracted_dir}")
    print("-" * 75)

    tot, tr, va, te = contar_imagens_extraidas(extracted_dir)
    if tot == 1416 and not forcar_download:
        print(f"  [OK] Base 1 validada localmente em:\n       {extracted_dir}")
        print(f"       Total: {tot} imagens (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("=" * 75)
        return zip_dest

    if not forcar_download and tentar_copiar_de_pasta_local(extracted_dir, zip_dest):
        tot, tr, va, te = contar_imagens_extraidas(extracted_dir)
        if tot == 1416:
            print(f"  [OK] Base 1 sincronizada ({tot} imagens).")
            print("=" * 75)
            return zip_dest

    if os.path.exists(zip_dest) and os.path.getsize(zip_dest) > 100 * 1024 * 1024 and not forcar_download:
        tam_mb = os.path.getsize(zip_dest) / (1024 * 1024)
        print(f"  [OK] Pacote local 'construction-ppe.zip' localizado ({tam_mb:.2f} MB).")
    else:
        def _progresso(blocos, tam_bloco, tam_total):
            if tam_total > 0:
                bx = (blocos * tam_bloco) / (1024 * 1024)
                tt = tam_total / (1024 * 1024)
                pct = min(100.0, (bx / tt) * 100.0)
                print(f"\r  [*] Baixando: {bx:6.1f} MB / {tt:6.1f} MB ({pct:5.1f}%)", end="", flush=True)

        try:
            print(f"  [*] Baixando pacote de:\n      {URL_RELEASE_TCC}")
            urllib.request.urlretrieve(URL_RELEASE_TCC, zip_dest, reporthook=_progresso)
            print(f"\n  [OK] Pacote salvo em: {zip_dest}")
        except Exception as e:
            print(f"\n  [!] Falha ao baixar de {URL_RELEASE_TCC} ({e}).")
            if not os.path.exists(zip_dest):
                raise RuntimeError(
                    f"Não foi possível obter o arquivo 'construction-ppe.zip' a partir de {URL_RELEASE_TCC}."
                ) from e

    if extrair:
        print(f"  [*] Extraindo conteúdo para: {extracted_dir} ...")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(extracted_dir)
        corrigir_pasta_aninhada(extracted_dir)
        tot, tr, va, te = contar_imagens_extraidas(extracted_dir)
        print(f"  [OK] Extração concluída. Imagens verificadas: {tot} (Treino: {tr} | Val: {va} | Teste: {te})")

    print("=" * 75)
    return zip_dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aquisição e validação da Base 1 (construction-ppe.zip)")
    parser.add_argument("--apenas-zip", action="store_true", help="Apenas garante o arquivo .zip sem extrair")
    parser.add_argument("--forcar-download", action="store_true", help="Força novo download mesmo se já existir localmente")
    parser.add_argument("--no-pause", action="store_true", help="Não pausa o terminal ao finalizar")
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        baixar_arquivo_zip(extrair=not args.apenas_zip, forcar_download=args.forcar_download)
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 75)
        print("  [ERRO EM baixar_banco_unico.py]")
        print("!" * 75)
        print(f"  -> Tipo     : {type(exc).__name__}")
        print(f"  -> Mensagem : {exc}")
        traceback.print_exc()
        print("!" * 75)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
