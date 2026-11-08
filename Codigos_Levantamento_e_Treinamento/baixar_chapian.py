# -*- coding: utf-8 -*-
"""
Módulo: baixar_chapian.py
Script utilitário para aquisição e validação da Base 3 (Chapian Construction Site PPE Detection),
composta por 200 imagens de canteiros de obras e 1.546 caixas delimitadoras válidas nas 4 classes
do projeto (Hardhat, Safety-Vest, Person, Safety-Shoes) nativas no formato YOLO (.txt).

Fontes Oficiais de Distribuição:
  - GitHub Releases (Principal): https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/chapian-ppe.zip
  - Espelho Hugging Face Hub (Contingência automática): https://huggingface.co/datasets/Chapian/PPE_detection

Uso no terminal:
    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/baixar_chapian.py"
"""

import os
import sys
import json
import glob
import shutil
import zipfile
import traceback
import urllib.request
import concurrent.futures
import argparse

URL_RELEASE_CHAPIAN_TCC = "https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/chapian-ppe.zip"
REPOSITORIO_HF_CHAPIAN = "Chapian/PPE_detection"
HF_CHAPIAN_REPO = REPOSITORIO_HF_CHAPIAN


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


def corrigir_particoes_chapian(diretorio_chapian):
    """
    Garante que o diretório da Base 3 possua as 3 partições padronizadas (train: 100, valid: 50, test: 50).
    Caso o pacote tenha subpasta aninhada ou venha do espelho bruto (100 train / 100 test),
    organiza automaticamente as pastas (50 valid / 50 test).
    """
    if not os.path.isdir(diretorio_chapian):
        return
    subpasta = os.path.join(diretorio_chapian, "chapian-ppe")
    if os.path.isdir(subpasta) and os.path.isdir(os.path.join(subpasta, "train")):
        for item in os.listdir(subpasta):
            origem = os.path.join(subpasta, item)
            destino = os.path.join(diretorio_chapian, item)
            if os.path.exists(destino):
                if os.path.isdir(destino):
                    shutil.rmtree(destino)
                else:
                    os.remove(destino)
            shutil.move(origem, destino)
        shutil.rmtree(subpasta, ignore_errors=True)

    pasta_valid_img = os.path.join(diretorio_chapian, "valid", "images")
    pasta_valid_lbl = os.path.join(diretorio_chapian, "valid", "labels")
    os.makedirs(pasta_valid_img, exist_ok=True)
    os.makedirs(pasta_valid_lbl, exist_ok=True)

    qtd_valid = len(glob.glob(os.path.join(pasta_valid_img, "*.*")))
    imagens_teste = sorted(glob.glob(os.path.join(diretorio_chapian, "test", "images", "*.*")))
    if qtd_valid == 0 and len(imagens_teste) == 100:
        for caminho_img in imagens_teste[:50]:
            nome_base = os.path.splitext(os.path.basename(caminho_img))[0]
            extensao = os.path.splitext(caminho_img)[1]
            caminho_lbl = os.path.join(diretorio_chapian, "test", "labels", f"{nome_base}.txt")
            shutil.move(caminho_img, os.path.join(pasta_valid_img, f"{nome_base}{extensao}"))
            if os.path.exists(caminho_lbl):
                shutil.move(caminho_lbl, os.path.join(pasta_valid_lbl, f"{nome_base}.txt"))


def normalizar_labels_5_colunas_chapian(diretorio_chapian):
    """Converte eventuais linhas poligonais (> 5 colunas) em caixas delimitadoras YOLO de 5 colunas."""
    for sp in ["train", "valid", "test"]:
        pasta_lbl = os.path.join(diretorio_chapian, sp, "labels")
        if not os.path.isdir(pasta_lbl):
            continue
        for arq_lbl in glob.glob(os.path.join(pasta_lbl, "*.txt")):
            with open(arq_lbl, "r", encoding="utf-8") as f:
                linhas = f.readlines()
            novas = []
            alterado = False
            for linha in linhas:
                partes = linha.strip().split()
                if not partes:
                    continue
                if len(partes) > 5:
                    cid = int(partes[0])
                    vals = [float(v) for v in partes[1:]]
                    xs = vals[0::2]
                    ys = vals[1::2]
                    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
                    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
                    xc, yc, bw, bh = (x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0
                    novas.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    alterado = True
                elif len(partes) == 5:
                    novas.append(linha if linha.endswith("\n") else linha + "\n")
            if alterado:
                with open(arq_lbl, "w", encoding="utf-8") as fw:
                    fw.writelines(novas)


def contar_imagens_chapian(diretorio_chapian):
    """Retorna a contagem (total, treino, validacao, teste) de imagens no diretório chapian-ppe."""
    if not os.path.isdir(diretorio_chapian):
        return 0, 0, 0, 0
    corrigir_particoes_chapian(diretorio_chapian)
    tr = len(glob.glob(os.path.join(diretorio_chapian, "train", "images", "*.*")))
    va = len(glob.glob(os.path.join(diretorio_chapian, "valid", "images", "*.*")))
    te = len(glob.glob(os.path.join(diretorio_chapian, "test", "images", "*.*")))
    return (tr + va + te), tr, va, te


def baixar_via_huggingface(pasta_destino, destino_zip=None):
    """Baixa os arquivos nativos YOLO (.txt e .jpg) diretamente do espelho no Hugging Face Hub."""
    print(f"  [*] Baixando Base 3 diretamente do espelho Hugging Face ({REPOSITORIO_HF_CHAPIAN})...")
    url_api = f"https://huggingface.co/api/datasets/{REPOSITORIO_HF_CHAPIAN}"
    dados = json.loads(urllib.request.urlopen(url_api).read().decode("utf-8"))
    arquivos = [
        f["rfilename"]
        for f in dados.get("siblings", [])
        if not f["rfilename"].startswith(".") and not f["rfilename"].endswith((".pt", ".cache"))
    ]

    os.makedirs(pasta_destino, exist_ok=True)
    concluidos = [0]
    total_arq = len(arquivos)

    def _baixar_item(arquivo_relativo):
        caminho_final = os.path.join(pasta_destino, *arquivo_relativo.split("/"))
        os.makedirs(os.path.dirname(caminho_final), exist_ok=True)
        if not os.path.exists(caminho_final) or os.path.getsize(caminho_final) == 0:
            url = f"https://huggingface.co/datasets/{REPOSITORIO_HF_CHAPIAN}/resolve/main/{arquivo_relativo}"
            urllib.request.urlretrieve(url, caminho_final)
        concluidos[0] += 1
        if concluidos[0] % 25 == 0 or concluidos[0] == total_arq:
            pct = (concluidos[0] / max(1, total_arq)) * 100.0
            print(f"\r  [*] Progresso do download: {concluidos[0]}/{total_arq} arquivos ({pct:5.1f}%)", end="", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(_baixar_item, arquivos))
    print()

    corrigir_particoes_chapian(pasta_destino)
    normalizar_labels_5_colunas_chapian(pasta_destino)

    if destino_zip:
        os.makedirs(os.path.dirname(destino_zip), exist_ok=True)
        with zipfile.ZipFile(destino_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for raiz, _, lista_arquivos in os.walk(pasta_destino):
                for nome_arq in sorted(lista_arquivos):
                    if nome_arq.endswith((".pt", ".cache")):
                        continue
                    caminho_completo = os.path.join(raiz, nome_arq)
                    caminho_rel = os.path.relpath(caminho_completo, pasta_destino).replace("\\", "/")
                    zf.write(caminho_completo, caminho_rel)
        print(f"  [OK] Pacote compactado salvo em: {destino_zip} ({os.path.getsize(destino_zip) / (1024*1024):.2f} MB)")


def baixar_base_chapian(diretorio_saida=None, url_release=URL_RELEASE_CHAPIAN_TCC, forcar_download=False):
    """
    Valida a disponibilidade local da Base 3 (Chapian PPE Detection - 200 imagens, 4 classes YOLO .txt)
    ou realiza sua aquisição via GitHub Releases v1.0.0 (com contingência via Hugging Face Hub).
    """
    diretorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not diretorio_saida:
        diretorio_saida = os.path.join(diretorio_base, "datasets", "raw")

    os.makedirs(diretorio_saida, exist_ok=True)
    pasta_destino = os.path.join(diretorio_saida, "chapian-ppe")
    destino_zip = os.path.join(diretorio_saida, "chapian-ppe.zip")
    zip_pacote_release = os.path.join(diretorio_base, "Pacote_Release_GitHub", "chapian-ppe.zip")

    print("\n" + "=" * 75)
    print("  ETAPA 3: AQUISIÇÃO E VALIDAÇÃO DA BASE 3 (CHAPIAN PPE - 200 IMAGENS)")
    print("=" * 75)
    print(f"  -> Fonte Oficial : {url_release}")
    print(f"  -> Arquivo Alvo  : {destino_zip}")
    print(f"  -> Pasta Alvo    : {pasta_destino}")
    print("-" * 75)

    tot, tr, va, te = contar_imagens_chapian(pasta_destino)
    if tot >= 200 and not forcar_download:
        print("  [OK] Base 3 (Chapian PPE) validada localmente.")
        print(f"       Total: {tot} imagens (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("=" * 75)
        return pasta_destino

    for zip_candidato in [destino_zip, zip_pacote_release]:
        if os.path.exists(zip_candidato) and os.path.getsize(zip_candidato) > 5 * 1024 * 1024 and not forcar_download:
            print(f"  [*] Extraindo Base 3 a partir do pacote local: {zip_candidato}")
            with zipfile.ZipFile(zip_candidato, "r") as zf:
                zf.extractall(pasta_destino)
            tot, tr, va, te = contar_imagens_chapian(pasta_destino)
            if tot >= 200:
                print(f"  [OK] Base 3 extraída ({tot} imagens: Treino={tr} | Validação={va} | Teste={te}).")
                print("=" * 75)
                return pasta_destino

    print("  [*] Baixando pacote da Release oficial do GitHub...")
    try:
        def _progresso(blocos, tam_bloco, tam_total):
            if tam_total > 0:
                bx = (blocos * tam_bloco) / (1024 * 1024)
                tt = tam_total / (1024 * 1024)
                pct = min(100.0, (bx / tt) * 100.0)
                print(f"\r  [*] Baixando: {bx:6.1f} MB / {tt:6.1f} MB ({pct:5.1f}%)", end="", flush=True)

        urllib.request.urlretrieve(url_release, destino_zip, reporthook=_progresso)
        print(f"\n  [OK] Pacote salvo em: {destino_zip}")
        print(f"  [*] Extraindo conteúdo para: {pasta_destino} ...")
        with zipfile.ZipFile(destino_zip, "r") as zf:
            zf.extractall(pasta_destino)
        tot, tr, va, te = contar_imagens_chapian(pasta_destino)
        print(f"  [OK] Extração concluída. Imagens verificadas: {tot} (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("=" * 75)
        return pasta_destino
    except Exception as e_rel:
        print(f"\n  [!] Download via GitHub Releases não concluído ({e_rel}).")
        print("  [*] Acionando contingência automática via espelho Hugging Face Hub...")

    baixar_via_huggingface(pasta_destino, destino_zip=destino_zip)
    if os.path.exists(destino_zip):
        os.makedirs(os.path.dirname(zip_pacote_release), exist_ok=True)
        shutil.copy2(destino_zip, zip_pacote_release)

    tot, tr, va, te = contar_imagens_chapian(pasta_destino)
    print(f"  [OK] Base 3 pronta ({tot} imagens: Treino={tr} | Validação={va} | Teste={te}).")
    print("=" * 75)
    return pasta_destino


# Alias de compatibilidade
download_chapian_dataset = baixar_base_chapian


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aquisição e validação da Base 3 (Chapian PPE Detection - chapian-ppe.zip)")
    parser.add_argument("--url-release", "--release-url", dest="url_release", type=str, default=URL_RELEASE_CHAPIAN_TCC, help="URL do arquivo chapian-ppe.zip no GitHub Release")
    parser.add_argument("--forcar-download", action="store_true", help="Força novo download mesmo se já existir localmente")
    parser.add_argument("--no-pause", action="store_true", help="Não pausa o terminal ao finalizar")
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        baixar_base_chapian(url_release=args.url_release, forcar_download=args.forcar_download)
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 75)
        print("  [ERRO EM baixar_chapian.py]")
        print("!" * 75)
        print(f"  -> Tipo     : {type(exc).__name__}")
        print(f"  -> Mensagem : {exc}")
        traceback.print_exc()
        print("!" * 75)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
