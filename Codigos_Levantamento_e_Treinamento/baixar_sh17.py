# -*- coding: utf-8 -*-
"""
Módulo: baixar_sh17.py
Script utilitário para aquisição e validação da Base 4 (SH17 Dataset - Human & PPE Detection),
composta pelo subconjunto filtrado de 313 imagens completas e 2.519 caixas delimitadoras válidas
nas 4 classes do projeto (Hardhat, Safety-Vest, Person, Safety-Shoes) nativas no formato YOLO (.txt).

Fontes Oficiais de Distribuição:
  - GitHub Releases (Principal): https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/sh17-ppe.zip
  - Espelho Hugging Face Hub (Contingência): https://huggingface.co/datasets/fathansanum/SH-17-Dataset

Uso no terminal:
    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/baixar_sh17.py"
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

URL_RELEASE_SH17_TCC = "https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/sh17-ppe.zip"
REPOSITORIO_HF_SH17 = "fathansanum/SH-17-Dataset"
HF_SH17_REPO = REPOSITORIO_HF_SH17


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


def corrigir_pasta_sh17(diretorio_sh17):
    """Normaliza a estrutura de diretórios caso o pacote possua subpasta raiz duplicada."""
    if not os.path.isdir(diretorio_sh17):
        return
    subpasta = os.path.join(diretorio_sh17, "sh17-ppe")
    if os.path.isdir(subpasta) and os.path.isdir(os.path.join(subpasta, "train")):
        for item in os.listdir(subpasta):
            origem = os.path.join(subpasta, item)
            destino = os.path.join(diretorio_sh17, item)
            if os.path.exists(destino):
                if os.path.isdir(destino):
                    shutil.rmtree(destino)
                else:
                    os.remove(destino)
            shutil.move(origem, destino)
        shutil.rmtree(subpasta, ignore_errors=True)


def contar_imagens_sh17(diretorio_sh17):
    """Retorna a contagem (total, treino, validacao, teste) de imagens no diretório sh17-ppe."""
    if not os.path.isdir(diretorio_sh17):
        return 0, 0, 0, 0
    corrigir_pasta_sh17(diretorio_sh17)
    tr = len(glob.glob(os.path.join(diretorio_sh17, "train", "images", "*.*")))
    va = len(glob.glob(os.path.join(diretorio_sh17, "valid", "images", "*.*")))
    te = len(glob.glob(os.path.join(diretorio_sh17, "test", "images", "*.*")))
    return (tr + va + te), tr, va, te


def baixar_via_huggingface_sh17(pasta_destino, destino_zip=None):
    """
    Contingência automática que adquire o subconjunto completo de 313 imagens da base SH17
    diretamente do espelho Hugging Face Hub caso o arquivo .zip ainda não esteja disponível.
    """
    print(f"  [*] Baixando Base 4 (SH17 PPE) do espelho Hugging Face ({REPOSITORIO_HF_SH17})...")
    os.makedirs(pasta_destino, exist_ok=True)
    tarefas_download = []

    for sp in ["train", "valid", "test"]:
        os.makedirs(os.path.join(pasta_destino, sp, "images"), exist_ok=True)
        os.makedirs(os.path.join(pasta_destino, sp, "labels"), exist_ok=True)
        url_indice = f"https://huggingface.co/datasets/{REPOSITORIO_HF_SH17}/resolve/main/SH17-dataset-1/{sp}/_annotations.coco.json"
        dados_indice = json.loads(urllib.request.urlopen(url_indice).read().decode("utf-8"))
        mapa_cat = {c["id"]: c["name"] for c in dados_indice.get("categories", [])}
        mapa_img = {im["id"]: im for im in dados_indice.get("images", [])}
        anotacoes_img = {}
        for a in dados_indice.get("annotations", []):
            nome_c = mapa_cat.get(a["category_id"], "")
            if nome_c in {"10", "16", "0", "14"}:
                anotacoes_img.setdefault(a["image_id"], []).append((int(nome_c), a["bbox"]))

        for iid, lista_anot in anotacoes_img.items():
            classes_presentes = set(cid for cid, _ in lista_anot)
            if {0, 10, 16, 14}.issubset(classes_presentes):
                info_im = mapa_img[iid]
                nome_arq = info_im["file_name"]
                larg_im = float(info_im.get("width", 640))
                alt_im = float(info_im.get("height", 640))
                nome_base = os.path.splitext(nome_arq)[0]
                caminho_lbl = os.path.join(pasta_destino, sp, "labels", f"{nome_base}.txt")
                linhas_txt = []
                for cid, bbox in lista_anot:
                    x, y, bw, bh = [float(v) for v in bbox]
                    xc = max(0.0, min(1.0, (x + bw / 2.0) / larg_im))
                    yc = max(0.0, min(1.0, (y + bh / 2.0) / alt_im))
                    wn = max(0.0, min(1.0, bw / larg_im))
                    hn = max(0.0, min(1.0, bh / alt_im))
                    if wn > 0.001 and hn > 0.001:
                        linhas_txt.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                if linhas_txt:
                    with open(caminho_lbl, "w", encoding="utf-8") as fw:
                        fw.writelines(linhas_txt)
                    dst_img = os.path.join(pasta_destino, sp, "images", nome_arq)
                    url_img = f"https://huggingface.co/datasets/{REPOSITORIO_HF_SH17}/resolve/main/SH17-dataset-1/{sp}/{nome_arq}"
                    tarefas_download.append((url_img, dst_img))

    concluidos = [0]
    total_arq = len(tarefas_download)

    def _baixar_item(item):
        url_img, dst_img = item
        if not os.path.exists(dst_img) or os.path.getsize(dst_img) < 1000:
            urllib.request.urlretrieve(url_img, dst_img)
        concluidos[0] += 1
        if concluidos[0] % 25 == 0 or concluidos[0] == total_arq:
            pct = (concluidos[0] / max(1, total_arq)) * 100.0
            print(f"\r  [*] Progresso do download: {concluidos[0]}/{total_arq} imagens ({pct:5.1f}%)", end="", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(_baixar_item, tarefas_download))
    print()

    if destino_zip:
        os.makedirs(os.path.dirname(destino_zip), exist_ok=True)
        with zipfile.ZipFile(destino_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for raiz, _, lista_arquivos in os.walk(pasta_destino):
                for nome_arq in sorted(lista_arquivos):
                    caminho_completo = os.path.join(raiz, nome_arq)
                    caminho_rel = os.path.relpath(caminho_completo, pasta_destino).replace("\\", "/")
                    zf.write(caminho_completo, caminho_rel)
        print(f"  [OK] Pacote compactado salvo em: {destino_zip} ({os.path.getsize(destino_zip) / (1024*1024):.2f} MB)")


def baixar_base_sh17(diretorio_saida=None, url_release=URL_RELEASE_SH17_TCC, forcar_download=False):
    """
    Valida a disponibilidade local da Base 4 (SH17 PPE Dataset - 313 imagens completas em YOLO .txt)
    ou realiza sua aquisição via GitHub Releases v1.0.0 (com contingência via Hugging Face Hub).
    """
    diretorio_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not diretorio_saida:
        diretorio_saida = os.path.join(diretorio_base, "datasets", "raw")

    os.makedirs(diretorio_saida, exist_ok=True)
    pasta_destino = os.path.join(diretorio_saida, "sh17-ppe")
    destino_zip = os.path.join(diretorio_saida, "sh17-ppe.zip")
    zip_pacote_release = os.path.join(diretorio_base, "Pacote_Release_GitHub", "sh17-ppe.zip")

    print("\n" + "=" * 75)
    print("  ETAPA 4: AQUISIÇÃO E VALIDAÇÃO DA BASE 4 (SH17 PPE - 313 IMAGENS YOLO)")
    print("=" * 75)
    print(f"  -> Fonte Oficial : {url_release}")
    print(f"  -> Arquivo Alvo  : {destino_zip}")
    print(f"  -> Pasta Alvo    : {pasta_destino}")
    print("-" * 75)

    tot, tr, va, te = contar_imagens_sh17(pasta_destino)
    if tot >= 313 and not forcar_download:
        print("  [OK] Base 4 (SH17 PPE) validada localmente.")
        print(f"       Total: {tot} imagens (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("=" * 75)
        return pasta_destino

    for zip_candidato in [destino_zip, zip_pacote_release]:
        if os.path.exists(zip_candidato) and os.path.getsize(zip_candidato) > 5 * 1024 * 1024 and not forcar_download:
            print(f"  [*] Extraindo Base 4 a partir do pacote local: {zip_candidato}")
            with zipfile.ZipFile(zip_candidato, "r") as zf:
                zf.extractall(pasta_destino)
            tot, tr, va, te = contar_imagens_sh17(pasta_destino)
            if tot >= 313:
                print(f"  [OK] Base 4 extraída ({tot} imagens: Treino={tr} | Validação={va} | Teste={te}).")
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
        tot, tr, va, te = contar_imagens_sh17(pasta_destino)
        print(f"  [OK] Extração concluída. Imagens verificadas: {tot} (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("=" * 75)
        return pasta_destino
    except Exception as e_rel:
        print(f"\n  [!] Download via GitHub Releases não concluído ({e_rel}).")
        print("  [*] Acionando contingência automática via espelho Hugging Face Hub...")

    baixar_via_huggingface_sh17(pasta_destino, destino_zip=destino_zip)
    if os.path.exists(destino_zip):
        os.makedirs(os.path.dirname(zip_pacote_release), exist_ok=True)
        shutil.copy2(destino_zip, zip_pacote_release)

    tot, tr, va, te = contar_imagens_sh17(pasta_destino)
    print(f"  [OK] Base 4 pronta ({tot} imagens: Treino={tr} | Validação={va} | Teste={te}).")
    print("=" * 75)
    return pasta_destino


# Alias de compatibilidade
download_sh17_dataset = baixar_base_sh17


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aquisição e validação da Base 4 (SH17 PPE Dataset - sh17-ppe.zip)")
    parser.add_argument("--url-release", "--release-url", dest="url_release", type=str, default=URL_RELEASE_SH17_TCC, help="URL do arquivo sh17-ppe.zip no GitHub Release")
    parser.add_argument("--forcar-download", action="store_true", help="Força novo download mesmo se já existir localmente")
    parser.add_argument("--no-pause", action="store_true", help="Não pausa o terminal ao finalizar")
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        baixar_base_sh17(url_release=args.url_release, forcar_download=args.forcar_download)
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 75)
        print("  [ERRO EM baixar_sh17.py]")
        print("!" * 75)
        print(f"  -> Tipo     : {type(exc).__name__}")
        print(f"  -> Mensagem : {exc}")
        traceback.print_exc()
        print("!" * 75)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
