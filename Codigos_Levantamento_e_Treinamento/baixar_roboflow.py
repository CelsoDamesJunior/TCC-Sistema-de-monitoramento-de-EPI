# -*- coding: utf-8 -*-
"""
Módulo: baixar_roboflow.py
Script utilitário para aquisição e validação da Base 2 (Roboflow Construction PPE - ppe-fruwx),
composta por 4.460 imagens e 22.574 caixas delimitadoras válidas nas 4 classes do projeto
(16.051 caixas nas 2.639 imagens completas com Person) no formato nativo YOLO (.txt).

Fontes de Distribuição:
  - GitHub Releases: https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/roboflow-safety.zip
  - Espelho Hugging Face: https://huggingface.co/datasets/vikasaarna/roboflow-construction-ppe-fruwx
  - API Roboflow Universe (opcional mediante chave de API)

Uso no terminal:
    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/baixar_roboflow.py"
"""

import os
import sys
import glob
import shutil
import zipfile
import traceback
import urllib.request
import argparse

import json
import concurrent.futures

URL_RELEASE_ROBOFLOW_TCC = "https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/roboflow-safety.zip"
REPOSITORIO_HF_ROBOFLOW = "vikasaarna/roboflow-construction-ppe-fruwx"


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


def corrigir_pasta_roboflow(rf_dir):
    """Normaliza a estrutura de diretórios caso o arquivo compactado possua subpasta raiz duplicada."""
    if not os.path.isdir(rf_dir):
        return
    for nome_sub in ["Construction-Site-Safety-1", "data", "roboflow-safety"]:
        subpasta = os.path.join(rf_dir, nome_sub)
        if os.path.isdir(subpasta) and os.path.isdir(os.path.join(subpasta, "train")):
            for item in os.listdir(subpasta):
                origem = os.path.join(subpasta, item)
                destino = os.path.join(rf_dir, item)
                if os.path.exists(destino):
                    if os.path.isdir(destino):
                        shutil.rmtree(destino)
                    else:
                        os.remove(destino)
                shutil.move(origem, destino)
            shutil.rmtree(subpasta, ignore_errors=True)


def normalizar_labels_5_colunas_roboflow(rf_dir):
    """Converte eventuais linhas poligonais (> 5 colunas) em caixas delimitadoras YOLO de 5 colunas."""
    convertidas = 0
    for sp in ["train", "valid", "test"]:
        pasta_lbl = os.path.join(rf_dir, sp, "labels")
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
                    convertidas += 1
                elif len(partes) == 5:
                    novas.append(linha if linha.endswith("\n") else linha + "\n")
            if alterado:
                with open(arq_lbl, "w", encoding="utf-8") as fw:
                    fw.writelines(novas)
    return convertidas


def contar_imagens_roboflow(rf_dir):
    """Retorna a contagem (total, train, valid, test) de imagens no diretório Construction-Site-Safety-1."""
    if not os.path.isdir(rf_dir):
        return 0, 0, 0, 0
    corrigir_pasta_roboflow(rf_dir)
    tr = len(glob.glob(os.path.join(rf_dir, "train", "images", "*.*")))
    va = len(glob.glob(os.path.join(rf_dir, "valid", "images", "*.*")))
    te = len(glob.glob(os.path.join(rf_dir, "test", "images", "*.*")))
    return (tr + va + te), tr, va, te


def baixar_via_huggingface_roboflow(target_rf_dir, destino_zip=None):
    """Baixa a Base 2 (4.460 imagens YOLO .txt) diretamente do espelho no Hugging Face Hub."""
    print(f"  [*] Baixando Base 2 diretamente do espelho Hugging Face ({REPOSITORIO_HF_ROBOFLOW})...")
    url_api = f"https://huggingface.co/api/datasets/{REPOSITORIO_HF_ROBOFLOW}"
    dados = json.loads(urllib.request.urlopen(url_api).read().decode("utf-8"))
    arquivos = [
        f["rfilename"]
        for f in dados.get("siblings", [])
        if f["rfilename"].startswith("data/") and not os.path.basename(f["rfilename"]).startswith(".")
    ]

    os.makedirs(target_rf_dir, exist_ok=True)
    concluidos = [0]
    total_arq = len(arquivos)

    def _baixar_item(arquivo_relativo):
        rel_limpo = arquivo_relativo[len("data/"):] if arquivo_relativo.startswith("data/") else arquivo_relativo
        caminho_final = os.path.join(target_rf_dir, *rel_limpo.split("/"))
        os.makedirs(os.path.dirname(caminho_final), exist_ok=True)
        if not os.path.exists(caminho_final) or os.path.getsize(caminho_final) == 0:
            url = f"https://huggingface.co/datasets/{REPOSITORIO_HF_ROBOFLOW}/resolve/main/{arquivo_relativo}"
            urllib.request.urlretrieve(url, caminho_final)
        concluidos[0] += 1
        if concluidos[0] % 200 == 0 or concluidos[0] == total_arq:
            pct = (concluidos[0] / max(1, total_arq)) * 100.0
            print(f"\r  [*] Progresso do download: {concluidos[0]}/{total_arq} arquivos ({pct:5.1f}%)", end="", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        list(executor.map(_baixar_item, arquivos))
    print()

    corrigir_pasta_roboflow(target_rf_dir)
    normalizar_labels_5_colunas_roboflow(target_rf_dir)

    if destino_zip:
        os.makedirs(os.path.dirname(destino_zip), exist_ok=True)
        with zipfile.ZipFile(destino_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for raiz, _, lista_arquivos in os.walk(target_rf_dir):
                for nome_arq in sorted(lista_arquivos):
                    if nome_arq.startswith("."):
                        continue
                    caminho_completo = os.path.join(raiz, nome_arq)
                    caminho_rel = os.path.relpath(caminho_completo, target_rf_dir).replace("\\", "/")
                    zf.write(caminho_completo, caminho_rel)
        print(f"  [OK] Pacote compactado salvo em: {destino_zip} ({os.path.getsize(destino_zip) / (1024*1024):.2f} MB)")


def download_dataset(
    api_key="",
    workspace="constructionsite",
    project_name="ppe-fruwx",
    version_num=5,
    output_dir=None,
    forcar_download=False,
):
    """
    Valida a disponibilidade local da Base 2 (4.460 imagens nativas YOLO .txt)
    ou realiza sua aquisição via pacote local / GitHub Releases / Hugging Face / API Roboflow.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not output_dir:
        output_dir = os.path.join(base_dir, "datasets", "raw", "roboflow")

    os.makedirs(output_dir, exist_ok=True)
    target_rf_dir = os.path.join(output_dir, "Construction-Site-Safety-1")
    zip_rf_dest = os.path.join(output_dir, "roboflow-safety.zip")
    zip_raw_root = os.path.join(base_dir, "datasets", "raw", "roboflow-safety.zip")
    zip_pacote_release = os.path.join(base_dir, "Pacote_Release_GitHub", "roboflow-safety.zip")

    print("\n" + "=" * 75)
    print("  ETAPA 2: AQUISIÇÃO E VALIDAÇÃO DA BASE 2 (ROBOFLOW PPE - 4.460 IMAGENS)")
    print("=" * 75)
    print(f"  -> Fonte Oficial : {URL_RELEASE_ROBOFLOW_TCC}")
    print(f"  -> Espelho HF    : https://huggingface.co/datasets/{REPOSITORIO_HF_ROBOFLOW}")
    print(f"  -> Destino Final : {target_rf_dir}")
    print("-" * 75)

    tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
    if tot >= 4460 and not forcar_download:
        normalizar_labels_5_colunas_roboflow(target_rf_dir)
        print("  [OK] Base 2 (Roboflow Construction PPE - ppe-fruwx) validada localmente.")
        print(f"       Total: {tot} imagens (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("=" * 75)
        return True

    for zip_candidato in [zip_rf_dest, zip_raw_root, zip_pacote_release]:
        if os.path.exists(zip_candidato) and os.path.getsize(zip_candidato) > 100 * 1024 * 1024 and not forcar_download:
            print(f"  [*] Extraindo Base 2 a partir do pacote local: {zip_candidato}")
            if os.path.exists(target_rf_dir):
                shutil.rmtree(target_rf_dir)
            os.makedirs(target_rf_dir, exist_ok=True)
            with zipfile.ZipFile(zip_candidato, "r") as zf:
                zf.extractall(target_rf_dir)
            normalizar_labels_5_colunas_roboflow(target_rf_dir)
            tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
            if tot >= 4460:
                print(f"  [OK] Base 2 extraída ({tot} imagens: Treino={tr} | Validação={va} | Teste={te}).")
                print("=" * 75)
                return True

    print("  [*] Baixando pacote da Release oficial do GitHub...")
    try:
        def _progresso(blocos, tam_bloco, tam_total):
            if tam_total > 0:
                bx = (blocos * tam_bloco) / (1024 * 1024)
                tt = tam_total / (1024 * 1024)
                pct = min(100.0, (bx / tt) * 100.0)
                print(f"\r  [*] Baixando: {bx:6.1f} MB / {tt:6.1f} MB ({pct:5.1f}%)", end="", flush=True)

        urllib.request.urlretrieve(URL_RELEASE_ROBOFLOW_TCC, zip_rf_dest, reporthook=_progresso)
        if os.path.exists(zip_rf_dest) and os.path.getsize(zip_rf_dest) > 100 * 1024 * 1024:
            print(f"\n  [OK] Pacote salvo em: {zip_rf_dest}")
            if os.path.exists(target_rf_dir):
                shutil.rmtree(target_rf_dir)
            os.makedirs(target_rf_dir, exist_ok=True)
            print(f"  [*] Extraindo para: {target_rf_dir} ...")
            with zipfile.ZipFile(zip_rf_dest, "r") as zf:
                zf.extractall(target_rf_dir)
            normalizar_labels_5_colunas_roboflow(target_rf_dir)
            tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
            if tot >= 4460:
                print(f"  [OK] Base 2 extraída ({tot} imagens: Treino={tr} | Validação={va} | Teste={te}).")
                print("=" * 75)
                return True
            print(f"\n  [!] Pacote da Release continha {tot} imagens (esperado >= 4.460). Acionando espelho Hugging Face...")
    except Exception as e_rel:
        print(f"\n  [!] Download via GitHub Releases não concluído ({e_rel}).")

    try:
        baixar_via_huggingface_roboflow(target_rf_dir, destino_zip=zip_raw_root)
        tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
        if tot >= 4460:
            print(f"  [OK] Base 2 pronta via espelho Hugging Face ({tot} imagens: Treino={tr} | Validação={va} | Teste={te}).")
            print("=" * 75)
            return True
    except Exception as e_hf:
        print(f"\n  [!] Contingência via Hugging Face falhou ({e_hf}).")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [-] Biblioteca 'roboflow' não instalada e arquivo remoto indisponível.")
        print("=" * 75)
        return False

    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        print("  [-] Chave de API do Roboflow (ROBOFLOW_API_KEY) não informada.")
        print("=" * 75)
        return False

    print("  [*] Autenticando na API do Roboflow...")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_num)

        print(f"  [*] Baixando dataset versão {version_num} no formato YOLOv8 (.txt)...")
        old_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            dataset = version.download("yolov8")
            normalizar_labels_5_colunas_roboflow(target_rf_dir)
            print(f"\n  [OK] Dataset baixado em: {dataset.location}")
        finally:
            os.chdir(old_cwd)

        print("=" * 75)
        return True
    except Exception as e:
        print(f"\n  [-] Falha ao baixar dataset via API do Roboflow: {e}")
        print("=" * 75)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aquisição e validação da Base 2 (Roboflow Construction PPE - ppe-fruwx)")
    parser.add_argument("--api-key", "--api_key", dest="api_key", type=str, default="", help="Chave de API do Roboflow")
    parser.add_argument("--version", type=int, default=5, help="Versão do dataset no Roboflow (padrão: 5)")
    parser.add_argument("--workspace", type=str, default="constructionsite", help="Workspace no Roboflow")
    parser.add_argument("--project", type=str, default="ppe-fruwx", help="Nome do projeto no Roboflow")
    parser.add_argument("--forcar-download", action="store_true", help="Força o download mesmo se já existir localmente")
    parser.add_argument("--no-pause", action="store_true", help="Não pausa o terminal ao finalizar")
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        sucesso = download_dataset(
            api_key=args.api_key,
            workspace=args.workspace,
            project_name=args.project,
            version_num=args.version,
            forcar_download=args.forcar_download,
        )
        if not sucesso:
            erro_ocorreu = True
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 75)
        print("  [ERRO EM baixar_roboflow.py]")
        print("!" * 75)
        print(f"  -> Tipo     : {type(exc).__name__}")
        print(f"  -> Mensagem : {exc}")
        traceback.print_exc()
        print("!" * 75)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
