# -*- coding: utf-8 -*-
"""
Módulo: baixar_roboflow.py
Realiza a aquisição da Base Complementar 'Roboflow Construction Site Safety'
(398 imagens e 580 caixas válidas no padrão YOLO .txt).

FONTE OFICIAL DE DOWNLOAD (BLINDADA):
  - GitHub Releases do TCC: https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/roboflow-safety.zip
  - Fallback: API Oficial do Roboflow Universe ou Cópia Local

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

URL_RELEASE_ROBOFLOW_TCC = "https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/roboflow-safety.zip"


def pausar_se_interativo(forcar_sem_pausa=False, houve_erro=False):
    """
    Mantém o terminal aberto ao final da execução (especialmente em caso de erro ou duplo clique)
    para que o usuário possa ler todas as mensagens com clareza.
    """
    if forcar_sem_pausa:
        return
    try:
        if houve_erro or (sys.stdin and sys.stdin.isatty()):
            print("\n" + "-" * 75)
            input("  [PAUSA] Pressione ENTER para finalizar e fechar esta janela...")
    except (EOFError, KeyboardInterrupt):
        pass


def contar_imagens_roboflow(rf_dir):
    """Retorna (total, train, valid, test) de imagens na pasta Construction-Site-Safety-1."""
    if not os.path.isdir(rf_dir):
        return 0, 0, 0, 0
    tr = len(glob.glob(os.path.join(rf_dir, "train", "images", "*.*")))
    va = len(glob.glob(os.path.join(rf_dir, "valid", "images", "*.*")))
    te = len(glob.glob(os.path.join(rf_dir, "test", "images", "*.*")))
    return (tr + va + te), tr, va, te


def tentar_sincronizar_roboflow_local(target_rf_dir, zip_dest):
    """Verifica se as 398 imagens ou o .zip já existem no computador."""
    user_home = os.path.expanduser("~")
    possiveis_zips = [
        os.path.join(user_home, "OneDrive - Frilog Transportes", "Área de Trabalho", "roboflow-safety.zip"),
        os.path.join(user_home, "Desktop", "roboflow-safety.zip"),
    ]
    for p_zip in possiveis_zips:
        if os.path.exists(p_zip) and os.path.getsize(p_zip) > 10 * 1024 * 1024:
            print(f"  [*] Encontrado pacote original em: {p_zip}")
            shutil.copy2(p_zip, zip_dest)
            return True

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raiz_ws = os.path.dirname(base_dir)
    candidato = os.path.join(raiz_ws, "Projeto_EPI", "datasets", "raw", "roboflow", "Construction-Site-Safety-1")
    if os.path.isdir(candidato):
        tot, tr, va, te = contar_imagens_roboflow(candidato)
        if tot >= 398:
            print(f"  [*] Base Roboflow local encontrada em: {candidato} ({tot} imagens).")
            print(f"  [*] Sincronizando para: {target_rf_dir} ...")
            if os.path.exists(target_rf_dir):
                shutil.rmtree(target_rf_dir)
            shutil.copytree(candidato, target_rf_dir)
            return True
    return False


def download_dataset(
    api_key="",
    workspace="roboflow-universe-projects",
    project_name="construction-site-safety",
    version_num=1,
    output_dir=None,
    forcar_download=False,
):
    """
    Verifica localmente, baixa da Release oficial do GitHub ou baixa via API do Roboflow.
    """
    if not output_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "datasets", "raw", "roboflow")

    os.makedirs(output_dir, exist_ok=True)
    target_rf_dir = os.path.join(output_dir, "Construction-Site-Safety-1")
    zip_rf_dest = os.path.join(output_dir, "roboflow-safety.zip")

    print("\n" + "=" * 75)
    print("  ETAPA 2: AQUISIÇÃO E VALIDAÇÃO DA BASE 2 (ROBOFLOW SAFETY - 398 IMAGENS)")
    print("=" * 75)
    print(f"  -> Release Oficial : {URL_RELEASE_ROBOFLOW_TCC}")
    print(f"  -> Destino Final   : {target_rf_dir}")
    print("-" * 75)

    # 1. Verifica se já existe localmente com as 398 imagens
    tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
    if tot >= 398 and not forcar_download:
        print(f"  [OK] Base 2 (Roboflow) já está presente e validada localmente!")
        print(f"       Total: {tot} imagens (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("  [i] Nenhum download externo ou chave de API é necessário.")
        print("=" * 75)
        return True

    # 2. Tenta sincronizar de pastas locais (Desktop ou Projeto_EPI)
    if not forcar_download and tentar_sincronizar_roboflow_local(target_rf_dir, zip_rf_dest):
        tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
        if tot >= 398:
            print(f"  [OK] Base 2 sincronizada com sucesso ({tot} imagens)!")
            print("=" * 75)
            return True

    # 3. Tenta baixar o arquivo roboflow-safety.zip direto da Release do GitHub
    print("  [*] Tentando baixar pacote oficial da Release do GitHub...")
    try:
        def _progresso(blocos, tam_bloco, tam_total):
            if tam_total > 0:
                bx = (blocos * tam_bloco) / (1024 * 1024)
                tt = tam_total / (1024 * 1024)
                pct = min(100.0, (bx / tt) * 100.0)
                print(f"\r  [*] Baixando: {bx:6.1f} MB / {tt:6.1f} MB ({pct:5.1f}%)", end="", flush=True)

        urllib.request.urlretrieve(URL_RELEASE_ROBOFLOW_TCC, zip_rf_dest, reporthook=_progresso)
        print(f"\n  [OK] Pacote baixado com sucesso em: {zip_rf_dest}")
        print(f"  [*] Extraindo para: {target_rf_dir} ...")
        with zipfile.ZipFile(zip_rf_dest, "r") as zf:
            zf.extractall(target_rf_dir)
        tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
        print(f"  [OK] Base 2 extraída e pronta ({tot} imagens)!")
        print("=" * 75)
        return True
    except Exception as e_rel:
        print(f"\n  [!] Download via Release não concluído ({e_rel}).")

    # 4. Fallback: Se o usuário tiver a API Key do Roboflow
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [-] ERRO: Não foi possível obter a Base 2 pela Release e a biblioteca 'roboflow' não está instalada.")
        print("  [!] Solução rápida: Publique a release em https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/new")
        print("=" * 75)
        return False

    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        print("  [-] Nenhuma API Key do Roboflow informada.")
        print("  [!] Publique a Release v1.0.0 no GitHub com o arquivo 'roboflow-safety.zip'.")
        print("=" * 75)
        return False

    print("  [*] Autenticando com a API do Roboflow...")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_num)

        print(f"  [*] Baixando dataset versão {version_num} no formato YOLOv8 (.txt)...")
        old_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            dataset = version.download("yolov8")
            print(f"\n  [OK] Sucesso! Dataset baixado em: {dataset.location}")
        finally:
            os.chdir(old_cwd)

        print("=" * 75)
        return True
    except Exception as e:
        print(f"\n  [-] Falha ao baixar dataset do Roboflow via API: {e}")
        print("=" * 75)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica ou baixa o Dataset Construction Site Safety do Roboflow")
    parser.add_argument("--api-key", "--api_key", dest="api_key", type=str, default="", help="Chave de API do Roboflow")
    parser.add_argument("--version", type=int, default=1, help="Versão do dataset no Roboflow (padrão: 1)")
    parser.add_argument("--workspace", type=str, default="roboflow-universe-projects", help="Workspace no Roboflow")
    parser.add_argument("--project", type=str, default="construction-site-safety", help="Nome do projeto no Roboflow")
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
        print("  [ERRO INESPERADO EM baixar_roboflow.py]")
        print("!" * 75)
        print(f"  -> Tipo do erro : {type(exc).__name__}")
        print(f"  -> Mensagem     : {exc}")
        traceback.print_exc()
        print("!" * 75)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
