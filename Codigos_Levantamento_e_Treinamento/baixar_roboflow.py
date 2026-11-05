# -*- coding: utf-8 -*-
"""
Módulo: baixar_roboflow.py
Realiza a verificação local ou o download automatizado da Base Complementar
'Roboflow Construction Site Safety' (398 imagens e 580 caixas válidas no padrão YOLO .txt).

Caso as 398 imagens já estejam presentes localmente (ou na estrutura do projeto),
o script valida a base imediatamente sem exigir chave de API nem conexão externa.

Uso no terminal:
    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/baixar_roboflow.py"
"""

import os
import sys
import glob
import shutil
import traceback
import argparse


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


def tentar_sincronizar_roboflow_local(target_rf_dir):
    """
    Procura se a pasta 'Construction-Site-Safety-1' (398 imagens) já existe em
    'Projeto_EPI/datasets/raw/roboflow/Construction-Site-Safety-1' e copia automaticamente.
    """
    raiz_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(target_rf_dir))))
    candidato = os.path.join(raiz_workspace, "Projeto_EPI", "datasets", "raw", "roboflow", "Construction-Site-Safety-1")
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
    Verifica ou realiza o download automatizado do dataset Construction Site Safety via API do Roboflow.
    """
    if not output_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "datasets", "raw", "roboflow")

    os.makedirs(output_dir, exist_ok=True)
    target_rf_dir = os.path.join(output_dir, "Construction-Site-Safety-1")

    print("\n" + "=" * 75)
    print("  ETAPA 2: VERIFICAÇÃO E DOWNLOAD DA BASE 2 (ROBOFLOW CONSTRUCTION SAFETY)")
    print("=" * 75)
    print(f"  -> Workspace : {workspace}")
    print(f"  -> Projeto   : {project_name} (Versão {version_num})")
    print(f"  -> Destino   : {target_rf_dir}")
    print("-" * 75)

    # 1. Verifica se já existe localmente com as 398 imagens
    tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
    if tot >= 398 and not forcar_download:
        print(f"  [OK] Base 2 (Roboflow) já está presente e validada localmente!")
        print(f"       Total: {tot} imagens (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("  [i] Nenhum download externo ou chave de API é necessário.")
        print("=" * 75)
        return True

    # 2. Tenta sincronizar da pasta Projeto_EPI do repositório caso exista
    if not forcar_download and tentar_sincronizar_roboflow_local(target_rf_dir):
        tot, tr, va, te = contar_imagens_roboflow(target_rf_dir)
        print(f"  [OK] Base 2 sincronizada com sucesso ({tot} imagens: {tr} treino, {va} val, {te} teste)!")
        print("=" * 75)
        return True

    # 3. Caso precise baixar via API do Roboflow
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [-] ERRO: A biblioteca 'roboflow' não está instalada no ambiente Python atual.")
        print("  [!] Solução: Execute no terminal: pip install roboflow")
        print("=" * 75)
        return False

    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        print("  [-] ERRO: A base Roboflow não foi encontrada localmente e nenhuma API Key foi informada.")
        print("\n  [!] Como resolver (Escolha uma das opções):")
        print("      Opção A (Automática via API):")
        print("        1. Acesse https://app.roboflow.com (conta gratuita) -> Settings -> Roboflow API")
        print("        2. Copie sua 'Private API Key' e execute:")
        print('           python "Base e treinamento/Codigos_Levantamento_e_Treinamento/baixar_roboflow.py" --api-key SUA_CHAVE')
        print("      Opção B (Manual pelo Navegador):")
        print("        1. Acesse https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety")
        print("        2. Baixe a versão 1 no formato 'YOLOv8' (.zip) e extraia em:")
        print(f"           {target_rf_dir}")
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
        print("  [!] Verifique se sua API Key está correta ou utilize a cópia local em datasets/raw/roboflow.")
        print("=" * 75)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica ou baixa o Dataset Construction Site Safety do Roboflow")
    parser.add_argument("--api-key", "--api_key", dest="api_key", type=str, default="", help="Chave de API privada do Roboflow")
    parser.add_argument("--version", type=int, default=1, help="Versão do dataset no Roboflow (padrão: 1)")
    parser.add_argument("--workspace", type=str, default="roboflow-universe-projects", help="Workspace no Roboflow")
    parser.add_argument("--project", type=str, default="construction-site-safety", help="Nome do projeto no Roboflow")
    parser.add_argument("--forcar-download", action="store_true", help="Força o download via API mesmo se a base já existir localmente")
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
