# -*- coding: utf-8 -*-
"""
Módulo: baixar_banco_unico.py
Script utilitário dedicado para baixar e validar o pacote oficial 'construction-ppe'
(1.416 imagens e 7.194 caixas válidas nas 4 classes do TCC) para a Base 1.

FONTE OFICIAL DE DOWNLOAD (BLINDADA):
  - GitHub Releases do TCC: https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/construction-ppe.zip
  - Fallback Ultralytics: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip

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

URL_RELEASE_TCC = "https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/construction-ppe.zip"
URL_ULTRALYTICS_FALLBACK = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"


def pausar_se_interativo(forcar_sem_pausa=False, houve_erro=False):
    """
    Evita que a janela do terminal seja fechada automaticamente ao clicar duas vezes no script
    ou quando ocorre algum erro, permitindo ler o diagnóstico completo na tela.
    """
    if forcar_sem_pausa:
        return
    try:
        if houve_erro or (sys.stdin and sys.stdin.isatty()):
            print("\n" + "-" * 75)
            input("  [PAUSA] Pressione ENTER para finalizar e fechar esta janela...")
    except (EOFError, KeyboardInterrupt):
        pass


def corrigir_pasta_aninhada(extracted_dir):
    """
    Se o arquivo .zip continha uma subpasta interna 'construction-ppe/construction-ppe',
    move automaticamente 'images/', 'labels/' e 'data.yaml' para o nível correto.
    """
    subpasta_interna = os.path.join(extracted_dir, "construction-ppe")
    if os.path.isdir(subpasta_interna) and os.path.isdir(os.path.join(subpasta_interna, "images")):
        print("  [*] Ajustando estrutura de subpasta aninhada (construction-ppe/construction-ppe)...")
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
        print("  [OK] Estrutura corrigida para:", extracted_dir)


def contar_imagens_extraidas(extracted_dir):
    """Retorna (total, train, val, test) de imagens na pasta extraída."""
    if not os.path.isdir(extracted_dir):
        return 0, 0, 0, 0
    corrigir_pasta_aninhada(extracted_dir)
    tr = len(glob.glob(os.path.join(extracted_dir, "images", "train", "*.*")))
    va = len(glob.glob(os.path.join(extracted_dir, "images", "val", "*.*")))
    te = len(glob.glob(os.path.join(extracted_dir, "images", "test", "*.*")))
    return (tr + va + te), tr, va, te


def tentar_copiar_de_pasta_local(extracted_dir, zip_dest):
    """Verifica se as 1.416 imagens ou o .zip já existem no computador."""
    # Candidato 1: Desktop
    user_home = os.path.expanduser("~")
    possiveis_zips = [
        os.path.join(user_home, "OneDrive - Frilog Transportes", "Área de Trabalho", "construction-ppe.zip"),
        os.path.join(user_home, "Desktop", "construction-ppe.zip"),
    ]
    for p_zip in possiveis_zips:
        if os.path.exists(p_zip) and os.path.getsize(p_zip) > 100 * 1024 * 1024:
            print(f"  [*] Encontrado pacote original em: {p_zip}")
            shutil.copy2(p_zip, zip_dest)
            return True

    # Candidato 2: Projeto_EPI
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raiz_ws = os.path.dirname(base_dir)
    candidato_dir = os.path.join(raiz_ws, "Projeto_EPI", "datasets", "raw", "construction-ppe")
    if os.path.isdir(candidato_dir):
        tot, tr, va, te = contar_imagens_extraidas(candidato_dir)
        if tot == 1416:
            print(f"  [*] Sincronizando Base 1 (1.416 imagens) a partir de: {candidato_dir}")
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
            shutil.copytree(candidato_dir, extracted_dir)
            return True
    return False


def baixar_arquivo_zip(extrair=True, forcar_download=False):
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(pasta_script)
    raw_dir = os.path.join(base_dir, "datasets", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_dest = os.path.join(raw_dir, "construction-ppe.zip")
    extracted_dir = os.path.join(raw_dir, "construction-ppe")

    print("\n" + "=" * 75)
    print("  ETAPA 1: AQUISIÇÃO E VALIDAÇÃO DA BASE 1 (CONSTRUCTION-PPE - 1.416 IMAGENS)")
    print("=" * 75)
    print(f"  -> Release Oficial : {URL_RELEASE_TCC}")
    print(f"  -> Arquivo Alvo    : {zip_dest}")
    print(f"  -> Pasta Alvo      : {extracted_dir}")
    print("-" * 75)

    # 1. Verifica se a base descompactada já está completa (1.416 imagens)
    tot, tr, va, te = contar_imagens_extraidas(extracted_dir)
    if tot == 1416 and not forcar_download:
        print(f"  [OK] Base 1 já validada localmente em:\n       {extracted_dir}")
        print(f"       Total: {tot} imagens (Treino: {tr} | Validação: {va} | Teste: {te})")
        print("  [i] Nenhum novo download é necessário.")
        print("=" * 75)
        return zip_dest

    # 2. Se não estiver extraída, tenta usar cópia local (Desktop ou Projeto_EPI)
    if not forcar_download and tentar_copiar_de_pasta_local(extracted_dir, zip_dest):
        tot, tr, va, te = contar_imagens_extraidas(extracted_dir)
        if tot == 1416:
            print(f"  [OK] Base 1 sincronizada com sucesso ({tot} imagens)!")
            print("=" * 75)
            return zip_dest

    # 3. Se o .zip já existir localmente
    if os.path.exists(zip_dest) and os.path.getsize(zip_dest) > 100 * 1024 * 1024 and not forcar_download:
        tam_mb = os.path.getsize(zip_dest) / (1024 * 1024)
        print(f"  [OK] Pacote local 'construction-ppe.zip' encontrado ({tam_mb:.2f} MB).")
    else:
        # 4. Baixa direto da Release oficial do GitHub
        def _progresso(blocos, tam_bloco, tam_total):
            if tam_total > 0:
                bx = (blocos * tam_bloco) / (1024 * 1024)
                tt = tam_total / (1024 * 1024)
                pct = min(100.0, (bx / tt) * 100.0)
                print(f"\r  [*] Baixando: {bx:6.1f} MB / {tt:6.1f} MB ({pct:5.1f}%)", end="", flush=True)

        url_sucesso = None
        for tentativa_url in [URL_RELEASE_TCC, URL_ULTRALYTICS_FALLBACK]:
            try:
                print(f"  [*] Baixando pacote oficial de:\n      {tentativa_url}")
                urllib.request.urlretrieve(tentativa_url, zip_dest, reporthook=_progresso)
                print(f"\n  [OK] Pacote salvo em: {zip_dest}")
                url_sucesso = tentativa_url
                break
            except Exception as e:
                print(f"\n  [!] Aviso: Não foi possível obter de {tentativa_url} ({e}). Tentando próxima fonte...")

        if not url_sucesso and not os.path.exists(zip_dest):
            raise RuntimeError(
                "Não foi possível baixar o arquivo construction-ppe.zip.\n"
                f"Publique a Release em: https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/new\n"
                f"ou copie o arquivo para: {zip_dest}"
            )

    if extrair:
        print(f"  [*] Extraindo conteúdo para: {extracted_dir} ...")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(extracted_dir)
        corrigir_pasta_aninhada(extracted_dir)
        tot, tr, va, te = contar_imagens_extraidas(extracted_dir)
        print(f"  [OK] Extração concluída! Imagens detectadas: {tot} (Treino: {tr} | Val: {va} | Teste: {te})")

    print("=" * 75)
    return zip_dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa e valida o arquivo construction-ppe.zip (Base 1 do TCC)")
    parser.add_argument("--apenas-zip", action="store_true", help="Apenas garante o .zip sem descompactar novamente")
    parser.add_argument("--forcar-download", action="store_true", help="Força novo download mesmo se já existir localmente")
    parser.add_argument("--no-pause", action="store_true", help="Não pausa o terminal ao finalizar")
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        baixar_arquivo_zip(extrair=not args.apenas_zip, forcar_download=args.forcar_download)
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 75)
        print("  [ERRO DETECTADO EM baixar_banco_unico.py]")
        print("!" * 75)
        print(f"  -> Tipo do erro : {type(exc).__name__}")
        print(f"  -> Mensagem     : {exc}")
        print("\n  [Detalhes técnicos (Traceback)]:")
        traceback.print_exc()
        print("!" * 75)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
