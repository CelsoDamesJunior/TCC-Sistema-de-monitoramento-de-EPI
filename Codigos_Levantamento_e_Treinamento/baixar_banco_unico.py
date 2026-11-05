# -*- coding: utf-8 -*-
"""
Módulo: baixar_banco_unico.py
Script utilitário dedicado para baixar e validar o pacote oficial 'construction-ppe.zip'
(1.416 imagens e 7.194 caixas válidas nas 4 classes do TCC) diretamente do repositório
oficial da Ultralytics caso o projeto seja clonado em outro computador, ou utilizar a
cópia local já validada sem sobrescrever nem gerar pastas aninhadas.

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

CONSTRUCTION_PPE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"


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


def tentar_sincronizar_de_projeto_epi(extracted_dir):
    """
    Caso a base local em 'Base e treinamento/datasets/raw/construction-ppe' esteja ausente
    ou incompleta, verifica se a pasta irmã 'Projeto_EPI/datasets/raw/construction-ppe'
    possui as 1.416 imagens completas do TCC e sincroniza automaticamente.
    """
    raiz_workspace = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(extracted_dir))))
    candidato = os.path.join(raiz_workspace, "Projeto_EPI", "datasets", "raw", "construction-ppe")
    if os.path.isdir(candidato):
        tot, tr, va, te = contar_imagens_extraidas(candidato)
        if tot == 1416:
            print(f"  [*] Encontrada base local íntegra em: {candidato} (1.416 imagens).")
            print(f"  [*] Sincronizando para: {extracted_dir} ...")
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
            shutil.copytree(candidato, extracted_dir)
            return True
    return False


def baixar_arquivo_zip(extrair=True, forcar_download=False):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "datasets", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_dest = os.path.join(raw_dir, "construction-ppe.zip")
    extracted_dir = os.path.join(raw_dir, "construction-ppe")

    print("\n" + "=" * 75)
    print("  ETAPA 1: VERIFICAÇÃO E DOWNLOAD DA BASE 1 (ULTRALYTICS CONSTRUCTION-PPE)")
    print("=" * 75)
    print(f"  -> URL Oficial : {CONSTRUCTION_PPE_URL}")
    print(f"  -> Arquivo ZIP : {zip_dest}")
    print(f"  -> Pasta Alvo  : {extracted_dir}")
    print("-" * 75)

    # 1. Verifica se a base de 1.416 imagens do TCC já está presente e íntegra
    total_atual, tr_atual, va_atual, te_atual = contar_imagens_extraidas(extracted_dir)
    if total_atual == 1416 and not forcar_download:
        print(f"  [OK] Base 1 já está presente e 100% íntegra em:\n       {extracted_dir}")
        print(f"       Total: {total_atual} imagens (Treino: {tr_atual} | Val: {va_atual} | Teste: {te_atual})")
        print("  [i] Nenhum novo download é necessário (use --forcar-download se quiser baixar novamente).")
        print("=" * 75)
        return zip_dest

    # 2. Se não estiver extraída mas existir na pasta Projeto_EPI do repositório, sincroniza
    if not forcar_download and tentar_sincronizar_de_projeto_epi(extracted_dir):
        total_atual, tr_atual, va_atual, te_atual = contar_imagens_extraidas(extracted_dir)
        print(f"  [OK] Base 1 sincronizada com sucesso ({total_atual} imagens: {tr_atual} treino, {va_atual} val, {te_atual} teste)!")
        print("=" * 75)
        return zip_dest

    # 3. Se já existir um arquivo .zip local válido e não foi pedido --forcar-download, extrai dele
    if os.path.exists(zip_dest) and os.path.getsize(zip_dest) > 50 * 1024 * 1024 and not forcar_download:
        tam_mb = os.path.getsize(zip_dest) / (1024 * 1024)
        print(f"  [OK] Pacote local 'construction-ppe.zip' encontrado ({tam_mb:.2f} MB).")
    else:
        print("  [*] Iniciando download do pacote 'construction-ppe.zip' da Ultralytics...")

        def _progresso(blocos, tam_bloco, tam_total):
            if tam_total > 0:
                baixado_mb = (blocos * tam_bloco) / (1024 * 1024)
                total_mb = tam_total / (1024 * 1024)
                pct = min(100.0, (baixado_mb / total_mb) * 100.0)
                print(f"\r  [*] Baixando: {baixado_mb:6.1f} MB / {total_mb:6.1f} MB ({pct:5.1f}%)", end="", flush=True)

        urllib.request.urlretrieve(CONSTRUCTION_PPE_URL, zip_dest, reporthook=_progresso)
        print(f"\n  [OK] Arquivo 'construction-ppe.zip' salvo em:\n       {zip_dest}")

    if extrair:
        print(f"  [*] Extraindo conteúdo para: {extracted_dir} ...")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(extracted_dir)
        corrigir_pasta_aninhada(extracted_dir)
        total_atual, tr_atual, va_atual, te_atual = contar_imagens_extraidas(extracted_dir)
        print(f"  [OK] Extração concluída! Imagens detectadas: {total_atual} (Treino: {tr_atual} | Val: {va_atual} | Teste: {te_atual})")

    print("=" * 75)
    return zip_dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa e valida o arquivo construction-ppe.zip (Base 1 do TCC)")
    parser.add_argument("--apenas-zip", action="store_true", help="Apenas garante o .zip sem descompactar novamente")
    parser.add_argument("--forcar-download", action="store_true", help="Força um novo download da internet mesmo se já existir localmente")
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
        print("\n  [Como resolver]:")
        print("  1. Verifique sua conexão com a internet caso o arquivo .zip ainda não tenha sido baixado.")
        print("  2. Verifique se há espaço em disco suficiente e permissão de escrita na pasta 'datasets/raw'.")
        print("\n  [Detalhes técnicos (Traceback)]:")
        traceback.print_exc()
        print("!" * 75)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
