# -*- coding: utf-8 -*-
"""
Módulo: preparar_dados.py
Pipeline de Engenharia de Dados do Projeto:
  1. Verifica e garante a integridade das quatro bases nativas no formato YOLO (.txt):
     - Base 1 (Principal): Ultralytics Construction-PPE (1.416 imagens limpas sem cisalhamento -> 1.403 válidas com Person e 9.775 caixas)
     - Base 2 (Complementar 1): Roboflow Construction PPE ppe-fruwx (4.460 imagens originais -> 2.631 únicas com Person e 16.012 caixas)
     - Base 3 (Complementar 2): Chapian Construction Site PPE Detection (200 imagens originais -> 187 únicas com Person e 1.544 caixas)
     - Base 4 (Complementar 3): SH17 Dataset - Human & PPE Detection (313 imagens completas com Person e 2.519 caixas)
  2. Executa deduplicação criptográfica MD5 (blocos de 64 KB).
  3. Padroniza os identificadores das 4 classes NR-06 / NR-18 (0: Hardhat, 1: Safety-Vest, 2: Person, 3: Safety-Shoes)
     e remove imagens órfãs/incompletas (sem anotação de Person) que causariam falsos negativos no treinamento.
  4. Aplica equalização adaptativa CLAHE suave (clipLimit=1.5) no canal L (CIE LAB) preservando objetos pequenos (Safety-Shoes).
  5. Gera o dataset consolidado 'unified_tcc' (4.534 imagens e 29.850 caixas delimitadoras) e os manifestos 'data.yaml'.

Uso no terminal:
    python "Base e treinamento/Codigos_Levantamento_e_Treinamento/preparar_dados.py" --apply-clahe
"""

import os
import sys
import glob
import hashlib
import shutil
import zipfile
import traceback
import urllib.request
import argparse

try:
    import cv2
except ImportError as exc_cv2:
    print("\n[ERRO CRÍTICO] Biblioteca 'opencv-python' (cv2) não encontrada.")
    print(" -> Instale executando: pip install opencv-python")
    raise SystemExit(1) from exc_cv2

try:
    import yaml
except ImportError as exc_yaml:
    print("\n[ERRO CRÍTICO] Biblioteca 'PyYAML' (yaml) não encontrada.")
    print(" -> Instale executando: pip install pyyaml")
    raise SystemExit(1) from exc_yaml

from baixar_roboflow import download_dataset as download_roboflow_dataset
from baixar_chapian import download_chapian_dataset, URL_RELEASE_CHAPIAN_TCC, HF_CHAPIAN_REPO
from baixar_sh17 import download_sh17_dataset, URL_RELEASE_SH17_TCC, HF_SH17_REPO

# Fontes de distribuição das bases de dados no formato YOLO (.txt)
CONSTRUCTION_PPE_URL = "https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/construction-ppe.zip"
ROBOFLOW_RELEASE_URL = "https://github.com/CelsoDamesJunior/TCC-Sistema-de-monitoramento-de-EPI/releases/download/v1.0.0/roboflow-safety.zip"
CHAPIAN_RELEASE_URL = URL_RELEASE_CHAPIAN_TCC
SH17_RELEASE_URL = URL_RELEASE_SH17_TCC
CONSTRUCTION_PPE_DOCS = "https://docs.ultralytics.com/datasets/detect/construction-ppe/"
ROBOFLOW_SAFETY_URL = "https://huggingface.co/datasets/vikasaarna/roboflow-construction-ppe-fruwx"
CHAPIAN_DOCS_URL = f"https://huggingface.co/datasets/{HF_CHAPIAN_REPO}"
SH17_DOCS_URL = f"https://huggingface.co/datasets/{HF_SH17_REPO}"

# Mapeamento Taxonômico 1: Base Principal Ultralytics Construction-PPE (11 classes -> 4 classes alvo)
# Original: 0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none, 6: Person, 7: no_helmet, 8: no_goggle, 9: no_gloves, 10: no_boots
# Padronizado: 0: Hardhat, 1: Safety-Vest, 2: Person, 3: Safety-Shoes
SINGLE_DB_CLASS_MAPPING = {
    0: 0,  # helmet -> Hardhat (ID 0)
    2: 1,  # vest   -> Safety-Vest (ID 1)
    6: 2,  # Person -> Person (ID 2)
    3: 3   # boots  -> Safety-Shoes (ID 3)
}

# Mapeamento Taxonômico 2: Base Complementar 1 Roboflow Construction PPE - ppe-fruwx (8 classes -> 4 classes alvo)
# Original: 0: boots, 1: helmet, 2: helmet on, 3: no boots, 4: no helmet, 5: no vest, 6: person, 7: vest
# Compatibilidade adicional com esquema antigo (4: Hardhat, 12: Safety Vest, 9: Person, 11: Safety Shoes)
ROBOFLOW_CLASS_MAPPING = {
    1: 0,   # helmet    -> Hardhat (ID 0)
    2: 0,   # helmet on -> Hardhat (ID 0)
    7: 1,   # vest      -> Safety-Vest (ID 1)
    6: 2,   # person    -> Person (ID 2)
    0: 3    # boots     -> Safety-Shoes (ID 3)
}

# Mapeamento Taxonômico 3: Base Complementar 2 Chapian Construction Site PPE Detection (9 classes -> 4 classes alvo)
# Original: 0: boots, 1: gloves, 2: hardhat, 3: no_boots, 4: no_gloves, 5: no_hardhat, 6: no_vest, 7: person, 8: vest
CHAPIAN_CLASS_MAPPING = {
    2: 0,   # hardhat -> Hardhat (ID 0)
    8: 1,   # vest    -> Safety-Vest (ID 1)
    7: 2,   # person  -> Person (ID 2)
    0: 3    # boots   -> Safety-Shoes (ID 3)
}

# Mapeamento Taxonômico 4: Base Complementar 3 SH17 Dataset (17 classes -> 4 classes alvo)
# Original: 0: person, 10: helmet, 14: shoes, 16: safety-vest
SH17_CLASS_MAPPING = {
    10: 0,  # helmet      -> Hardhat (ID 0)
    16: 1,  # safety-vest -> Safety-Vest (ID 1)
    0: 2,   # person      -> Person (ID 2)
    14: 3   # shoes       -> Safety-Shoes (ID 3)
}


def pausar_se_interativo(forcar_sem_pausa=False, houve_erro=False):
    """Aguarda confirmação do usuário ao final de execuções interativas em terminal."""
    if forcar_sem_pausa:
        return
    try:
        if houve_erro or (sys.stdin and sys.stdin.isatty()):
            print("\n" + "-" * 80)
            input("  [PAUSA] Pressione ENTER para finalizar...")
    except (EOFError, KeyboardInterrupt):
        pass


def corrigir_pasta_aninhada(extracted_dir):
    """Normaliza a estrutura de diretórios caso o arquivo compactado possua subpasta raiz duplicada."""
    subpasta = os.path.join(extracted_dir, "construction-ppe")
    if os.path.isdir(subpasta) and os.path.isdir(os.path.join(subpasta, "images")):
        for item in os.listdir(subpasta):
            src_item = os.path.join(subpasta, item)
            dst_item = os.path.join(extracted_dir, item)
            if os.path.exists(dst_item):
                if os.path.isdir(dst_item):
                    shutil.rmtree(dst_item)
                else:
                    os.remove(dst_item)
            shutil.move(src_item, dst_item)
        shutil.rmtree(subpasta, ignore_errors=True)


def apply_clahe(image_path, output_path, target_size=(640, 640), clip_limit=1.5, tile_grid_size=(8, 8)):
    """
    Aplica redimensionamento espacial (interpolação INTER_AREA)
    e equalização de histograma adaptativa limitada por contraste (CLAHE suave, clipLimit=1.5)
    no canal L do espaço de cores CIE LAB, preservando bordas de objetos pequenos (Safety-Shoes).
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    if target_size is not None and (img.shape[1], img.shape[0]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a_channel, b_channel))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, final, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return True


def inspect_and_standardize_labels(label_path, class_mapping=None):
    """
    Lê e padroniza em memória os identificadores de classes de um arquivo .txt no formato YOLO.
    Converte automaticamente eventuais anotações poligonais (> 5 colunas) para caixas delimitadoras (5 colunas).
    Retorna (new_lines, ignored_count, kept_by_class).
    """
    if class_mapping is None:
        class_mapping = SINGLE_DB_CLASS_MAPPING

    if not os.path.exists(label_path):
        return [], 0, {0: 0, 1: 0, 2: 0, 3: 0}

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    kept_by_class = {0: 0, 1: 0, 2: 0, 3: 0}
    ignored_count = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        old_class = int(parts[0])
        if old_class in class_mapping:
            new_class = class_mapping[old_class]
            vals = [float(v) for v in parts[1:]]
            if len(vals) > 4:
                xs = vals[0::2]
                ys = vals[1::2]
                x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
                y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
                coords = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0]
            else:
                coords = [max(0.0, min(1.0, v)) for v in vals[:4]]
            if coords[2] > 0.001 and coords[3] > 0.001:
                new_lines.append(f"{new_class} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                kept_by_class[new_class] += 1
        else:
            ignored_count += 1

    return new_lines, ignored_count, kept_by_class


def standardize_labels(label_path, output_path, class_mapping=None):
    """
    Padroniza os identificadores de classes de um arquivo .txt no formato YOLO e grava em disco.
    """
    new_lines, ignored_count, kept_by_class = inspect_and_standardize_labels(label_path, class_mapping)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    return len(new_lines), ignored_count, kept_by_class


def compute_md5(file_path):
    """Calcula o hash MD5 de um arquivo em blocos de 64 KB para verificação de integridade e deduplicação."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_single_dataset(base_dir):
    """
    Garante a disponibilidade da Base 1 (Ultralytics Construction-PPE: 1.416 imagens limpas)
    em datasets/raw/construction-ppe.
    """
    raw_dir = os.path.join(base_dir, 'datasets', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    zip_dest = os.path.join(raw_dir, 'construction-ppe.zip')
    zip_rel = os.path.join(base_dir, 'Pacote_Release_GitHub', 'construction-ppe.zip')
    extracted_dir = os.path.join(raw_dir, 'construction-ppe')

    corrigir_pasta_aninhada(extracted_dir)

    if os.path.exists(extracted_dir) and os.path.exists(os.path.join(extracted_dir, 'images', 'train')):
        n_train = len(glob.glob(os.path.join(extracted_dir, 'images', 'train', '*.*')))
        n_val = len(glob.glob(os.path.join(extracted_dir, 'images', 'val', '*.*')))
        n_test = len(glob.glob(os.path.join(extracted_dir, 'images', 'test', '*.*')))
        if (n_train + n_val + n_test) == 1416:
            print(f"[OK] Base 1 'construction-ppe' validada em: {extracted_dir} (1.416 imagens)")
            return extracted_dir

    for cand_zip in [zip_dest, zip_rel]:
        if os.path.exists(cand_zip) and os.path.getsize(cand_zip) > 50 * 1024 * 1024:
            print(f"[*] Extraindo Base 1 a partir do arquivo local: {cand_zip}")
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
            os.makedirs(extracted_dir, exist_ok=True)
            with zipfile.ZipFile(cand_zip, 'r') as zf:
                zf.extractall(extracted_dir)
            corrigir_pasta_aninhada(extracted_dir)
            return extracted_dir

    print(f"[*] Baixando Base 1 (Ultralytics Construction-PPE) de:\n    {CONSTRUCTION_PPE_URL}")
    urllib.request.urlretrieve(CONSTRUCTION_PPE_URL, zip_dest)
    print(f"[+] Download concluído ({os.path.getsize(zip_dest) / (1024*1024):.1f} MB). Extraindo arquivos...")

    with zipfile.ZipFile(zip_dest, 'r') as zf:
        zf.extractall(extracted_dir)
    corrigir_pasta_aninhada(extracted_dir)

    print(f"[OK] Extração concluída em: {extracted_dir}")
    return extracted_dir


def ensure_roboflow_dataset(base_dir):
    """
    Garante a disponibilidade da Base 2 (Roboflow Construction PPE ppe-fruwx: 4.460 imagens)
    em datasets/raw/roboflow/Construction-Site-Safety-1.
    """
    output_dir = os.path.join(base_dir, 'datasets', 'raw', 'roboflow')
    src_rf = os.path.join(output_dir, 'Construction-Site-Safety-1')
    download_roboflow_dataset(output_dir=output_dir)
    return src_rf


def ensure_chapian_dataset(base_dir):
    """
    Garante a disponibilidade da Base 3 (Chapian PPE Detection: 200 imagens nativas YOLO .txt)
    em datasets/raw/chapian-ppe.
    """
    raw_dir = os.path.join(base_dir, 'datasets', 'raw')
    return download_chapian_dataset(diretorio_saida=raw_dir)


def ensure_sh17_dataset(base_dir):
    """
    Garante a disponibilidade da Base 4 (SH17 PPE Dataset: 313 imagens completas nativas YOLO .txt)
    em datasets/raw/sh17-ppe.
    """
    raw_dir = os.path.join(base_dir, 'datasets', 'raw')
    return download_sh17_dataset(diretorio_saida=raw_dir)


def build_unified_dataset(use_clahe=True, clip_limit=1.5, filtrar_incompletas_rf=True):
    """
    Constrói o dataset consolidado e padronizado (datasets/processed/unified_tcc)
    integrando as quatro bases nativas no padrão YOLO (.txt):
      1. Base 1 (Principal): Ultralytics Construction-PPE (1.416 imagens limpas -> 1.403 válidas com Person e 9.775 instâncias)
      2. Base 2 (Complementar 1): Roboflow Construction PPE ppe-fruwx (4.460 imagens -> 2.631 únicas com Person e 16.012 instâncias)
      3. Base 3 (Complementar 2): Chapian Construction Site PPE Detection (187 imagens únicas com Person e 1.544 instâncias)
      4. Base 4 (Complementar 3): SH17 PPE Dataset (313 imagens completas com Person e 2.519 instâncias)
    """
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(pasta_script)
    src_ppe = download_single_dataset(base_dir)
    src_rf = ensure_roboflow_dataset(base_dir)
    src_ch = ensure_chapian_dataset(base_dir)
    src_sh17 = ensure_sh17_dataset(base_dir)
    out_dir = os.path.join(base_dir, 'datasets', 'processed', 'unified_tcc')

    print("\n" + "=" * 88)
    print("  INICIANDO PIPELINE DE ENGENHARIA DE DADOS (UNIFIED_TCC - 4 BASES NATIVAS YOLO)")
    print("=" * 88)
    print(f"  -> Filtro CLAHE (CIE LAB) : {'ATIVADO (640x640 INTER_AREA, clipLimit=' + str(clip_limit) + ')' if use_clahe else 'DESATIVADO (cópia direta)'}")
    print(f"  -> Base 1 (Principal)     : {src_ppe}")
    print(f"  -> Base 2 (Roboflow PPE)  : {src_rf} (Filtro anti-falso-negativo: {'ATIVO' if filtrar_incompletas_rf else 'INATIVO'})")
    print(f"  -> Base 3 (Chapian PPE)   : {src_ch}")
    print(f"  -> Base 4 (SH17 PPE)      : {src_sh17}")
    print(f"  -> Destino Consolidado    : {out_dir}")
    print("-" * 88)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

    seen_hashes = set()
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    source_counts = {'construction_ppe': 0, 'roboflow_safety': 0, 'chapian_ppe': 0, 'sh17_ppe': 0}
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    ppe_split_boxes = {'train': {0: 0, 1: 0, 2: 0, 3: 0}, 'val': {0: 0, 1: 0, 2: 0, 3: 0}, 'test': {0: 0, 1: 0, 2: 0, 3: 0}}
    rf_split_boxes = {'train': {0: 0, 1: 0, 2: 0, 3: 0}, 'val': {0: 0, 1: 0, 2: 0, 3: 0}, 'test': {0: 0, 1: 0, 2: 0, 3: 0}}
    ch_split_boxes = {'train': {0: 0, 1: 0, 2: 0, 3: 0}, 'val': {0: 0, 1: 0, 2: 0, 3: 0}, 'test': {0: 0, 1: 0, 2: 0, 3: 0}}
    sh17_split_boxes = {'train': {0: 0, 1: 0, 2: 0, 3: 0}, 'val': {0: 0, 1: 0, 2: 0, 3: 0}, 'test': {0: 0, 1: 0, 2: 0, 3: 0}}
    total_ignored = 0
    duplicates_removed = 0
    empty_or_incomplete_skipped = 0

    # 1. Ingestão da Base 1: Ultralytics Construction-PPE (1.416 imagens brutas)
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(src_ppe, 'images', split)
        lbl_dir = os.path.join(src_ppe, 'labels', split)
        if not os.path.exists(img_dir):
            continue

        arquivos_split = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        print(f"  [*] Processando Base 1 (Construction-PPE) - partição '{split}' ({len(arquivos_split)} arquivos)...")

        for img_path in arquivos_split:
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            src_lbl = os.path.join(lbl_dir, f"{base_name}.txt")
            new_lines, ignored_n, kept_dict = inspect_and_standardize_labels(src_lbl, SINGLE_DB_CLASS_MAPPING)

            if filtrar_incompletas_rf and (len(new_lines) == 0 or kept_dict[2] == 0):
                empty_or_incomplete_skipped += 1
                total_ignored += ignored_n
                continue

            img_hash = compute_md5(img_path)
            if img_hash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(img_hash)

            dst_img = os.path.join(out_dir, 'images', split, f"{base_name}.jpg")
            dst_lbl = os.path.join(out_dir, 'labels', split, f"{base_name}.txt")

            if use_clahe:
                apply_clahe(img_path, dst_img, target_size=(640, 640), clip_limit=clip_limit)
            else:
                shutil.copy2(img_path, dst_img)

            with open(dst_lbl, 'w', encoding='utf-8') as fw:
                fw.writelines(new_lines)

            total_ignored += ignored_n
            for cid, cnt in kept_dict.items():
                class_counts[cid] += cnt
                ppe_split_boxes[split][cid] += cnt

            split_counts[split] += 1
            source_counts['construction_ppe'] += 1

    # 2. Ingestão da Base 2: Roboflow Construction PPE (vikasaarna/roboflow-construction-ppe-fruwx)
    rf_split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    if os.path.exists(src_rf):
        for rf_split, target_split in rf_split_map.items():
            img_dir = os.path.join(src_rf, rf_split, 'images')
            lbl_dir = os.path.join(src_rf, rf_split, 'labels')
            if not os.path.exists(img_dir):
                continue

            arquivos_rf = sorted(glob.glob(os.path.join(img_dir, '*.*')))
            print(f"  [*] Processando Base 2 (Roboflow PPE)     - partição '{rf_split} -> {target_split}' ({len(arquivos_rf)} arquivos)...")

            for img_path in arquivos_rf:
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    continue

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                src_lbl = os.path.join(lbl_dir, f"{base_name}.txt")
                new_lines, ignored_n, kept_dict = inspect_and_standardize_labels(src_lbl, ROBOFLOW_CLASS_MAPPING)

                # Filtra imagens da Base 2 que ficaram sem anotações válidas ou sem a classe Person (evita falsos negativos)
                if filtrar_incompletas_rf and (len(new_lines) == 0 or kept_dict[2] == 0):
                    empty_or_incomplete_skipped += 1
                    total_ignored += ignored_n
                    continue
                elif len(new_lines) == 0:
                    empty_or_incomplete_skipped += 1
                    total_ignored += ignored_n
                    continue

                img_hash = compute_md5(img_path)
                if img_hash in seen_hashes:
                    duplicates_removed += 1
                    continue
                seen_hashes.add(img_hash)

                out_name = f"rf_{base_name}"
                dst_img = os.path.join(out_dir, 'images', target_split, f"{out_name}.jpg")
                dst_lbl = os.path.join(out_dir, 'labels', target_split, f"{out_name}.txt")

                if use_clahe:
                    apply_clahe(img_path, dst_img, target_size=(640, 640), clip_limit=clip_limit)
                else:
                    shutil.copy2(img_path, dst_img)

                with open(dst_lbl, 'w', encoding='utf-8') as fw:
                    fw.writelines(new_lines)

                total_ignored += ignored_n
                for cid, cnt in kept_dict.items():
                    class_counts[cid] += cnt
                    rf_split_boxes[target_split][cid] += cnt

                split_counts[target_split] += 1
                source_counts['roboflow_safety'] += 1
    else:
        print(f"  [!] AVISO: Base 2 não localizada em {src_rf}.")

    # 3. Ingestão da Base 3: Chapian Construction Site PPE Detection (Nativa YOLO .txt - 4 Classes)
    ch_split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    if os.path.exists(src_ch):
        for ch_split, target_split in ch_split_map.items():
            img_dir = os.path.join(src_ch, ch_split, 'images')
            lbl_dir = os.path.join(src_ch, ch_split, 'labels')
            if not os.path.exists(img_dir):
                continue

            arquivos_ch = sorted(glob.glob(os.path.join(img_dir, '*.*')))
            print(f"  [*] Processando Base 3 (Chapian PPE)      - partição '{ch_split} -> {target_split}' ({len(arquivos_ch)} arquivos)...")

            for img_path in arquivos_ch:
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    continue

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                src_lbl = os.path.join(lbl_dir, f"{base_name}.txt")
                new_lines, ignored_n, kept_dict = inspect_and_standardize_labels(src_lbl, CHAPIAN_CLASS_MAPPING)

                if len(new_lines) == 0 or (filtrar_incompletas_rf and kept_dict[2] == 0):
                    empty_or_incomplete_skipped += 1
                    total_ignored += ignored_n
                    continue

                img_hash = compute_md5(img_path)
                if img_hash in seen_hashes:
                    duplicates_removed += 1
                    continue
                seen_hashes.add(img_hash)

                out_name = f"ch_{base_name}"
                dst_img = os.path.join(out_dir, 'images', target_split, f"{out_name}.jpg")
                dst_lbl = os.path.join(out_dir, 'labels', target_split, f"{out_name}.txt")

                if use_clahe:
                    apply_clahe(img_path, dst_img, target_size=(640, 640), clip_limit=clip_limit)
                else:
                    shutil.copy2(img_path, dst_img)

                with open(dst_lbl, 'w', encoding='utf-8') as fw:
                    fw.writelines(new_lines)

                total_ignored += ignored_n
                for cid, cnt in kept_dict.items():
                    class_counts[cid] += cnt
                    ch_split_boxes[target_split][cid] += cnt

                split_counts[target_split] += 1
                source_counts['chapian_ppe'] += 1
    else:
        print(f"  [!] AVISO: Base 3 não localizada em {src_ch}.")

    # 4. Ingestão da Base 4: SH17 Dataset - Human & PPE Detection (Nativa YOLO .txt - 4 Classes)
    sh17_split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    if os.path.exists(src_sh17):
        for sh_split, target_split in sh17_split_map.items():
            img_dir = os.path.join(src_sh17, sh_split, 'images')
            lbl_dir = os.path.join(src_sh17, sh_split, 'labels')
            if not os.path.exists(img_dir):
                continue

            arquivos_sh = sorted(glob.glob(os.path.join(img_dir, '*.*')))
            print(f"  [*] Processando Base 4 (SH17 PPE)         - partição '{sh_split} -> {target_split}' ({len(arquivos_sh)} arquivos)...")

            for img_path in arquivos_sh:
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    continue

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                src_lbl = os.path.join(lbl_dir, f"{base_name}.txt")
                new_lines, ignored_n, kept_dict = inspect_and_standardize_labels(src_lbl, SH17_CLASS_MAPPING)

                if len(new_lines) == 0 or (filtrar_incompletas_rf and kept_dict[2] == 0):
                    empty_or_incomplete_skipped += 1
                    total_ignored += ignored_n
                    continue

                img_hash = compute_md5(img_path)
                if img_hash in seen_hashes:
                    duplicates_removed += 1
                    continue
                seen_hashes.add(img_hash)

                out_name = f"sh17_{base_name}"
                dst_img = os.path.join(out_dir, 'images', target_split, f"{out_name}.jpg")
                dst_lbl = os.path.join(out_dir, 'labels', target_split, f"{out_name}.txt")

                if use_clahe:
                    apply_clahe(img_path, dst_img, target_size=(640, 640), clip_limit=clip_limit)
                else:
                    shutil.copy2(img_path, dst_img)

                with open(dst_lbl, 'w', encoding='utf-8') as fw:
                    fw.writelines(new_lines)

                total_ignored += ignored_n
                for cid, cnt in kept_dict.items():
                    class_counts[cid] += cnt
                    sh17_split_boxes[target_split][cid] += cnt

                split_counts[target_split] += 1
                source_counts['sh17_ppe'] += 1
    else:
        print(f"  [!] AVISO: Base 4 não localizada em {src_sh17}.")

    yaml_data = {
        'path': out_dir.replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 4,
        'names': {
            0: 'Hardhat',
            1: 'Safety-Vest',
            2: 'Person',
            3: 'Safety-Shoes'
        }
    }
    yaml_path = os.path.join(out_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    local_code_yaml = os.path.join(pasta_script, 'data.yaml')
    with open(local_code_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    tot_ppe_boxes = sum(sum(d.values()) for d in ppe_split_boxes.values())
    tot_rf_boxes = sum(sum(d.values()) for d in rf_split_boxes.values())
    tot_ch_boxes = sum(sum(d.values()) for d in ch_split_boxes.values())
    tot_sh17_boxes = sum(sum(d.values()) for d in sh17_split_boxes.values())

    print("\n" + "=" * 88)
    print("[+] DATASET CONSOLIDADO (UNIFIED_TCC) PREPARADO COM SUCESSO")
    print("=" * 88)
    print(f"    - Base 1 (Principal):      Ultralytics Construction-PPE ({source_counts['construction_ppe']} imagens | {tot_ppe_boxes} caixas)")
    print(f"    - Base 2 (Complementar 1): Roboflow Construction PPE    ({source_counts['roboflow_safety']} imagens | {tot_rf_boxes} caixas)")
    print(f"    - Base 3 (Complementar 2): Chapian Construction PPE     ({source_counts['chapian_ppe']} imagens | {tot_ch_boxes} caixas)")
    print(f"    - Base 4 (Complementar 3): SH17 PPE Dataset             ({source_counts['sh17_ppe']} imagens | {tot_sh17_boxes} caixas)")
    print(f"    - Diretório Consolidado:   {out_dir}")
    print(f"    - Partições (Total: {sum(split_counts.values())} imagens):")
    print(f"        * Treino (train):      {split_counts['train']} imagens")
    print(f"        * Validação (val):     {split_counts['val']} imagens")
    print(f"        * Teste (test):        {split_counts['test']} imagens")
    print(f"    - Duplicatas MD5 removidas:             {duplicates_removed}")
    print(f"    - Imagens órfãs/incompletas filtradas:  {empty_or_incomplete_skipped}")
    print(f"    - Instâncias Válidas Retidas (4 Classes NR-06 / NR-18):")
    print(f"        * 0: Hardhat (Capacete):       {class_counts[0]}")
    print(f"        * 1: Safety-Vest (Colete):     {class_counts[1]}")
    print(f"        * 2: Person (Trabalhador):     {class_counts[2]}")
    print(f"        * 3: Safety-Shoes (Calçados):  {class_counts[3]}")
    print(f"    - Total de Bounding Boxes Válidas: {sum(class_counts.values())} (Descartadas fora de escopo: {total_ignored})")
    print(f"    - Manifesto YAML gerado em:        {yaml_path}")
    print(f"    - Manifesto local sincronizado em: {local_code_yaml}")
    print("=" * 88 + "\n")
    return yaml_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Engenharia de Dados (Construction-PPE + Roboflow + Chapian + SH17 + CLAHE)")
    parser.add_argument("--apply-clahe", action="store_true", default=True, help="Aplica redimensionamento 640x640 e CLAHE no canal L (CIE LAB) (ativo por padrão)")
    parser.add_argument("--no-clahe", action="store_true", help="Desativa a aplicação do filtro CLAHE")
    parser.add_argument("--clip-limit", type=float, default=1.5, help="Limite de corte (clipLimit) do CLAHE (padrão suave: 1.5)")
    parser.add_argument("--manter-incompletas-rf", action="store_true", help="Mantém todas as imagens da Base 2 mesmo sem anotação de Person")
    parser.add_argument("--no-pause", action="store_true", help="Não pausa o terminal ao finalizar")
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        usar_clahe = False if args.no_clahe else args.apply_clahe
        build_unified_dataset(
            use_clahe=usar_clahe,
            clip_limit=args.clip_limit,
            filtrar_incompletas_rf=not args.manter_incompletas_rf,
        )
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 80)
        print("  [ERRO EM preparar_dados.py]")
        print("!" * 80)
        print(f"  -> Tipo     : {type(exc).__name__}")
        print(f"  -> Mensagem : {exc}")
        traceback.print_exc()
        print("!" * 80)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
