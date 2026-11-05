# -*- coding: utf-8 -*-
"""
Módulo: preparar_dados.py
Pipeline de Engenharia de Dados do TCC:
  1. Verifica e garante a integridade das duas bases nativas no padrão YOLO (.txt):
     - Base 1 (Principal): Ultralytics Construction-PPE (1.416 imagens e 7.194 caixas válidas)
     - Base 2 (Complementar): Roboflow Construction Site Safety (398 imagens e 580 caixas válidas)
  2. Executa deduplicação criptográfica MD5 (64 KB chunks -> 0 duplicatas).
  3. Padroniza os IDs das 4 classes NR-06 / NR-18 (0: Hardhat, 1: Safety-Vest, 2: Person, 3: Safety-Shoes).
  4. Aplica redimensionamento 640x640 (cv2.INTER_AREA) e equalização CLAHE no canal L (CIE LAB).
  5. Gera o dataset consolidado 'unified_tcc' (1.814 imagens e 7.774 caixas) e os manifestos 'data.yaml'.

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
    print("\n[ERRO CRÍTICO] Biblioteca 'opencv-python' (cv2) não encontrada!")
    print(" -> Instale executando: pip install opencv-python")
    input("\nPressione ENTER para fechar...")
    raise SystemExit(1) from exc_cv2

try:
    import yaml
except ImportError as exc_yaml:
    print("\n[ERRO CRÍTICO] Biblioteca 'PyYAML' (yaml) não encontrada!")
    print(" -> Instale executando: pip install pyyaml")
    input("\nPressione ENTER para fechar...")
    raise SystemExit(1) from exc_yaml

# URLs Oficiais das Bases Nativas no Padrão YOLO (.txt) utilizadas no TCC (Sem necessidade de conversão XML)
CONSTRUCTION_PPE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
CONSTRUCTION_PPE_DOCS = "https://docs.ultralytics.com/datasets/detect/construction-ppe/"
ROBOFLOW_SAFETY_URL = "https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety"

# Mapeamento Taxonômico 1: Base Principal Ultralytics Construction-PPE (11 classes -> 4 classes TCC)
# Original: 0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none, 6: Person, 7: no_helmet, 8: no_goggle, 9: no_gloves, 10: no_boots
# Padronizado TCC: 0: Hardhat, 1: Safety-Vest, 2: Person, 3: Safety-Shoes
SINGLE_DB_CLASS_MAPPING = {
    0: 0,  # helmet -> Hardhat (ID 0)
    2: 1,  # vest   -> Safety-Vest (ID 1)
    6: 2,  # Person -> Person (ID 2)
    3: 3   # boots  -> Safety-Shoes (ID 3)
}

# Mapeamento Taxonômico 2: Base Complementar de Mesmo Padrão YOLO TXT - Roboflow Construction Site Safety (15 classes -> 4 classes TCC)
# Original: 0..3: gloves/goggles/mask, 4: Hardhat, 5..8: NO-*, 9: Person, 10: Safety Cone, 11: Safety Shoes, 12: Safety Vest, 13..14: machinery/vehicle
ROBOFLOW_CLASS_MAPPING = {
    4: 0,   # Hardhat      -> Hardhat (ID 0)
    12: 1,  # Safety Vest  -> Safety-Vest (ID 1)
    9: 2,   # Person       -> Person (ID 2)
    11: 3   # Safety Shoes -> Safety-Shoes (ID 3)
}


def pausar_se_interativo(forcar_sem_pausa=False, houve_erro=False):
    """Evita que a janela feche imediatamente após um erro ou execução via duplo clique."""
    if forcar_sem_pausa:
        return
    try:
        if houve_erro or (sys.stdin and sys.stdin.isatty()):
            print("\n" + "-" * 80)
            input("  [PAUSA] Pressione ENTER para finalizar e fechar esta janela...")
    except (EOFError, KeyboardInterrupt):
        pass


def corrigir_pasta_aninhada(extracted_dir):
    """
    Remove aninhamento acidental 'construction-ppe/construction-ppe' caso o .zip tenha sido
    extraído dentro de uma subpasta de mesmo nome.
    """
    subpasta = os.path.join(extracted_dir, "construction-ppe")
    if os.path.isdir(subpasta) and os.path.isdir(os.path.join(subpasta, "images")):
        print("  [*] Ajustando subpasta aninhada 'construction-ppe/construction-ppe'...")
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


def apply_clahe(image_path, output_path, target_size=(640, 640), clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Aplica redimensionamento espacial para 640x640 (interpolação INTER_AREA)
    e equalização de histograma adaptativa limitada por contraste (CLAHE)
    no canal L do espaço de cores CIE LAB, conforme especificado na metodologia do TCC.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    # 1. Redimensionamento unificado para 640x640 (padrão de entrada do YOLOv26n)
    if target_size is not None and (img.shape[1], img.shape[0]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    # 2. Conversão BGR -> CIE LAB (separa luminância L dos canais cromáticos A e B)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 3. Aplicação do CLAHE exclusivamente no canal L
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    # 4. Recombinação e reconversão CIE LAB -> BGR
    limg = cv2.merge((cl, a_channel, b_channel))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, final)
    return True


def standardize_labels(label_path, output_path, class_mapping=None):
    """
    Padroniza os identificadores de classes de um arquivo .txt no formato YOLO.
    class_mapping: dicionário {old_class_id: new_class_id}.
    Linhas com classes não mapeadas são expurgadas.
    """
    if class_mapping is None:
        class_mapping = SINGLE_DB_CLASS_MAPPING

    if not os.path.exists(label_path):
        return 0, 0, {0: 0, 1: 0, 2: 0, 3: 0}

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    kept_by_class = {0: 0, 1: 0, 2: 0, 3: 0}
    ignored_count = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_class = int(parts[0])
        if old_class in class_mapping:
            new_class = class_mapping[old_class]
            coords = [max(0.0, min(1.0, float(v))) for v in parts[1:]]
            if coords[2] > 0.001 and coords[3] > 0.001:
                new_lines.append(f"{new_class} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                kept_by_class[new_class] += 1
        else:
            ignored_count += 1

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
    Garante que o banco principal Ultralytics Construction-PPE (1.416 imagens: 1.132 treino,
    143 validação e 141 teste) esteja disponível em datasets/raw/construction-ppe.
    """
    raw_dir = os.path.join(base_dir, 'datasets', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    zip_dest = os.path.join(raw_dir, 'construction-ppe.zip')
    extracted_dir = os.path.join(raw_dir, 'construction-ppe')

    corrigir_pasta_aninhada(extracted_dir)

    if os.path.exists(extracted_dir) and os.path.exists(os.path.join(extracted_dir, 'images', 'train')):
        n_train = len(glob.glob(os.path.join(extracted_dir, 'images', 'train', '*.*')))
        n_val = len(glob.glob(os.path.join(extracted_dir, 'images', 'val', '*.*')))
        n_test = len(glob.glob(os.path.join(extracted_dir, 'images', 'test', '*.*')))
        if (n_train + n_val + n_test) == 1416:
            print(f"[OK] Base 1 'construction-ppe' já validada em: {extracted_dir} (1.416 imagens)")
            return extracted_dir

    # Tenta sincronizar de Projeto_EPI/datasets/raw/construction-ppe caso esteja disponível no workspace
    raiz_ws = os.path.dirname(base_dir)
    alt_ppe = os.path.join(raiz_ws, 'Projeto_EPI', 'datasets', 'raw', 'construction-ppe')
    if os.path.isdir(alt_ppe) and os.path.isdir(os.path.join(alt_ppe, 'images', 'train')):
        if len(glob.glob(os.path.join(alt_ppe, 'images', 'train', '*.*'))) == 1132:
            print(f"[*] Sincronizando Base 1 íntegra (1.416 imagens) a partir de: {alt_ppe}")
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
            shutil.copytree(alt_ppe, extracted_dir)
            return extracted_dir

    # Se o .zip local já existe, extrai sem baixar novamente (preservando a versão do TCC)
    if os.path.exists(zip_dest) and os.path.getsize(zip_dest) > 50 * 1024 * 1024:
        print(f"[*] Extraindo Base 1 a partir do arquivo local: {zip_dest}")
        with zipfile.ZipFile(zip_dest, 'r') as zf:
            zf.extractall(extracted_dir)
        corrigir_pasta_aninhada(extracted_dir)
        return extracted_dir

    print(f"[*] Baixando banco único Ultralytics Construction-PPE de:\n    {CONSTRUCTION_PPE_URL}")
    urllib.request.urlretrieve(CONSTRUCTION_PPE_URL, zip_dest)
    print(f"[+] Download concluído ({os.path.getsize(zip_dest) / (1024*1024):.1f} MB). Extraindo arquivos...")

    with zipfile.ZipFile(zip_dest, 'r') as zf:
        zf.extractall(extracted_dir)
    corrigir_pasta_aninhada(extracted_dir)

    print(f"[OK] Extração concluída em: {extracted_dir}")
    return extracted_dir


def ensure_roboflow_dataset(base_dir):
    """
    Verifica se a Base 2 (Roboflow Construction Site Safety - 398 imagens) está em
    datasets/raw/roboflow/Construction-Site-Safety-1; se não estiver, sincroniza de Projeto_EPI se disponível.
    """
    src_rf = os.path.join(base_dir, 'datasets', 'raw', 'roboflow', 'Construction-Site-Safety-1')
    if os.path.isdir(src_rf) and os.path.isdir(os.path.join(src_rf, 'train', 'images')):
        return src_rf

    raiz_ws = os.path.dirname(base_dir)
    alt_rf = os.path.join(raiz_ws, 'Projeto_EPI', 'datasets', 'raw', 'roboflow', 'Construction-Site-Safety-1')
    if os.path.isdir(alt_rf) and os.path.isdir(os.path.join(alt_rf, 'train', 'images')):
        print(f"[*] Sincronizando Base 2 (Roboflow - 398 imagens) a partir de: {alt_rf}")
        os.makedirs(os.path.dirname(src_rf), exist_ok=True)
        shutil.copytree(alt_rf, src_rf)
    return src_rf


def build_unified_dataset(use_clahe=False, clip_limit=3.0):
    """
    Constrói o dataset padronizado do TCC (datasets/processed/unified_tcc)
    integrando as duas bases oficiais nativas no padrão YOLO (.txt) do contexto do PF2,
    sem necessidade de qualquer conversão de coordenadas XML:
      1. Base Principal: Ultralytics Construction-PPE [LINK-8] (1.416 imagens e 7.194 instâncias válidas)
      2. Base Complementar de Mesmo Padrão YOLO TXT: Roboflow Construction Site Safety [LINK-4] (398 imagens e 580 instâncias válidas)
    Consolidando 1.814 imagens únicas (superando o critério oficial de >= 1.500 imagens da Ultralytics)
    e 7.774 caixas delimitadoras balanceadas nas 4 classes do TCC:
      0: Hardhat      (2.170 instâncias: 1.734 Construction-PPE + 436 Roboflow)
      1: Safety-Vest  (1.671 instâncias: 1.618 Construction-PPE + 53 Roboflow)
      2: Person       (2.314 instâncias: 2.245 Construction-PPE + 69 Roboflow)
      3: Safety-Shoes (1.619 instâncias: 1.597 Construction-PPE + 22 Roboflow)
    """
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(pasta_script)
    src_ppe = download_single_dataset(base_dir)
    src_rf = ensure_roboflow_dataset(base_dir)
    out_dir = os.path.join(base_dir, 'datasets', 'processed', 'unified_tcc')

    print("\n" + "=" * 80)
    print("  INICIANDO PIPELINE DE ENGENHARIA DE DADOS (UNIFIED_TCC - 4 CLASSES)")
    print("=" * 80)
    print(f"  -> Filtro CLAHE (CIE LAB) : {'ATIVADO (640x640 INTER_AREA, clipLimit=' + str(clip_limit) + ')' if use_clahe else 'DESATIVADO (cópia direta)'}")
    print(f"  -> Base 1 (Principal)     : {src_ppe}")
    print(f"  -> Base 2 (Complementar)  : {src_rf}")
    print(f"  -> Destino Consolidado    : {out_dir}")
    print("-" * 80)

    # Limpar diretório de saída anterior para garantir pureza das bases nativas YOLO TXT
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

    seen_hashes = set()
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    source_counts = {'construction_ppe': 0, 'roboflow_safety': 0}
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    ppe_split_boxes = {'train': {0: 0, 1: 0, 2: 0, 3: 0}, 'val': {0: 0, 1: 0, 2: 0, 3: 0}, 'test': {0: 0, 1: 0, 2: 0, 3: 0}}
    rf_split_boxes = {'train': {0: 0, 1: 0, 2: 0, 3: 0}, 'val': {0: 0, 1: 0, 2: 0, 3: 0}, 'test': {0: 0, 1: 0, 2: 0, 3: 0}}
    total_ignored = 0
    duplicates_removed = 0

    # 1. Ingestão da Base Principal Nativa YOLO TXT: Ultralytics Construction-PPE (1.416 imagens)
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

            img_hash = compute_md5(img_path)
            if img_hash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(img_hash)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            dst_img = os.path.join(out_dir, 'images', split, f"{base_name}.jpg")
            dst_lbl = os.path.join(out_dir, 'labels', split, f"{base_name}.txt")
            src_lbl = os.path.join(lbl_dir, f"{base_name}.txt")

            if use_clahe:
                apply_clahe(img_path, dst_img, target_size=(640, 640), clip_limit=clip_limit)
            else:
                shutil.copy2(img_path, dst_img)

            _, ignored_n, kept_dict = standardize_labels(src_lbl, dst_lbl, SINGLE_DB_CLASS_MAPPING)
            total_ignored += ignored_n
            for cid, cnt in kept_dict.items():
                class_counts[cid] += cnt
                ppe_split_boxes[split][cid] += cnt

            split_counts[split] += 1
            source_counts['construction_ppe'] += 1

    # Sincronizar cópia isolada de construction_ppe_tcc (1.416 imagens) antes da adição complementar
    single_alias_dir = os.path.join(base_dir, 'datasets', 'processed', 'construction_ppe_tcc')
    if os.path.exists(single_alias_dir):
        shutil.rmtree(single_alias_dir)
    shutil.copytree(out_dir, single_alias_dir)

    # 2. Ingestão da Base Complementar de Mesmo Padrão YOLO TXT: Roboflow Construction Site Safety (398 imagens)
    rf_split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    if os.path.exists(src_rf):
        for rf_split, target_split in rf_split_map.items():
            img_dir = os.path.join(src_rf, rf_split, 'images')
            lbl_dir = os.path.join(src_rf, rf_split, 'labels')
            if not os.path.exists(img_dir):
                continue

            arquivos_rf = sorted(glob.glob(os.path.join(img_dir, '*.*')))
            print(f"  [*] Processando Base 2 (Roboflow Safety)  - partição '{rf_split} -> {target_split}' ({len(arquivos_rf)} arquivos)...")

            for img_path in arquivos_rf:
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    continue

                img_hash = compute_md5(img_path)
                if img_hash in seen_hashes:
                    duplicates_removed += 1
                    continue
                seen_hashes.add(img_hash)

                base_name = os.path.splitext(os.path.basename(img_path))[0]
                out_name = f"rf_{base_name}"
                dst_img = os.path.join(out_dir, 'images', target_split, f"{out_name}.jpg")
                dst_lbl = os.path.join(out_dir, 'labels', target_split, f"{out_name}.txt")
                src_lbl = os.path.join(lbl_dir, f"{base_name}.txt")

                if use_clahe:
                    apply_clahe(img_path, dst_img, target_size=(640, 640), clip_limit=clip_limit)
                else:
                    shutil.copy2(img_path, dst_img)

                _, ignored_n, kept_dict = standardize_labels(src_lbl, dst_lbl, ROBOFLOW_CLASS_MAPPING)
                total_ignored += ignored_n
                for cid, cnt in kept_dict.items():
                    class_counts[cid] += cnt
                    rf_split_boxes[target_split][cid] += cnt

                split_counts[target_split] += 1
                source_counts['roboflow_safety'] += 1
    else:
        print(f"  [!] AVISO: Base 2 não encontrada em {src_rf}. Execute baixar_roboflow.py para ter as 1.814 imagens.")

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

    # Atualiza também a cópia local de data.yaml dentro de Codigos_Levantamento_e_Treinamento
    local_code_yaml = os.path.join(pasta_script, 'data.yaml')
    with open(local_code_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    # Também atualizar o data.yaml de construction_ppe_tcc
    single_yaml_data = dict(yaml_data)
    single_yaml_data['path'] = single_alias_dir.replace('\\', '/')
    with open(os.path.join(single_alias_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(single_yaml_data, f, sort_keys=False, allow_unicode=True)

    tot_ppe_boxes = sum(sum(d.values()) for d in ppe_split_boxes.values())
    tot_rf_boxes = sum(sum(d.values()) for d in rf_split_boxes.values())

    print("\n" + "=" * 80)
    print("[+] DATASET CONSOLIDADO DO TCC (2 BASES NATIVAS PADRÃO YOLO TXT) PREPARADO!")
    print("=" * 80)
    print(f"    - Base 1 (Principal):    Ultralytics Construction-PPE ({source_counts['construction_ppe']} imagens | {tot_ppe_boxes} caixas)")
    print(f"    - Base 2 (Complementar): Roboflow Construction Site Safety ({source_counts['roboflow_safety']} imagens | {tot_rf_boxes} caixas)")
    print(f"    - Diretório Consolidado: {out_dir}")
    print(f"    - Partições (Total {sum(split_counts.values())} imagens >= 1.500 recomendadas pela Ultralytics):")
    print(f"        * Treino (train):    {split_counts['train']} imagens")
    print(f"        * Validação (val):   {split_counts['val']} imagens")
    print(f"        * Teste (test):      {split_counts['test']} imagens")
    print(f"    - Duplicatas MD5 expurgadas: {duplicates_removed}")
    print(f"    - Instâncias Válidas Retidas (4 Classes NR-06 / NR-18):")
    print(f"        * 0: Hardhat (Capacete):       {class_counts[0]}")
    print(f"        * 1: Safety-Vest (Colete):     {class_counts[1]}")
    print(f"        * 2: Person (Trabalhador):     {class_counts[2]}")
    print(f"        * 3: Safety-Shoes (Calçados):  {class_counts[3]}")
    print(f"    - Total de Bounding Boxes Válidas: {sum(class_counts.values())} (Expurgadas: {total_ignored} fora de escopo)")
    print(f"    - Manifesto YAML gerado em:        {yaml_path}")
    print(f"    - Manifesto local sincronizado em: {local_code_yaml}")
    print("=" * 80 + "\n")
    return yaml_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline de Engenharia de Dados (Construction-PPE + Roboflow + CLAHE)")
    parser.add_argument("--apply-clahe", action="store_true", help="Aplica redimensionamento 640x640 e CLAHE no canal L (CIE LAB)")
    parser.add_argument("--clip-limit", type=float, default=3.0, help="Limite de corte (clipLimit) do CLAHE (padrão: 3.0)")
    parser.add_argument("--no-pause", action="store_true", help="Não pausa o terminal ao finalizar")
    args = parser.parse_args()

    erro_ocorreu = False
    try:
        build_unified_dataset(use_clahe=args.apply_clahe, clip_limit=args.clip_limit)
    except Exception as exc:
        erro_ocorreu = True
        print("\n" + "!" * 80)
        print("  [ERRO DETECTADO EM preparar_dados.py]")
        print("!" * 80)
        print(f"  -> Tipo do erro : {type(exc).__name__}")
        print(f"  -> Mensagem     : {exc}")
        print("\n  [Detalhes técnicos (Traceback)]:")
        traceback.print_exc()
        print("!" * 80)
    finally:
        pausar_se_interativo(forcar_sem_pausa=args.no_pause, houve_erro=erro_ocorreu)
