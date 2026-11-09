# Registro Experimental de Treinamento e Otimização (TCC2 - YOLOv26n)

**Projeto:** PF2 - Sistema de Monitoramento de Uso de EPI em Construção Civil (NR-06 / NR-18)  
**Arquitetura Avaliada:** `YOLOv26n` (`yolo26n.pt` / `yolo26n.yaml` — 260 camadas, 2.505.360 parâmetros, 5,9 GFLOPs)  
**Ambiente de Execução:** Google Colab — GPU NVIDIA Tesla T4 (15 GB VRAM), PyTorch 2.14.1+cu130, Ultralytics 8.4.208  
**Classes Monitoradas (`nc = 4`):** `0: Hardhat` (Capacete), `1: Safety-Vest` (Colete), `2: Person` (Trabalhador), `3: Safety-Shoes` (Calçado de Segurança)

---

## 1. Experimento 1 — Modelo Baseline Consolidado (`ppe_detection_model`)

### 1.1. Configuração do Dataset Consolidado (`unified_tcc` — Modo Amplo)
- **Fontes Integradas (4 Bases Nativas YOLO `.txt`):**
  1. **Base 1 (Principal):** Ultralytics Construction-PPE (`1.403` imagens válidas | `9.775` caixas)
  2. **Base 2 (Complementar 1):** Roboflow Construction PPE `ppe-fruwx` (`2.631` imagens únicas com `Person` | `16.012` caixas)
  3. **Base 3 (Complementar 2):** Chapian Construction Site PPE (`187` imagens únicas com `Person` | `1.544` caixas)
  4. **Base 4 (Complementar 3):** SH17 PPE Dataset (`313` imagens completas | `2.519` caixas)
- **Particionamento Total (`4.534` imagens únicas | `29.850` caixas delimitadoras):**
  - **Treino (`train`):** `3.516` imagens
  - **Validação (`val`):** `602` imagens (`4.243` instâncias)
  - **Teste (`test`):** `416` imagens
- **Distribuição por Classe (`29.850` instâncias):**
  - `0: Hardhat`: `7.509`
  - `1: Safety-Vest`: `5.817`
  - `2: Person`: `9.150`
  - `3: Safety-Shoes`: `7.374`
- **Pré-processamento e Hiperparâmetros (Experimento 1):**
  - Resolução espacial: `640x640` (`INTER_AREA`) + CLAHE no canal $L$ (CIE LAB, `clipLimit = 1.5`)
  - Épocas: `100` (concluído em `1,672` horas na Tesla T4, pico de VRAM `3,21 GB`)
  - Otimizador: `AdamW` (`lr = 0.00125`, `momentum = 0.9`, `cos_lr = True`)
  - Aumentações: `mosaic = 1.0`, `close_mosaic = 20`, `erasing = 0.0`, `copy_paste = 0.0`, `mixup = 0.0`

### 1.2. Resultados Quantitativos Obtidos no Experimento 1 (`weights/best.pt`)

| Classe | Imagens (`val`) | Instâncias | Precision ($P$) | Recall ($R$) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`all` (Média Global)** | **602** | **4.243** | **84,2%** (`0.842`) | **75,3%** (`0.753`) | **81,0%** (`0.810`) | **54,4%** (`0.544`) |
| `0: Hardhat` (Capacete) | 510 | 1.082 | **82,5%** (`0.825`) | **83,5%** (`0.835`) | **86,1%** (`0.861`) | **52,4%** (`0.524`) |
| `1: Safety-Vest` (Colete) | 432 | 817 | **84,5%** (`0.845`) | **75,8%** (`0.758`) | **81,4%** (`0.814`) | **57,1%** (`0.571`) |
| `2: Person` (Trabalhador) | 602 | 1.289 | **89,6%** (`0.896`) | **91,6%** (`0.916`) | **94,5%** (`0.945`) | **69,5%** (`0.695`) |
| `3: Safety-Shoes` (Botina) | 385 | 1.055 | **80,3%** (`0.803`) | **50,5%** (`0.505`) | **62,2%** (`0.622`) | **38,7%** (`0.387`) |

- **Velocidade de Inferência (Tesla T4):** `0,2 ms` pré-processamento + `4,4 ms` inferência + `1,7 ms` pós-processamento (`~158 FPS`).

---

## 2. Diagnóstico Técnico e Proposta do Experimento 2 (`ppe_detection_model_v2`)

### 2.1. Causa-Raiz do Recall Moderado em `Safety-Shoes` (50,5%) no Experimento 1
Uma auditoria automatizada sobre as anotações originais das 4 bases revelou dois fatores limitantes para a classe `Safety-Shoes` (`ID 3`):
1. **Falsos Negativos de Calçados na Base 2 (`Roboflow ppe-fruwx`):** Das `2.631` imagens retidas da Base 2 no Experimento 1, **1.138 imagens** possuíam trabalhadores de corpo inteiro (`Person` anotado), mas **não possuíam nenhuma anotação nos pés** (nem `boots` e nem `no boots`). Durante o treinamento, essas 1.138 imagens ensinaram o detector a suprimir predições de botinas reais (gerando alta precisão de `80,3%`, porém revocação de `50,5%`).
2. **Escala Espacial Reduzida (`640x640`):** Como calçados ocupam menos de $1\%$ da área total do quadro, a resolução `640x640` reduz botinas distantes para caixas de apenas `10x12` pixels.

### 2.2. Melhorias Implementadas no Experimento 2 (Opções 1 e 2)
1. **Opção 2 — Filtro Anti-Falso-Negativo de Calçados (`Possui_Falso_Negativo_Calcado_RF` em `preparar_dados.py`):**
   - Remove da Base 2 exclusivamente as imagens em que a anotação dos pés foi omitida pelo autor original (imagens sem `boots` e sem `no boots` explícito).
   - Preserva **100% das instâncias reais de `Safety-Shoes` (`7.374` caixas)** enquanto elimina 1.138 imagens com falsos negativos de botinas, consolidando **3.396 imagens únicas (`2.611` treino, `461` validação e `324` teste)** e **24.417 caixas delimitadoras** altamente balanceadas:
     - `0: Hardhat`: `5.560` caixas
     - `1: Safety-Vest`: `4.595` caixas
     - `2: Person`: `6.888` caixas
     - `3: Safety-Shoes`: `7.374` caixas
2. **Opção 1 — Aumento de Resolução (`imgsz = 800`) + Aumentações para Objetos Pequenos (`copy_paste = 0.3`, `mixup = 0.1`):**
   - A resolução `800x800` aumenta em **+56,25% a área em pixels** de objetos pequenos (`Safety-Shoes` e `Hardhat`), utilizando apenas ~5,0 GB dos 15 GB disponíveis na GPU Tesla T4.
   - A aumentação `copy_paste = 0.3` e `mixup = 0.1` multiplica a frequência de exposição das instâncias menores durante as primeiras 80 épocas.
3. **Preservação Integral do Experimento 1:**
   - O treinamento do Experimento 2 salva seus artefatos em `models/ppe_detection_model_v2`, mantendo intacta a pasta `models/ppe_detection_model` (ou seu backup `models/backup_exp1_81mAP`).

---

## 3. Comandos de Backup e Execução no Google Colab

### Passo A: Fazer Backup Seguro do Experimento 1 (`81,0% mAP@50`) no Colab
Execute antes de iniciar o novo experimento para garantir que os pesos, gráficos e CSV do Experimento 1 fiquem preservados:

```bash
# 1. Criar cópia de segurança da pasta do Experimento 1 (81.0% mAP50) e compactar em .zip
!cp -r /content/TCC-Sistema-de-monitoramento-de-EPI/models/ppe_detection_model /content/TCC-Sistema-de-monitoramento-de-EPI/models/backup_exp1_81mAP
!zip -r /content/backup_exp1_81mAP.zip /content/TCC-Sistema-de-monitoramento-de-EPI/models/ppe_detection_model
```
*(Você também pode baixar `/content/backup_exp1_81mAP.zip` pela aba lateral de arquivos do Colab).*

### Passo B: Executar o Experimento 2 (Opções 1 e 2 Ativadas)
Após atualizar os scripts `preparar_dados.py`, `gerenciar_bases.py` e `treinar_modelo.py` no Colab:

```bash
# 1. Reconstruir o dataset unified_tcc com o Filtro Anti-Falso-Negativo de Botinas (Opção 2) e resolução 800x800 (Opção 1)
!python preparar_dados.py --apply-clahe --imgsz 800 --no-pause

# 2. Verificar o relatório de status das bases e distribuição balanceada
!python gerenciar_bases.py --action status --no-pause

# 3. Treinar o Experimento 2 (imgsz=800, copy_paste=0.3, mixup=0.1) salvando em models/ppe_detection_model_v2
!python treinar_modelo.py --epochs 100 --imgsz 800 --batch 16 --copy-paste 0.3 --mixup 0.1 --device 0 --name ppe_detection_model_v2 --no-pause
```

### Passo C (Reversibilidade): Como Reproduzir Exatamente o Experimento 1 Quando Desejar
Caso queira regerar o dataset e o treino idênticos ao Experimento 1 (`640x640`, sem o filtro de calçados da Base 2):

```bash
!python preparar_dados.py --apply-clahe --imgsz 640 --manter-sem-calcado-rf --no-pause
!python treinar_modelo.py --epochs 100 --imgsz 640 --batch 16 --copy-paste 0.0 --mixup 0.0 --device 0 --name ppe_detection_model_exp1 --no-pause
```
