import os
import sys
import argparse

def download_dataset(api_key, workspace="roboflow-universe-projects", project_name="construction-site-safety", version_num=1, output_dir=None):
    """
    Realiza o download automatizado do dataset Construction Site Safety via API oficial do Roboflow.
    """
    if not output_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'datasets', 'raw', 'roboflow')
        
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  DOWNLOAD DO DATASET: ROBOFLOW CONSTRUCTION SITE SAFETY")
    print("=" * 70)
    print(f"[*] Workspace: {workspace}")
    print(f"[*] Projeto:   {project_name}")
    print(f"[*] Versão:    {version_num}")
    print(f"[*] Destino:   {output_dir}")
    print("=" * 70)
    
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[-] ERRO: A biblioteca 'roboflow' não está instalada.")
        print("[!] Instale executando: pip install roboflow")
        return False
        
    if not api_key:
        api_key = os.environ.get('ROBOFLOW_API_KEY')
        
    if not api_key:
        print("\n[-] ERRO: Nenhuma chave de API (API Key) fornecida.")
        print("\n[!] Como obter sua chave gratuita do Roboflow:")
        print("    1. Acesse: https://app.roboflow.com (crie uma conta gratuita)")
        print("    2. Vá em: Settings (Engrenagem no topo) -> Roboflow API")
        print("    3. Copie a sua 'Private API Key'")
        print("    4. Execute este script informando a chave:")
        print("       python src/download_roboflow.py --api-key SUA_CHAVE_AQUI")
        print("    Ou defina a variável de ambiente: set ROBOFLOW_API_KEY=sua_chave")
        return False
        
    print("[*] Autenticando com a API do Roboflow...")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_num)
        
        print(f"[*] Baixando dataset versão {version_num} no formato YOLOv8...")
        old_cwd = os.getcwd()
        try:
            os.chdir(output_dir)
            dataset = version.download("yolov8")
            print(f"\n[+] Sucesso! Dataset baixado em: {dataset.location}")
        finally:
            os.chdir(old_cwd)
            
        return True
    except Exception as e:
        print(f"\n[-] Falha ao baixar dataset do Roboflow: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download do Dataset Construction Site Safety do Roboflow")
    parser.add_argument("--api-key", type=str, default="", help="vLRXqZMSSf2J5tco2FLg")
    parser.add_argument("--version", type=int, default=1, help="Versão do dataset no Roboflow (padrão: 1)")
    parser.add_argument("--workspace", type=str, default="roboflow-universe-projects", help="Workspace no Roboflow")
    parser.add_argument("--project", type=str, default="construction-site-safety", help="Nome do projeto no Roboflow")
    args = parser.parse_args()
    
    download_dataset(
        api_key=args.api_key,
        workspace=args.workspace,
        project_name=args.project,
        version_num=args.version
    )
