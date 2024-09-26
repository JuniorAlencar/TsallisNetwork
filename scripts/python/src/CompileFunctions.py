import compileall
from pathlib import Path
from multiprocessing import Pool, cpu_count
import zipfile
import os

# Função para compilar um arquivo
def compile_file(source_file):
    # Compilar o arquivo, colocando o .pyc no diretório padrão (__pycache__)
    print(f"Compilando {source_file}...")
    result = compileall.compile_file(source_file, quiet=1)
    
    if result:
        # Retornar o caminho do arquivo compilado no diretório __pycache__
        pycache_dir = source_file.parent / "__pycache__"
        pyc_files = list(pycache_dir.glob(f"{source_file.stem}*.pyc"))
        if pyc_files:
            return pyc_files[0]  # Retorna o primeiro arquivo .pyc encontrado
    return None

# Função principal para compilar arquivos de uma pasta e criar o zip
def compile_and_zip(source_dir, zip_filepath):
    # Lista de todos os arquivos .py na pasta
    files_to_compile = [Path(root) / file 
                        for root, dirs, files in os.walk(source_dir) 
                        for file in files if file.endswith('.py')]

    # Usando multiprocessing para compilar em paralelo
    with Pool(cpu_count()) as pool:
        compiled_files = pool.map(compile_file, files_to_compile)

    print("Compilação completa!")

    # Criar o arquivo zip com os arquivos compilados
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for compiled_file in compiled_files:
            if compiled_file:
                zipf.write(compiled_file, compiled_file.relative_to(source_dir.parent))
    
    print(f"Arquivo ZIP criado: {zip_filepath}")
