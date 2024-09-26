from src.CompileFunctions import *

foldername = 'BoltzmannMachine'
# Caminho de origem (pasta a ser compilada)
source_dir = Path(f'~/Documents/{foldername}').expanduser()

# Caminho do arquivo zip de destino
zip_filepath = Path(f'~/Documents/{foldername}.zip').expanduser()

# Compilar e criar o arquivo ZIP
compile_and_zip(source_dir, zip_filepath)
