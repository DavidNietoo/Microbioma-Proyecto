import subprocess
import os

def run_metaphlan_analysis(input_file):
    # Para obtener el nombre del folder y del archivo
    folder_name = os.path.basename(os.path.dirname(input_file))
    file_name = os.path.basename(input_file)
    
    # path necesario para el input en WSL y que se pueda utilizar en metaphlan
    input_folder = f"/mnt/c/Users/david/OneDrive/Escritorio/Proyectos/Microbioma-Proyecto/data/{folder_name}"
    input_file_path = f"{input_folder}/{file_name}"
    
    # Determinar el tipo de archivo de entrada
    if input_file.endswith('.bz2'):
        input_type = "bowtie2out"
    elif input_file.endswith('.fasta'):
        input_type = "fasta"
    else:
        input_type = "fastq"

    output_filename = f"{os.path.splitext(file_name)[0]}_results.txt"
    output_file = f"{input_folder}/{output_filename}"
    
    metaphlan_command = f"metaphlan {input_file_path} --input_type {input_type} -o {output_file}"
    
    comandos = [
        "cd /mnt/c/Users/david/OneDrive/Escritorio/Proyectos/Microbioma-Proyecto",
        "source api/scripts/activate",
        f"echo 'Running command: {metaphlan_command}'",
        metaphlan_command
    ]

    try:
        full_command = ["wsl", "/bin/bash", "-c", " && ".join(comandos)]
        process = subprocess.Popen(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Captura el stdout como stderr
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_message = f"Error executing the command. Return code: {process.returncode}\n"
            error_message += f"STDOUT:\n{stdout}\n"
            error_message += f"STDERR:\n{stderr}"
            print(error_message)
            return None, error_message
        else:
            print(f"Analysis completed. Results saved in {output_file}")
            # convierte la ruta de WSL a Windows
            windows_output_file = os.path.normpath(f"C:\\Users\\david\\OneDrive\\Escritorio\\Proyectos\\Microbioma-Proyecto\\data\\{folder_name}\\{output_filename}")
            return windows_output_file, None
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return None, error_message