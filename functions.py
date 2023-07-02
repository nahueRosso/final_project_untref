import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import json


def convert_bin(audio_path):
    waveform, sample_rate = librosa.load(audio_path)

    interval_duration = 0.1 
    
    interval_samples = int(interval_duration * sample_rate)

    total_intervals = len(waveform) // interval_samples

    energies = []
    for i in range(total_intervals):
        start = i * interval_samples
        end = (i + 1) * interval_samples
        interval_waveform = waveform[start:end]
        energy = sum(interval_waveform**2)
        energies.append(energy)

  
    threshold = 0.1  

    is_active = [energy > threshold for energy in energies]

    on_off = []
    
    for i, active in enumerate(is_active):
        
        if active:
            
            on_off.append(1)
            
        else:
            on_off.append(0)

    on_off = np.array(on_off)

    return on_off

def get_dataframes(parent_folder_path,texto_or_audio,pre_path='.'):
    
    carpeta2013_2015 = f'{pre_path}/databases_morsecode/{parent_folder_path}/'

    files = os.listdir(carpeta2013_2015)

    # print(archivos)

    data = []

    for file in files:
        in_carpeta2013_2015 = f'{pre_path}/databases_morsecode/{parent_folder_path}/{file}' 
        archivos_mp3 = glob.glob(os.path.join(in_carpeta2013_2015, '*.mp3'))
        archivos_txt = glob.glob(os.path.join(in_carpeta2013_2015, '*.txt'))

        for archivo_mp3,archivo_txt in zip(archivos_mp3,archivos_txt):
            data.append({'audio': archivo_mp3, 'texto': archivo_txt})
        # print(files_in_folder

    df = pd.DataFrame(data)
    
    option = list(np.array(df[f'{texto_or_audio}']))
    
    return option 

def get_all_texts_for_dic(list_files_txt):
    frases = []

    print(len(list_files_txt))

    for archivo in list_files_txt:
        with open(archivo, 'r') as file:
            contenido = file.read()
            frases.append(contenido)
            file.close()

    frases_array = []
    
    for frase in frases:
        frase_modificada = frase.replace('\n\n', '').replace('\n', ' ')
        frases_array.append(frase_modificada)
        
    if clean_texts(frases_array) == None:
        all_texts_clean = frases_array
    elif clean_texts(frases_array) != None:
        all_texts_clean = clean_texts(frases_array)
     
    diccionario = {}

    for path,content in zip(list_files_txt,all_texts_clean):
     
        diccionario[os.path.basename(path)] = content
 

    
    return diccionario

def clean_texts(array_texts):
    for i, elementos in enumerate(array_texts):
        contador = 0
        contador_2 = 0
        for j, elemento in enumerate(elementos[:100]): 
            if '‰' in elemento:
                contador += elemento.count('‰')
            if '=' in elemento:
                contador_2 += elemento.count('=')
        if contador_2 > 0:
            contador = contador_2 + 10         

        if contador == 3:
            sym_1 = elementos[:100].index('‰')
            sym_2 = elementos[sym_1 + 1:100].index('‰')
            sym_3 = elementos[sym_1 + sym_2 + 2 + 1:100].index('‰')
            array_texts[i] = elementos[sym_1 + sym_2 + sym_3 + 4:]
        elif contador == 2:
            sym_1 = elementos[:100].index('‰')
            sym_2 = elementos[sym_1 + 1:100].index('‰')
            array_texts[i] = elementos[sym_1 + sym_2 + 2:]
        elif contador == 13:
            sym_1 = elementos[:100].index('=')
            sym_2 = elementos[sym_1 + 1:100].index('=')
            sym_3 = elementos[sym_1 + sym_2 + 2 + 1:100].index('=')
            array_texts[i] = elementos[sym_1 + sym_2 + sym_3 + 4:]
        elif contador == 12:
            sym_1 = elementos[:100].index('=')
            sym_2 = elementos[sym_1 + 1:100].index('=')
            array_texts[i] = elementos[sym_1 + sym_2 + 2:]
        else:
            print('error',contador)

    for i in range(len(array_texts)):
        elementos = array_texts[i]
        ultimos_100 = elementos[-100:]
        if '‰' in ultimos_100:
            indice = ultimos_100.index('‰')
            elementos = elementos[:len(elementos) - 100 + indice]
            array_texts[i] = elementos
        elif '=' in ultimos_100:
            indice = ultimos_100.index('=')
            elementos = elementos[:len(elementos) - 100 + indice]
            array_texts[i] = elementos
        else:
            print('no anda')

    for i in array_texts:
        i.lstrip()            
        
def get_all_binarys_for_dic(list_paths_audio):  ##Nesecita la funcino covert_bin()
    
    print(f'largo total del array : {len(list_paths_audio)}')
    
    diccionario = {}

    for path in list_paths_audio:
        diccionario[os.path.basename(path)] = convert_bin(path)
        print(f'largo temporal del dic {os.path.basename(path)}: {len(diccionario)}')
        
        
    return diccionario

def covert_dic_to_json(diccionario,file_name_json):

    def convertir_a_lista(valor):
        if isinstance(valor, np.ndarray):
            return valor.tolist()
        return valor

    diccionario_convertido = {k: convertir_a_lista(v) for k, v in diccionario.items()}

    json_string = json.dumps(diccionario_convertido)

    print(json_string)

    with open(f'C:/Users/Usuario/Desktop/proyectos/PYTHON/UNTREF/clases1/final_project/json/{file_name_json}.json', 'w') as archivo:
        json.dump(json_string, archivo)

    print("Se exporto")
    
def data_separation(arr_train,arr_leabel):
    if len(arr_leabel) == len(arr_leabel):
        sep_1 = int(len(arr_train)*.7)
        sep_2 = int(len(arr_train)*.7)+int((len(arr_train)-int(len(arr_train)*.7))/2)
        tra_x = arr_train[:sep_1]
        val_x = arr_train[sep_1:sep_2]
        tes_x = arr_train[sep_2:]
        tra_y = arr_leabel[:sep_1]
        val_y = arr_leabel[sep_1:sep_2]
        tes_y = arr_leabel[sep_2:] 
        return (tra_x,tra_y),(val_x,val_y),(tes_x,tes_y)
    else:
        return print('son de diferente largo')