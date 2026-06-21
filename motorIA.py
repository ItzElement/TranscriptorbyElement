import os
import site
import math
import json
import urllib.request
import gc
import re
from difflib import SequenceMatcher
from faster_whisper import WhisperModel

# --- PARCHE PARA WINDOWS: Forzar a que encuentre las DLLs de NVIDIA ---
if os.name == 'nt':
    carpetas_pip = site.getsitepackages() + [site.getusersitepackages()]
    for carpeta in carpetas_pip:
        cublas_bin = os.path.join(carpeta, "nvidia", "cublas", "bin")
        cudnn_bin = os.path.join(carpeta, "nvidia", "cudnn", "bin")
        if os.path.exists(cublas_bin):
            os.add_dll_directory(cublas_bin)
            os.environ["PATH"] += os.pathsep + cublas_bin
        if os.path.exists(cudnn_bin):
            os.add_dll_directory(cudnn_bin)
            os.environ["PATH"] += os.pathsep + cudnn_bin
# ------------------------------------------------------------------------

def format_timestamp(seconds: float):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def capitalizar_primera_letra(texto):
    texto = texto.strip()
    for i, char in enumerate(texto):
        if char.isalpha(): 
            return texto[:i] + char.upper() + texto[i+1:]
    return texto

def corregir_lote_con_llama(lote_dicts, guion_referencia):
    url = "http://127.0.0.1:11434/api/generate"
    
    lineas_entrada = [f"{sub['index']}|{sub['text']}" for sub in lote_dicts]
    texto_entrada = "\n".join(lineas_entrada)
    
    prompt = f"""Eres un corrector ortográfico de subtítulos. Vas a comparar la ENTRADA (lo que escuchó la IA) con el GUION original.

REGLAS DE CORRECCIÓN (¡Síguelas estrictamente!):
1. ERRORES DE ESCUCHA: Si la ENTRADA dice algo erróneo (ej. "conseguir") pero el GUION dice la palabra correcta (ej. "construir"), OBLIGATORIAMENTE cámbialo por la palabra del guion.
2. REGIONALISMOS (VOSEO): Si la ENTRADA dice "querés", "tenés", OBLIGATORIAMENTE cámbialo a "querías", "tienes" según el guion.
3. CERO SINÓNIMOS: Si la entrada está bien, NO cambies palabras por sinónimos (ej. NUNCA cambies "estaba" por "había"). Respeta la palabra original.
4. IMPROVISACIONES: Si la entrada tiene palabras que no están en el guion, déjalas INTACTAS. El orador improvisó.
5. NO DUPLIQUES: Jamás repitas la oración de arriba.

Devuelve la lista con el formato exacto "ID|texto". Cero plática.

GUION ORIGINAL:
{guion_referencia}

ENTRADA A CORREGIR:
{texto_entrada}

SALIDA:
"""

    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            texto_corregido = result.get("response", "").strip()
            
            dict_respuestas = {}
            for linea in texto_corregido.split('\n'):
                partes = linea.split('|', 1)
                if len(partes) == 2:
                    idx_str = re.sub(r'\D', '', partes[0])
                    if idx_str:
                        idx = int(idx_str)
                        txt = partes[1].strip()
                        txt = txt.split('->')[0].strip()
                        txt = re.sub(r'\(.*\)', '', txt).strip() 
                        txt = re.sub(r'(?i)no hay error.*$', '', txt).strip()
                        txt = re.sub(r'(?i)se mantiene igual.*$', '', txt).strip()
                        dict_respuestas[idx] = txt
            
            textos_nuevos_aceptados = set()
            
            for sub in lote_dicts:
                texto_viejo = sub['text']
                if sub['index'] in dict_respuestas:
                    texto_nuevo = dict_respuestas[sub['index']]
                    if texto_viejo.lower() != texto_nuevo.lower():
                        pal_viejas = len(texto_viejo.split())
                        pal_nuevas = len(texto_nuevo.split())
                        if pal_nuevas > 0 and abs(pal_nuevas - pal_viejas) <= 3:
                            similitud = SequenceMatcher(None, texto_viejo.lower(), texto_nuevo.lower()).ratio()
                            if texto_nuevo.lower() in textos_nuevos_aceptados and texto_viejo.lower() not in textos_nuevos_aceptados:
                                print(f"[*] Escudo Anti-Duplicados en ID {sub['index']}: Llama repitió una línea. Se mantuvo original.")
                                textos_nuevos_aceptados.add(texto_viejo.lower())
                            elif similitud >= 0.55:
                                sub['text'] = texto_nuevo
                                textos_nuevos_aceptados.add(texto_nuevo.lower())
                            else:
                                print(f"[*] Escudo de Similitud en ID {sub['index']}: (Ratio {similitud:.2f}) Llama alucinó. Se mantuvo original.")
                                textos_nuevos_aceptados.add(texto_viejo.lower())
                        else:
                            print(f"[*] Escudo de Longitud en ID {sub['index']}: Llama alteró demasiadas palabras. Se mantuvo original.")
                            textos_nuevos_aceptados.add(texto_viejo.lower())
                    else:
                        textos_nuevos_aceptados.add(texto_viejo.lower())
                else:
                    textos_nuevos_aceptados.add(texto_viejo.lower())
                    
    except Exception as e:
        print(f"[ERROR OLLAMA]: {str(e)}")
        
    return lote_dicts


class MotorIA:
    """Clase principal que maneja todos los procesos pesados de Inteligencia Artificial"""
    def __init__(self):
        self.modelo_whisper = None

    def limpiar_vram_ollama(self, cb_log):
        cb_log("🧹 Vaciando VRAM de Llama 3...", "sys_msg")
        url = "http://127.0.0.1:11434/api/generate"
        payload = {"model": "llama3", "keep_alive": 0} 
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req)
        except Exception: pass

    def procesar_subtitulos(self, entrada, salida, motor, guion, cb_log, cb_seg, cb_metrics, cb_count, cb_finish):
        """Genera el archivo SRT con Whisper y Llama comunicándose con la UI mediante callbacks"""
        try:
            if not self.modelo_whisper:
                cb_log("Iniciando motor Whisper V3 (GPU). Por favor espera...", "sys_msg")
                self.modelo_whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
            
            cb_log("Analizando audio y calculando pausas...", "sys_msg")
            
            segments, info = self.modelo_whisper.transcribe(
                entrada, beam_size=5, language="es",
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=250),
                word_timestamps=True
            )

            cb_metrics(info.language, info.duration)

            MAX_PALABRAS = 14
            MIN_PALABRAS = 1
            sub_index = 1
            lista_srt_crudo = []
            
            for segment in segments:
                words = segment.words
                current_chunk_words = []
                chunk_start = None
                
                for idx, word_obj in enumerate(words):
                    if chunk_start is None:
                        chunk_start = word_obj.start
                    
                    palabra_cruda = word_obj.word.strip()
                    tiene_puntos = "..." in palabra_cruda or "…" in palabra_cruda
                    duracion = word_obj.end - word_obj.start
                    
                    gap = 0.0
                    if idx + 1 < len(words):
                        gap = words[idx+1].start - word_obj.end
                    pausa_por_silencio = gap >= 0.45
                    
                    palabra_limpia = palabra_cruda.lower()
                    for char in [',', '.', ';', ':', '!', '¡', '?', '¿', '…', '"', "'", '(', ')']:
                        palabra_limpia = palabra_limpia.replace(char, "")
                    
                    es_suspenso = tiene_puntos or (duracion > 1.2 and len(palabra_limpia) > 3)
                    
                    if es_suspenso:
                        palabra_limpia += "..."
                    
                    if "¿" in palabra_cruda: palabra_limpia = "¿" + palabra_limpia
                    if "?" in palabra_cruda: palabra_limpia = palabra_limpia + "?"
                    
                    current_chunk_words.append(palabra_limpia)
                    
                    pausa_fuerte = any(p in palabra_cruda for p in ['.', '?', '!', '…'])
                    pausa_suave = any(p in palabra_cruda for p in [',', ':', ';'])
                    
                    if (len(current_chunk_words) >= MAX_PALABRAS) or pausa_fuerte or (pausa_suave and len(current_chunk_words) >= MIN_PALABRAS) or es_suspenso or pausa_por_silencio:
                        chunk_end = word_obj.end
                        texto_final = " ".join(current_chunk_words)
                        texto_final = re.sub(r'\.{4,}', '...', texto_final)
                        
                        lista_srt_crudo.append({"index": sub_index, "start": format_timestamp(chunk_start), "end": format_timestamp(chunk_end), "text": texto_final})
                        cb_seg(chunk_start, chunk_end, texto_final)
                        cb_count()
                        
                        sub_index += 1
                        current_chunk_words = []
                        chunk_start = None
                
                if current_chunk_words:
                    chunk_end = segment.end
                    texto_final = " ".join(current_chunk_words)
                    texto_final = re.sub(r'\.{4,}', '...', texto_final)
                        
                    lista_srt_crudo.append({"index": sub_index, "start": format_timestamp(chunk_start), "end": format_timestamp(chunk_end), "text": texto_final})
                    cb_seg(chunk_start, chunk_end, texto_final)
                    cb_count()
                    sub_index += 1

            cb_log("\nLiberando memoria gráfica de Whisper...", "sys_msg")
            del self.modelo_whisper
            self.modelo_whisper = None
            gc.collect() 
                
            usar_llama = (motor == "Llama 3 Local" and len(guion) > 10)
            
            if usar_llama:
                cb_log("🧠 Iniciando Llama 3. Corrigiendo subtítulos por lotes...", "llama_msg")
                tamano_lote = 15
                total_lotes = math.ceil(len(lista_srt_crudo) / tamano_lote)
                
                for i in range(0, len(lista_srt_crudo), tamano_lote):
                    lote_actual = lista_srt_crudo[i : i + tamano_lote]
                    numero_lote = int(i / tamano_lote) + 1
                    cb_log(f"  -> Procesando lote {numero_lote} de {total_lotes}...", "llama_msg")
                    
                    corregir_lote_con_llama(lote_actual, guion)
                
                cb_log("✨ Llama 3 ha finalizado la corrección.", "llama_msg")
                self.limpiar_vram_ollama(cb_log)
                
            elif motor == "Llama 3 Local":
                cb_log("⚠️ Elegiste Llama 3 pero no pegaste el guion. El archivo SRT se guardó en crudo.", "sys_msg")

            # --- ESTÉTICA FINAL ---
            srt_final_text = ""
            capitalizar_sig = True
            
            for sub in lista_srt_crudo:
                texto_limpio = sub['text'].strip()
                texto_limpio = texto_limpio.replace("@", "o")
                
                texto_limpio = texto_limpio.replace("...", "___ELLIPSIS___").replace("…", "___ELLIPSIS___")
                texto_limpio = texto_limpio.replace("¿", "___QOPEN___").replace("?", "___QCLOSE___")
                
                for char in [',', '.', ';', ':', '!', '¡', '"', "'", '(', ')']:
                    texto_limpio = texto_limpio.replace(char, "")
                    
                texto_limpio = texto_limpio.replace("___ELLIPSIS___", "...")
                texto_limpio = texto_limpio.replace("___QOPEN___", "¿").replace("___QCLOSE___", "?")
                texto_limpio = texto_limpio.lower()
                
                if sub['index'] == 1:
                    texto_limpio = capitalizar_primera_letra(texto_limpio)
                elif "¿" in texto_limpio:
                    for i, char in enumerate(texto_limpio):
                        if char.isalpha():
                            texto_limpio = texto_limpio[:i] + char.upper() + texto_limpio[i+1:]
                            break
                elif capitalizar_sig and len(texto_limpio) > 0:
                    texto_limpio = capitalizar_primera_letra(texto_limpio)
                    capitalizar_sig = False
                    
                if texto_limpio.endswith("..."):
                    capitalizar_sig = True
                
                srt_final_text += f"{sub['index']}\n{sub['start']} --> {sub['end']}\n{texto_limpio}\n\n"
            
            with open(salida, "w", encoding="utf-8") as f:
                f.write(srt_final_text.strip() + "\n")

            cb_log(f"✅ ¡PROCESO COMPLETADO EXITOSAMENTE!", "sys_msg")
            cb_finish()
            
        except Exception as e:
            cb_log(f"❌ ERROR FATAL: {str(e)}", "sys_msg")
            cb_finish()