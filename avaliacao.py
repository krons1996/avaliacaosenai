import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import random



# Inicializa o mixer de áudio
pygame.mixer.init()

# Lista de áudios disponíveis
audio_files = ["bemtevi.mp3", "carcara.mp3", "joaodebarro.mp3", "sabia.mp3", "seriema.mp3"]

# Dicionário para mapear o nome dos arquivos para nomes descritivos
audio_nomes = {
    "bemtevi.mp3": "bem-te-vi",
    "carcara.mp3": "carcara",
    "joaodebarro.mp3": "joao-de-barro",
    "sabia.mp3": "sabia",
    # "sonsdocerrado.mp3": "variados",
    "seriema.mp3": "seriema"
}

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

# Função EAR
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))
    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear

# Função MAR
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar

# Limiares
ear_limiar = 0.28
mar_limiar = 0.17 
dormindo = 0

# Inicializa a câmera
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Estado do som
som_tocando = False
ultimo_tempo_audio = time.time()

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Verifica se é hora de tocar um novo áudio
        tempo_atual = time.time()
        if tempo_atual - ultimo_tempo_audio >= 10:
            audio_aleatorio = random.choice(audio_files)
            try:
                pygame.mixer.music.load(audio_aleatorio)
                pygame.mixer.music.play()
                ultimo_tempo_audio = tempo_atual
                nome_audio = audio_nomes.get(audio_aleatorio, "desconhecido")  # Obtém o nome descritivo do dicionário
            except pygame.error as e:
                print(f"Erro ao carregar o áudio {audio_aleatorio}: {e}")

        if saida_facemesh.multi_face_landmarks:
            print("Rosto detectado")
        else:
            print("Nenhum rosto detectado")
            if som_tocando:
                pygame.mixer.music.stop()  # Para o som
                som_tocando = False  # Atualiza o estado para som parado

        try:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 204, 0), thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark
                
                for id_coord, coord_xyz in enumerate(face):
                    if id_coord in p_olhos:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)
                    if id_coord in p_boca:
                        coord_cv = mp_drawing._normalized_to_pixel_coordinates(coord_xyz.x, coord_xyz.y, largura, comprimento)
                        cv2.circle(frame, coord_cv, 2, (255, 0, 0), -1)

                # Chamada do EAR e print
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                cv2.rectangle(frame, (0, 1), (312, 210), (58, 58, 55), -1)
                cv2.putText(frame, f"EAR: {round(ear, 2)}", (1, 24),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255), 2)
                
                # Verificação da limiar                
                # inica uma comparação entre o valor ear e ear_limiar
                
                if ear < ear_limiar: # se for verdade em que o ear é menor que o ear_limiar, a pessoa está dormindo
                    t_inicial = time.time() if dormindo == 0 else t_inicial # recebe o tempo inicial no momento em que a pessoa dorme
                    dormindo = 1  # no momento em que o ear é maior ou igual ao ear_limiar, recebe o valor de 1(acordado)
                if dormindo == 1 and ear >= ear_limiar: # se o ear for maior ou igual ao ear_limiar, a pessoa está acordada
                    dormindo = 0 # no momento em que o ear é menor ao ear_limiar, recebe o valor de 0(dormindo)
                t_final = time.time() # recebe o tempo no momento em que a pessoa acorda
                tempo = (t_final - t_inicial) if dormindo == 1 else 0.0  # calculando o tempo que o olho ficou fechado

                cv2.putText(frame, f"Tempo: {round(tempo, 3)}", (1, 80),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.9, (255, 255, 255), 2)
                if tempo >= 1.5:
                    cv2.rectangle(frame, (30, 400), (610, 452), (109, 233, 219), -1)
                    cv2.putText(frame, f"Muito tempo com olhos fechados!", (80, 435),
                                cv2.FONT_HERSHEY_DUPLEX,
                                0.85, (58, 58, 55), 1)
                    

                # Chamada do MAR e print
                # alarme da boca
                def tocar_alarme(arquivo_mp3):
                    pygame.mixer.init()  # Inicializa o mixer do pygame
                    pygame.mixer.music.load(arquivo_mp3)  # Carrega o arquivo MP3
                    pygame.mixer.music.play()  # Reproduz o alarme
                mar = calculo_mar(face, p_boca)
                # aqui acontece algo interessante, o audio fica em looping disparando, mas só possível ouvi-lo completo quando a condição é falsa, ou seja, quando a boca se abre o som do alarme é ouvido.
                # então, supri o fato de abrir a boca toca o alarme.                
                cv2.putText(frame, f"MAR: {round(mar, 2)}{ f'Boca aberta!' if mar >= mar_limiar else f'Boca fechada. {tocar_alarme("dontsleep/alarmeboca.mp3")}'}",
                            (2, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (0, 0, 255), 2)
               

                # Dentro do loop, adicione o texto para mostrar o áudio atual
                cv2.putText(frame, f"Audio: {nome_audio}", (1, 110),
                            cv2.FONT_HERSHEY_DUPLEX,
                          0.9, (255, 255, 255), 2)




        except Exception as e:
            print("Erro:", e)

        finally:
            print("Processamento concluído")
        
        cv2.imshow('Camera', frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()