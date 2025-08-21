###########################################################################
###                W  H  E  R  E  '  S     W  A  L  D  O  ?             ###
###########################################################################
### encontra uma imagem dentro de outra, ao estilo Where's Waldo?       ###
### nesta versão, implementa busca por ORB e homografia para marcação   ###
###########################################################################
### Prof. Filipo Mor - filipomor.com - github.com/ProfessorFilipo       ###
###########################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======= CONFIGURAÇÃO =======
IMG_GRANDE = "waldo001.jpeg"
IMG_ALVO   = "alvo002.jpg"
SAIDA_CAIXA   = "resultado_orb_homografia.png"
SAIDA_MATCHES = "resultado_orb_matches.png"
N_FEATURES = 5000     # número máximo de keypoints
MIN_MATCHES = 5      # mínimo de correspondências para considerar válido

# ======= CÓDIGO =======
img = cv2.imread(IMG_GRANDE, cv2.IMREAD_COLOR)
tpl = cv2.imread(IMG_ALVO, cv2.IMREAD_COLOR)

if img is None or tpl is None:
    raise SystemExit("Erro ao carregar imagens.")

# Converte para cinza
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

# Inicializa ORB
orb = cv2.ORB_create(nfeatures=N_FEATURES)

# Detecta keypoints e extrai descritores
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Verifica se há descritores
if des1 is None or des2 is None:
    raise SystemExit("Erro: não foram encontrados descritores em uma das imagens. \
                     Tente aumentar o tamanho do template ou ajustar parâmetros.")

# Força descritores para uint8 (ORB usa Hamming)
des1 = np.uint8(des1)
des2 = np.uint8(des2)

# Matcher (usando KNN)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Aplica razão de Lowe para filtrar bons matches
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print(f"Total de bons matches: {len(good)}")

if len(good) > MIN_MATCHES:
    # Extrai coordenadas dos matches
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)

    # Calcula homografia
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist() if mask is not None else None

    if M is not None:
        # Obtém contorno do template e projeta na imagem grande
        h, w = gray2.shape
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)

        # Desenha polígono na imagem grande
        img_out = img.copy()
        cv2.polylines(img_out, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
        cv2.imwrite(SAIDA_CAIXA, img_out)
        print(f"Objeto detectado! Caixa delimitadora salva em {SAIDA_CAIXA}")

        # Visualização dos matches filtrados
        draw_params = dict(matchColor=(0,255,0),
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2)
        img_matches = cv2.drawMatches(img, kp1, tpl, kp2, good, None, **draw_params)
        cv2.imwrite(SAIDA_MATCHES, img_matches)
        print(f"Matches desenhados salvos em {SAIDA_MATCHES}")

        # Exibe resultados lado a lado
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.title("Detecção do Waldo")
        plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("Matches ORB")
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    else:
        print("Homografia não pôde ser calculada.")
else:
    print("Poucos matches encontrados - tente ajustar parâmetros ou usar SIFT.")
