###########################################################################
###                W  H  E  R  E  '  S     W  A  L  D  O  ?             ###
###########################################################################
### encontra uma imagem dentro de outra, ao estilo Where's Waldo?       ###
### nesta versão, implementa rotações no padrão sendo buscado           ###
###########################################################################
### Prof. Filipo Mor - filipomor.com - github.com/ProfessorFilipo       ###
###########################################################################

import cv2
import numpy as np

# ======= CONFIGURAÇÃO RÁPIDA =======
IMG_GRANDE = "waldo001.jpeg"
IMG_ALVO   = "alvo001.jpg"
N_MAX      = 30
THRESH     = 0.70
ESCALAS    = [0.8, 1.0, 1.2, 1.4, 1.6]
ROTACOES   = [-30, -15, 0, 15, 30]   # ângulos em graus
NMS_IOU    = 0.30
SAIDA      = "resultado_rotacoes.png"

# ======= CÓDIGO =======
img = cv2.imread(IMG_GRANDE)
tpl = cv2.imread(IMG_ALVO)

if img is None or tpl is None:
    raise SystemExit("Erro ao carregar imagens. Verifique os caminhos.")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
(H_t, W_t) = tpl_gray.shape[:2]

boxes = []
scores = []

def rotacionar(imagem, angulo):
    """Rotaciona imagem mantendo o tamanho original (com bordas pretas)."""
    (h, w) = imagem.shape[:2]
    centro = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotada = cv2.warpAffine(imagem, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    return rotada

# Loop em escalas e rotações
for s in ESCALAS:
    newW = max(1, int(W_t * s))
    newH = max(1, int(H_t * s))
    tpl_s = cv2.resize(tpl_gray, (newW, newH), interpolation=cv2.INTER_AREA)

    for ang in ROTACOES:
        tpl_r = rotacionar(tpl_s, ang)
        (h, w) = tpl_r.shape[:2]

        # matchTemplate
        res = cv2.matchTemplate(img_gray, tpl_r, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= THRESH)

        for (y, x) in zip(*loc):
            boxes.append([int(x), int(y), int(w), int(h)])
            scores.append(float(res[y, x]))

# Se nada encontrado
if not boxes:
    print("Nenhuma ocorrência encontrada com o limiar dado.")
    raise SystemExit()

# NMS
idxs = cv2.dnn.NMSBoxes(boxes, scores, THRESH, NMS_IOU)
idxs = idxs.flatten().tolist() if len(idxs) else []

if not idxs:
    print("Nada após NMS (tente reduzir THRESH ou aumentar NMS_IOU).")
    raise SystemExit()

# Ordena e aplica corte N_MAX
idxs = sorted(idxs, key=lambda i: scores[i], reverse=True)
if isinstance(N_MAX, int):
    idxs = idxs[:N_MAX]

# Desenha resultados
out = img.copy()
for i in idxs:
    (x, y, w, h) = boxes[i]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(out, f"{scores[i]:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imwrite(SAIDA, out)
print(f"Encontradas {len(idxs)} ocorrências. Resultado salvo em: {SAIDA}")
