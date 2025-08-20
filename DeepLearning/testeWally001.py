###########################################################################
###                W  H  E  R  E  '  S     W  A  L  D  O  ?             ###
###########################################################################
### encontra uma imagem dentro de outra, ao estilo Where's Waldo?       ###
###########################################################################
### Prof. Filipo Mor - filipomor.com - github.com/ProfessorFilipo       ###
###########################################################################

# pip install opencv-python numpy
import cv2
import numpy as np

# ======= CONFIGURAÇÃO RÁPIDA =======
# Troque pelos seus arquivos:
IMG_GRANDE = "waldo001.jpeg"      # imagem grande
IMG_ALVO   = "alvo001.jpg"        # recorte com o padrão a ser buscado (um exemplo)
N_MAX      = 30                          # quantos achados você quer no máximo (use None para todos)
THRESH     = 0.70                        # limiar de similaridade (0..1) – ajuste se necessário
ESCALAS    = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]   # tente poucos fatores para manter simples/rápido
NMS_IOU    = 0.30                        # supressão de sobreposição (0.3 é um bom começo)
SAIDA      = "resultado_marcado.png"

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

for s in ESCALAS:
    # redimensiona o template
    newW = max(1, int(W_t * s))
    newH = max(1, int(H_t * s))
    tpl_s = cv2.resize(tpl_gray, (newW, newH), interpolation=cv2.INTER_AREA)
    (h, w) = tpl_s.shape[:2]

    # matchTemplate normalizado
    res = cv2.matchTemplate(img_gray, tpl_s, cv2.TM_CCOEFF_NORMED)

    # pega posições acima do limiar
    loc = np.where(res >= THRESH)
    for (y, x) in zip(*loc):
        boxes.append([int(x), int(y), int(w), int(h)])
        scores.append(float(res[y, x]))

# Se nada encontrado, finalize
if not boxes:
    print("Nenhuma ocorrência encontrada com o limiar dado.")
    raise SystemExit()

# NMS para remover caixas muito sobrepostas
# cv2.dnn.NMSBoxes espera: boxes=[x,y,w,h], scores, score_thresh, nms_thresh
idxs = cv2.dnn.NMSBoxes(boxes, scores, THRESH, NMS_IOU)
idxs = idxs.flatten().tolist() if len(idxs) else []

if not idxs:
    print("Nada após NMS (tente reduzir THRESH ou aumentar NMS_IOU).")
    raise SystemExit()

# Ordena por confianca (score) desc e aplica corte N_MAX
idxs = sorted(idxs, key=lambda i: scores[i], reverse=True)
if isinstance(N_MAX, int):
    idxs = idxs[:N_MAX]

# Desenha retângulos
out = img.copy()
for i in idxs:
    (x, y, w, h) = boxes[i]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # opcional: legenda com score
    cv2.putText(out, f"{scores[i]:.2f}", (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imwrite(SAIDA, out)
print(f"Encontradas {len(idxs)} ocorrências. Resultado salvo em: {SAIDA}")
