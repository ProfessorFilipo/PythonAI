###########################################################################
###                W  H  E  R  E  '  S     W  A  L  D  O  ?             ###
###########################################################################
### encontra uma imagem dentro de outra, ao estilo Where's Waldo?       ###
### nesta versão, implementa busca por ORB e homografia para marcação   ###
### implementa: ORB → AKAZE → SIFT + Homografia para maior robustez     ###
###########################################################################
### Prof. Filipo Mor - filipomor.com - github.com/ProfessorFilipo       ###
###########################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======= CONFIG =======
IMG_GRANDE = "waldo001.jpeg"
IMG_ALVO   = "alvo001.png"
OUT_POLY   = "saida_caixa.png"
OUT_MATCH  = "saida_matches.png"
MIN_MATCHES = 8          # mínimo de bons matches para tentar homografia
MIN_SIDE    = 96         # mínimo (em px) do menor lado do template após upscale
RATIO_BIN   = 0.80       # ratio test p/ descritores binários (ORB/AKAZE)
RATIO_FLOAT = 0.75       # ratio test p/ SIFT (float)
SHOW = True

# ======= PRÉ-PROCESSAMENTO =======
def preprocess_gray(g):
    # CLAHE + leve unsharp
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g1 = clahe.apply(g)
    blur = cv2.GaussianBlur(g1, (0,0), 1.0)
    sharp = cv2.addWeighted(g1, 1.5, blur, -0.5, 0)  # unsharp
    return sharp

def ensure_min_size(img_gray, min_side=MIN_SIDE):
    h, w = img_gray.shape[:2]
    smin = min(h, w)
    if smin >= min_side:
        return img_gray
    scale = float(min_side) / float(smin)
    new_w = int(round(w*scale))
    new_h = int(round(h*scale))
    return cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

# ======= HOMOGRAFIA + DESENHO =======
def homography_and_draw(img_color, tpl_gray, kp_img, kp_tpl, good, matchesMask=None,
                        out_poly=OUT_POLY, out_match=OUT_MATCH):
    src_pts = np.float32([kp_tpl[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return False

    h, w = tpl_gray.shape
    pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)

    img_out = img_color.copy()
    cv2.polylines(img_out, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
    cv2.imwrite(out_poly, img_out)
    print(f"[OK] Caixa delimitadora salva em: {out_poly}")

    # desenha matches (aproveita a mesma mask de RANSAC se vier de fora)
    if matchesMask is None and mask is not None:
        matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)
    # Atenção: aqui usamos o template "em 3 canais" só para visual,
    # então empilhamos a versão cinza:
    tpl_bgr = cv2.cvtColor(tpl_gray, cv2.COLOR_GRAY2BGR)
    img_matches = cv2.drawMatches(img_color, kp_img, tpl_bgr, kp_tpl, good, None, **draw_params)
    cv2.imwrite(out_match, img_matches)
    print(f"[OK] Matches salvos em: {out_match}")

    if SHOW:
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1); plt.title("Detecção"); plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.subplot(1,2,2); plt.title("Matches"); plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.show()

    return True

# ======= PIPELINES DE DETECÇÃO =======
def try_orb(gray_img, gray_tpl):
    orb = cv2.ORB_create(
        nfeatures=8000, scaleFactor=1.2, nlevels=12,
        edgeThreshold=15, firstLevel=0, WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
        fastThreshold=5
    )
    kp1, des1 = orb.detectAndCompute(gray_img, None)
    kp2, des2 = orb.detectAndCompute(gray_tpl, None)
    print(f"[ORB] kp_img={len(kp1)} kp_tpl={len(kp2)} des1={None if des1 is None else des1.shape} des2={None if des2 is None else des2.shape}")

    if des1 is None or des2 is None:
        return None

    des1 = np.uint8(des1); des2 = np.uint8(des2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < RATIO_BIN * n.distance:
            good.append(m)
    print(f"[ORB] good={len(good)}")
    return (kp1, kp2, good) if len(good) >= MIN_MATCHES else None

def try_akaze(gray_img, gray_tpl):
    akaze = cv2.AKAZE_create()  # por padrão, descritor binário (MLDB)
    kp1, des1 = akaze.detectAndCompute(gray_img, None)
    kp2, des2 = akaze.detectAndCompute(gray_tpl, None)
    print(f"[AKAZE] kp_img={len(kp1)} kp_tpl={len(kp2)} des1={None if des1 is None else des1.shape} des2={None if des2 is None else des2.shape}")

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < RATIO_BIN * n.distance:
            good.append(m)
    print(f"[AKAZE] good={len(good)}")
    return (kp1, kp2, good) if len(good) >= MIN_MATCHES else None

def try_sift(gray_img, gray_tpl):
    if not hasattr(cv2, "SIFT_create"):
        print("[SIFT] Não disponível nesta build do OpenCV. Instale opencv-contrib-python.")
        return None
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.2)
    kp1, des1 = sift.detectAndCompute(gray_img, None)
    kp2, des2 = sift.detectAndCompute(gray_tpl, None)
    print(f"[SIFT] kp_img={len(kp1)} kp_tpl={len(kp2)} des1={None if des1 is None else des1.shape} des2={None if des2 is None else des2.shape}")

    if des1 is None or des2 is None:
        return None

    # FLANN para descritores float32
    index_params = dict(algorithm=1, trees=5)  # KDTree
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < RATIO_FLOAT * n.distance:
            good.append(m)
    print(f"[SIFT] good={len(good)}")
    return (kp1, kp2, good) if len(good) >= MIN_MATCHES else None

# ======= MAIN =======
img = cv2.imread(IMG_GRANDE, cv2.IMREAD_COLOR)
tpl = cv2.imread(IMG_ALVO,   cv2.IMREAD_COLOR)
if img is None or tpl is None:
    raise SystemExit("Erro ao carregar imagens. Verifique caminhos.")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_tpl = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

# upscale + realce no template (normalmente é quem sofre com poucos cantos)
gray_tpl = ensure_min_size(gray_tpl, MIN_SIDE)
gray_img = preprocess_gray(gray_img)
gray_tpl = preprocess_gray(gray_tpl)

# Tenta ORB → AKAZE → SIFT
for method_name, method in [("ORB", try_orb), ("AKAZE", try_akaze), ("SIFT", try_sift)]:
    print(f"\n=== Tentando {method_name} ===")
    out = method(gray_img, gray_tpl)
    if out is None:
        print(f"[{method_name}] insuficiente. Tentando próximo...\n")
        continue
    kp_img, kp_tpl, good = out
    ok = homography_and_draw(img, gray_tpl, kp_img, kp_tpl, good)
    if ok:
        break
else:
    raise SystemExit("Falha: nenhum detector gerou matches suficientes. "
                     "Tente um template maior/mais texturizado, ou ajuste parâmetros.")

