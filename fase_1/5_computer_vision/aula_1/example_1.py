import cv2
import matplotlib.pyplot as plt

# Carregar a imagem
imagem = cv2.imread('C:\\Users\\Meu Computador\\Pictures\\fish.jpg')

# Desenhar retangulo
inicio = (10, 40)
fim = (210, 190)
cor = (255, 0, 0) # Azul
espessura = 2

imagem = cv2.rectangle(imagem.copy(), inicio, fim, cor, espessura)


# Suavizar a imagem
# imagem = cv2.GaussianBlur(imagem, (15, 15), 0)

# Converter a imagem de BGR para RGB
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Converter a imagem de BGR para grayscale
# imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Detectar bordas na imagem
# imagem = cv2.Canny(imagem, 100, 200)


# Exibir a imagem
plt.imshow(imagem)#, cmap='gray')
plt.axis('off') # Desabilita os eixos
plt.show()

cv2.imwrite('./fish.jpg', imagem)