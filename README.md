import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Definindo as transformações, como redimensionamento e conversão para tensor
transform = transforms.Compose([transforms.ToTensor()])

# Carregando o dataset MNIST (como exemplo)
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Criando o DataLoader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# Obtendo um lote de imagens
dataiter = iter(trainloader)
images, labels = next(dataiter)  # Corrigido para usar next()

# Exibindo as imagens
fig, axes = plt.subplots(1, 4, figsize=(12, 3))  # Aqui 1 linha e 4 colunas de imagens
for i in range(4):
    # Exibindo cada imagem, removendo a dimensão de canal (se for só uma imagem em grayscale)
    axes[i].imshow(images[i].squeeze(), cmap='gray')  # Exibindo cada imagem
    axes[i].set_title(f'Label: {labels[i].item()}')  # Exibindo o rótulo (classe) da imagem
    axes[i].axis('off')  # Desabilitar os eixos

plt.show()
Desenvolvedor de Redes Neurais com PyTorch - Classificação de Dígitos MNIST
No meu projeto mais recente, implementei uma rede neural do zero para classificar imagens do famoso dataset MNIST, utilizando PyTorch. Durante o desenvolvimento, apliquei conceitos essenciais de redes neurais, como camadas totalmente conectadas, regularização, otimização e função de perda. O modelo atingiu uma precisão superior a 90% na classificação de dígitos, validando a eficácia do modelo treinado.

Além disso, o projeto me permitiu aprofundar meu conhecimento em Machine Learning, especialmente na aplicação de redes neurais profundas (DNNs) para tarefas de visão computacional. A experiência foi enriquecedora, e agora estou mais preparado para aplicar esses conhecimentos em projetos futuros, explorando ainda mais o campo do aprendizado de máquina e inteligência artificial.

Esse projeto foi desenvolvido como parte do meu bootcamp de Machine Learning, onde aprendi a trabalhar com PyTorch, análise de dados, preparação de datasets e otimização de modelos.

Tecnologias utilizadas: PyTorch, Python, redes neurais, aprendizado supervisionado, MNIST.
