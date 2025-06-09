<p align="center">
    <img src="assets/icon-removebg-preview.png" alt="AgroIntelliVision Logo" width="350"/>
</p>

<h1 align="center">🌱 AgroIntelliVision</h1>

<p align="center"><i>Diagnóstico automatizado de doenças na folhagem da soja com Inteligência Artificial</i></p>

---

## Sobre o Projeto

O **AgroIntelliVision** é um sistema inteligente que utiliza Visão Computacional e Redes Neurais Convolucionais (CNNs) para identificar doenças em folhas de soja a partir de imagens. O objetivo é apoiar agricultores e pesquisadores no diagnóstico precoce de fitopatologias, promovendo uma agricultura mais eficiente e sustentável.

A aplicação oferece uma interface web para envio de imagens de folhas de soja, retornando um diagnóstico instantâneo com o nível de confiança da predição.

---

## Funcionalidades

- **📷 Upload de Imagem:** Interface drag-and-drop ou seleção de arquivo para envio da imagem.
- **🧠 Análise com IA:** Processamento da imagem por backend Flask e modelo de Deep Learning.
- **📊 Diagnóstico Instantâneo:** Exibição do resultado, doença detectada (ou folha saudável) e confiança.
- **🚀 Arquitetura Desacoplada:** Frontend (index.html) comunica-se com API backend (app.py).

---

## Tecnologias Utilizadas

**Backend:**
- 🐍 Python 3.10+
- 🧠 TensorFlow & Keras
- ⚡ Flask
- 🖼️ Pillow (PIL)

**Frontend:**
- 📄 HTML5
- 🎨 Tailwind CSS
- ⚙️ JavaScript (Vanilla)

**Machine Learning:**
- Modelo: Transfer learning (EfficientNetV2B2)
- Bibliotecas: NumPy, Matplotlib, Seaborn

---

## Estrutura do Projeto

```
AgroIntelliVision/
│
├── app.py                  # Servidor Flask (API)
├── main.py                 # Script para treinar o modelo
├── index.html              # Interface do usuário
├── README.md               # Este arquivo
│
├── assets/                 # Imagens e logos
├── config/
│   └── settings.py         # Configurações globais
│
├── data/
│   ├── preprocessing.py    # Data Augmentation
│   └── visualization.py    # Gráficos e visualizações
│
├── models/
│   ├── build_model.py      # Arquitetura do modelo
│   ├── train.py            # Compilação e otimizador
│   └── saved_models/       # Modelos treinados
│
└── utils/
        └── callbacks.py        # Callbacks do Keras
```

---

## Como Executar a Aplicação Web

1. **Pré-requisitos**
     - Python 3.10+ instalado
     - Git instalado

2. **Clonar o Repositório**
     ```bash
     git clone https://github.com/KatSilvax/AgroIntelliVision.git
     cd AgroIntelliVision
     ```

3. **Criar e Ativar Ambiente Virtual**
     ```bash
     # Windows
     python -m venv venv
     venv\Scripts\activate

     # macOS / Linux
     python3 -m venv venv
     source venv/bin/activate
     ```

4. **Instalar Dependências**
     - Crie um `requirements.txt` ou instale manualmente:
         ```
         tensorflow
         flask
         flask-cors
         numpy
         pillow
         ```
     - Instale:
         ```bash
         pip install -r requirements.txt
         ```

5. **Iniciar o Servidor Backend**
     ```bash
     python app.py
     ```
     > Certifique-se de que o modelo treinado (`modelo_soja.h5`) está em `models/saved_models/`.

6. **Abrir a Interface**
     - Abra o arquivo `index.html` no navegador.

---

## Como Treinar um Novo Modelo

1. **Organize seu Dataset**
     - Estrutura: `data/raw/DataSet/NOME_DA_DOENCA/img1.jpg`

2. **Ajuste as Configurações**
     - Edite `config/settings.py` conforme necessário.

3. **Instale Dependências Adicionais**
     ```bash
     pip install matplotlib seaborn
     ```

4. **Execute o Treinamento**
     ```bash
     python main.py
     ```
     - O modelo será salvo em `models/saved_models/`.

---

## Sobre o Modelo

- **Arquitetura:** EfficientNetV2B2 (transfer learning, ImageNet)
- **Classes:** 8 categorias (doenças da soja + folha saudável), selecionadas por serem as mais prevalentes no solo brasileiro e de maior impacto na agricultura nacional, conforme pesquisas e dados da Embrapa:
    - Ferrugem Asiática
    - Mancha Alvo
    - Oídio
    - Mancha Olho-de-Rã
    - Míldio
    - Crestamento Foliar de Cercospora
    - Antracnose
    - Folha Saudável

---

## Autores

<div align="center">

<table>
    <tr>
        <td align="center" width="150">
            <img src="assets/kat.webp" style="border-radius:50%;" width="120" height="120" alt="Katcilane Silva"/><br />
            <sub><b>Katcilane Silva</b></sub><br />
            <i>AI/ML Software Engineer</i>
        </td>
        <td align="center" width="150">
            <img src="assets/kaue.jpg" style="border-radius:50%;" width="120" height="120" alt="Kaue Ribeiro"/><br />
            <sub><b>Kaue Ribeiro</b></sub><br />
            <i>DevOps Engineer</i>
        </td>
    </tr>
</table>

</div>

<p align="center">
    <strong>Coordenador:</strong><br/>
    Patrick Ola Bressan
</p>

## Licença

Este projeto é de uso acadêmico e está sob licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais informações.
