
<p align="center">
  <img src="AgroIntelliVision/assets/icon.jpg" alt="AgroIntelliVision Logo" width="150"/>
</p>



## 🌱 AgroIntelliVision

> **Diagnóstico automatizado de doenças na folhagem da soja com Inteligência Artificial.**

AgroIntelliVision é um sistema inteligente que utiliza Visão Computacional e Redes Neurais Convolucionais (CNNs) para identificar doenças em folhas de soja a partir de imagens. O projeto visa apoiar agricultores e pesquisadores no diagnóstico precoce de fitopatologias, promovendo uma agricultura mais eficiente e sustentável.

---

### 📌 Funcionalidades

* 📷 Upload de imagem de folha de soja
* 🧠 Análise automática com modelo de IA treinado
* 💬 Diagnóstico via página web ou bot do Telegram
* 📦 Deploy com Docker e FastAPI

---

### 📁 Estrutura do Projeto

```bash
AgroIntelliVision/
├── app/                 # Backend com FastAPI e lógica do bot Telegram
├── frontend/            # Interface web com HTML + CSS + Bootstrap
├── model/               # Armazenamento e carregamento do modelo treinado
├── notebooks/           # Notebooks de prototipagem e exploração
├── Dockerfile           # Build do container da aplicação
├── docker-compose.yml   # Orquestração do container
├── requirements.txt     # Dependências do projeto
└── README.md            # Documentação do projeto
```

---

### 🚀 Como Executar

**Pré-requisitos:**

* Docker e Docker Compose instalados

**1. Clonar o repositório**

```bash
git clone https://github.com/KatSilvax/AgroIntelliVision.git
cd AgroIntelliVision
```

**2. Executar com Docker**

```bash
docker-compose up --build
```

**3. Acessar a aplicação**

* Interface web: `http://localhost:8000`
* API docs: `http://localhost:8000/docs`

---

### 🤖 Uso do Bot do Telegram

> Envie uma imagem de uma folha de soja para receber o diagnóstico diretamente pelo Telegram.

📌 Em desenvolvimento. (Será incluído o link para o bot e instruções de uso assim que finalizado.)

---

### 🧠 Tecnologias Utilizadas

* 🐍 Python 3.10
* 🔬 PyTorch (CNN)
* ⚡ FastAPI
* 💬 Telegram Bot API
* 🐳 Docker
* 💻 Bootstrap 5
* 🖼️ OpenCV

---

### 📊 Dataset e Modelo

* O modelo foi treinado com um dataset de imagens rotuladas de folhas de soja, contendo classes como:

  * Mancha-alvo
  * Ferrugem asiática
  * Mofo branco
  * Folha saudável
* Arquitetura baseada em CNN com ajustes de hiperparâmetros e validação cruzada.
* Resultados preliminares indicam alta acurácia (acima de 90%) nos testes com dados reais.

*(Mais detalhes técnicos em breve na pasta `model/` ou `notebooks/`.)*

---

### 📷 Exemplo de Diagnóstico

> *(Adicionar aqui um exemplo visual de input/output futuramente)*

---

### 👩‍💻 Autora

* **Kat Cilane** – Estudante de Ciência da Computação e pesquisadora bolsista de Iniciação Científica pelo IFMS.

---

### 🧪 Próximos Passos

* [ ] Aprimorar o front-end com interatividade
* [ ] Integrar com banco de dados para histórico de diagnósticos
* [ ] Publicar artigo científico sobre a abordagem
* [ ] Treinar com base de dados maior e mais diversa

---

### 📜 Licença

Este projeto é de uso acadêmico e está sob licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais informações.

