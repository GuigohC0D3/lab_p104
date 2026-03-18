# lab_p104

Implementacao didatica de um Transformer simples em PyTorch, organizada em modulos pequenos para facilitar leitura, manutencao e testes.

## Objetivo

O projeto demonstra os blocos principais de um Transformer:

- codificacao posicional senoidal
- bloco feed-forward
- atencao escalada
- camadas de encoder e decoder
- geracao autoregressiva com mascara causal

O arquivo principal continua sendo `lab_p104/transformer.py`, que instancia o modelo, executa o encoder e gera uma sequencia de tokens de exemplo.

## Estrutura

```text
lab_p104/
|-- lab_p104/
|   |-- attention.py
|   |-- components.py
|   |-- generation.py
|   |-- layers.py
|   |-- model.py
|   `-- transformer.py
|-- requirements.txt
|-- venv/
`-- README.md
```

## Requisitos

- Python 3
- dependencias listadas em `requirements.txt`

## Instalacao

No PowerShell, a partir da raiz do projeto:

```powershell
venv\Scripts\pip.exe install -r requirements.txt
```

Dependencias atuais:

- `torch==2.10.0`
- `numpy==2.3.3`

## Como rodar

No PowerShell, a partir da raiz do projeto:

```powershell
venv\Scripts\python.exe lab_p104\transformer.py
```

## O que o programa faz

Ao executar, o script:

1. define um vocabulario artificial
2. cria uma entrada simples para o encoder
3. executa o encoder
4. faz a geracao token a token a partir de `<START>`
5. imprime a sequencia gerada no terminal

## Modulos

- `components.py`: blocos basicos como `ResidualNorm`, `FeedForwardBlock` e `SinusoidalPosition`
- `attention.py`: implementacao da unidade de atencao
- `layers.py`: camadas e pilhas de encoder e decoder
- `model.py`: classe `SimpleTransformer`
- `generation.py`: funcao de inferencia autoregressiva
- `transformer.py`: ponto de entrada para demonstracao do projeto

## Validacao

Para verificar se os arquivos compilam corretamente:

```powershell
venv\Scripts\python.exe -m py_compile lab_p104\attention.py lab_p104\components.py lab_p104\layers.py lab_p104\model.py lab_p104\generation.py lab_p104\transformer.py
```
