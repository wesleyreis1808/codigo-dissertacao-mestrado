
# Simulação Capítulo 3

Comparação das técnicas GLRT, CFCPSC e WCFCPSC sob os cenário:

| Cenário   | Canal Seletivo    | Sombreamento  | Ruído
| --------  | -------           | ------------  | ------
| 1         | Não               | Não           | Uniforme  
| 2         | Sim               | Não           | Uniforme
| 3         | Sim               | Sim           | Uniforme
| 4         | Sim               | Sim           | Não-uniforme


## Execução

Para executar está simulação é necessário ter instalado o MatLab em uma versão igual ou superior a R2019a.

O arquivo que contém o script da simulação é o `article_Simulation.m`, os demais são arquivos de funções que são utilizadas.

Após executar simulação, as imagens contendo as curvas ROC de cada cenário estarão disponíveis no diretório `images` e o diretório `dats` conterá os arquivos com os pontos das curvas.