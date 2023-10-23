  
# Simulação Capítulo 4

Comparação das técnicas GLRT, WCFCPSC, GID, PRIDe e Redes Neurais Artificiais (RNAs).
 
## Cenários

As simulações realizadas no capítulo 4 utilizam dos seguintes cenários:

| Cenário | Canal Seletivo | Sombreamento | Ruído
| ------- | -------------- | ------------ | ------
| 1 		| Sim 			| Não 			| Uniforme
| 2 		| Sim 			| Sim 			| Uniforme
| 3 		| Sim 			| Sim 			| Não-uniforme
| 4 		| Sim 			| Não 			| Uniforme e dinâmico
| 5 		| Sim 			| Sim 			| Uniforme e dinâmico
| 6 		| Sim 			| Sim 			| Não-uniforme e dinâmico

## Ambiente

Os scripts foram executados utilizando a linguagem Python na versão 3.8.10.

As simulações foram realizadas utilizando um servidor com:

* CPU: Intel Xeon Silver 4314 (32 núcleos 2,4 GHz)
* GPU: NVidia RTX A4000 (16 GB GDDR6, 6144  CUDA cores)
* Memória RAM:  256 GB 

Drivers para computação paralela:

*  Driver GPU NVidia versão 516.94
* CUDA toolkit versão 12.1.0
* cuDNN SDK versão 8.3

Obs: Para mais detalhes da configuração do ambiente para computação paralela, consulte a [documentação do Tensorflow](https://www.tensorflow.org/install/gpu?hl=pt-br)

O arquivo **requirements.txt** contém os pacotes necessários para executar as simulações. Para instalar estes pacotes no seu ambiente basta executar 
		
	pip install -r requirements.txt

## Execução

Os arquivos a serem executados se encontram no diretório [**scripts**](./scripts). Para rodar um script basta executar

    python <NOME DO SCRIPT>
	
	# Exemplos
	## Criando os datasets do cenários 1, 2 e 3	
	python create_selective_dataset_noise_const.py
	
	## Avaliar o desempenho dos detectores sobre os cenários 1, 2 e 3 
	python evaluate_detectors_over_noise_const.py

Um diretório chamado **LOCAL_FILES** será criado dentro do `/capitulo-4` e conterá todos os artefatos resultantes das execuções.

A fim de facilitar o treinamento e avaliação do desempenho de diversos modelos de RNAs, o script `evaluate_RNAs.py` não é executado diretamente, e sim através do arquivo `run.sh`. Dessa forma, basta executar o comando:

	# Adicionar permissão de execução ao script
	chmod +x run.sh
	
	# Executa script
	./run.sh
 
