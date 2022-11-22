**Questão 1)** Considerando os dados presentes no arquivo class01.csv, treine o algoritmo Naive Bayes Gaussiano utilizando a metodologia de validação cruzada holdout (utilize para treino as 350 primeiras linhas e para validação as demais). Qual o valor da acurácia a base de treino? Qual o valor da acurácia na base de validação?

**Questão 2)** Considerando os dados presentes no arquivo class02.csv, treine o algoritmo 10-Nearest Neighbors (KNN com 𝑘 = 10 e distância Euclidiana), utilizando a metodologia de validação cruzada k-fold com 5 folds. Considere que a primeira pasta de validação seja formada pelas primeiras 20% linhas do arquivo, que a segunda pasta de validação seja formada pelas 20% linhas seguintes, e assim por diante, até atingir a última pasta, formada pelas 20% linhas finais da base. Qual a acurácia média para a base de validação?

**Questão 3)** Considerando os dados presentes no arquivo reg01.csv, obtenha um modelo de regressão linear com regularização L1 (LASSO com 𝛼 = 1) utilizando a metodologia Leave-One-out. Qual o valor médio do Root Mean Squared Error (RMSE) para a base de treino e para a base de validação?

**Questão 4)** Considerando os dados do arquivo reg02.csv, treine árvores de regressão, sem realizar podas, utilizando a metodologia de validação cruzada k-fold com 𝑘 = 5. Qual o valor do Mean Absolute Error (MAE) para a base de treino? Qual o valor médio do MAE para a base de validação?