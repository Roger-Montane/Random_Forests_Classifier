Iker Soto Picon (1565939)
Miquel Baños Boncompte (1463615)
Roger Montané Güell (1569031)

INSTRUCCIONS D'EXECUCIÓ:
Per tal d'executar el classificador, ens haurem de situar primer al directori Source de la carpeta, 
i a continuació caldrà anar a la terminal i introduir la comanda "python3 Client.py". El Client 
es pot executar amb 3 datasets diferents (Iris, Sonar i Mnist). Per tal de triar amb quin dataset 
es vol executar comentar/des-comentar les línies corresponents al dataset que volem. Les línies
necessàries per a l'execució de cada Dataset estàn delimitades i marcades per a cada Dataset al 
fitxer CLient.py i a la línia 195 del fitxer RandomForest.py (tal i com està ara, farà anar el 
Dataset Mnist). Un cop executat, per pantalla es mostraran diversos missatges durant l'execució 
informant de l'estat de l'entrenament dels arbres i, finalment, es mostrarà el temps d'execució 
del programa (en minuts), la precisió del classificador, la llista de les precisions (Ypred), la 
de valors esperats (Y_test) i, per últim, la llista que compara la llista de prediccions amb la 
llista de valors esperats (Y_test - Ypred), en el següent format:

    Program Executed in X minutes.
    accuracy Y %
    Ypred = [...]
    Y_test = [...]
    Y_test - Y_train = [...]
