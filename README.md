# Disclaimer
The files needed to load the datasets are NOT included in the repository since those were provided by the professor of the subject in which this project was carried out. This means that the IRIS dataset will be the only one to work "out of the box" since it is loaded from the sklearn datasets python library. As for the MNIST and sonar datasets, here are their respective links in order to get the data:
* http://yann.lecun.com/exdb/mnist/
* https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar,+Mines+vs.+Rocks%29
and you can create small scripts to load the data into the Client from the downloads from the before-mentioned pages.

# Random forests classifier
This project has been carried out with @isotopi and @BaosMiquel22

The project consists of a random forest implementation using Object Oriented Programming in Python that can be used in various datasets (3 of them included in the source code).

In order tu use it (in Linux), first open a terminal and navigate to the directory of the source code. Then, type `python3 Client.py` into the terminal and hit Enter. As before mentioned, the client can be executed with 3 different datasets (Iris, Sonar or MNIST). To choose which one to use, you will have to comment/uncomment the corresponding lines to each dataset in the `Client.py` file (it is specified in the file which lines of code are necessary for each dataset). If you want to use the Iris or Sonar datasets, you will also have to uncomment line 195 in the `RandomForest.py` file. By default, the classifier will use the MNIST dataset.

Once you execute the before mentioned command, several messages will be displayed during execution reporting on the state of tree training and, finally, the execution time (in minutes), the precision of the classifier, the list of predictions (Ypred), the list of expected values (Y_test) and the list comparing predicition versus expected values (Y_test - Ypred) will be displayed in the following format:

    Program Executed in X minutes.
    accuracy Y %
    Ypred = [...]
    Y_test = [...]
    Y_test - Ypred = [...]
    
---

# Classificador per random forests
Aquest projecte s'ha dut a terme amb @isotopi i @BaosMiquel22

EL projecte consisteix en una implementació de classificador per random forests utilitzant la Programació Orientada a Objectes en Python, la qual es pot en diferents datasets (3 del quals inclosos en la implmentació).

Per tal d'usar-lo (en Linux), primer obrir una terminal i navegar al directory que contingui els fitxers de codi. Un cop fet això, escriure `python3 Client.py` a la terminal i pitjar Enter. Com s'ha esmentat anteriorment, el client es pot executar amb 3 datasets diferents (Iris, Sonar o MNIST). Per tal de triar quin volem usar, haurem de comentar/descomentar les línies de codi corresponents a cada dataset en el fitxer `Client.py` (s'hi especifica quines línies de codi són necessàries per a cada dataset). Si es vol usar el dataset Iris o el Sonar, caldrà també descomentar la línia 195 del fitxer `RandomForest.py`. Per defecte, el client usa el dataset MNIST.

Una cop executada la comanda mencionada anteriorment, es començaran a mostrar diversos missatges sobre l'estat de l'entrenament dels arbres i un cop finalitzat l'entrenament, es mostrarà el temps d'execució (en minuts), la precisió del classificador, la llista de prediccions (Ypred), la llista de valors esperats (Y_test) i una llista en que es compara el valor de la predicció amb el valor esperat (Y_test - Ypred), en el següent format:

    Program Executed in X minutes.
    accuracy Y %
    Ypred = [...]
    Y_test = [...]
    Y_test - Ypred = [...]

---

# Classificador por random forests
Este proyecto se ha realizado con @isotopi y @BaosMiquel22

El proyecto consiste en una implementación de classificador por random forests utilizando la Programación Orientada a Objetos en Python, la cual se puede usar con varios datasets (3 de ellos incluidos en la implementación).

Con tal de poder usar el classificador (en Linux), serà necesario abrir un aterminal y navegar al directorio que contenga los ficheros de codigo. Una vez allí, escribir `python3 Client.py` en la terminal y pulasr Enter. Como se ha mencionado anteriormente, el cliente se puede ejecutar con 3 datasets distintos (Iris, Sonar, MNIST). Con tal de escoger cual queremos usar, será necesario comentar/descomentar las linias de código correspondientes a cada dataset en el fichero `Client.py` (se especifica dentro del mismo qué linias són necesárias para cada dataset). Si se quiere usar el dataset Iris o el Sonar, será necesario también descomentar la línia 195 del fichero `RandomForest.py`. Por defecto, el cliente usa el dataset MNIST.

Una vez ejecutado el comando mencionado anteriormente, se comenzarán a mostrar varios mensajes sobre el estado del entrenamiento de los árboles i una vez finalizado el entrenamiento, se mostrará el tiempo de ejecución (en minutos), la precisión del classificador, la lista de predicciones (Ypred), la lista de valores esperados (Y_test) y una lista en la que se comparan los valores esperados con las predicciones (Y_test - Ypred), en el siguiente formato:

    Program Executed in X minutes.
    accuracy Y %
    Ypred = [...]
    Y_test = [...]
    Y_test - Ypred = [...]

---

### General remarks
By default, the classifier generates a forest of 10 trees, which produces a precision of around 90% accuracy on prediction with the MNIST dataset which is a fair result. You can also experience with different forest sizes and with other datasets than the ones that are included to see how it affects performance (both the accuracy and the time it takes to generate a result).
