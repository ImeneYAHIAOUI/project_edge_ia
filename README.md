# Rendu Projet microIA

# Description du projet

Notre projet vise à identifier les chants d'oiseaux à partir de fichiers de test en utilisant une carte Nucleo-64 STM32L476 qui possède une faible quantité de RAM et de stockage (1 MB Flash, 128 KB SRAM) pour jouer les enregistrements, et à afficher les résultats dans la console. Nous avons construit notre base de données en récupérant des enregistrement grâce à l’API du site Xeno-canto, présentée pendant le cours, pour atteindre cet objectif.

# Description du flux du travail

Nous avons choisi d’avoir quatre classes dans notre base de données pour quatre espèces (Bruant jaune, Bruant zizi, Coucou gris et Gobemouche gris)

Nous avons récupéré tous les enregistrements de chants de qualité A et B pour les toutes les espèces et les chants de qualité C pour le Gobemouche gris car il n’a pas autant d’enregistrements de bonne qualité que les autres espèces.

Pour cela, Nous avons utilisé la librairie python [xeno-canto](https://pypi.org/project/xeno-canto/) qui est un wrapper d’api désigner afin d’aider les utilisateurs de récupérer les données souhaitées de xeno-canto.org.

Cette api nous a permis de récupérer des enregistrements de tailles différentes en format mp3. Nous avons donc utilisé ffmpeg pour découper ses derniers en segments de 10 secondes et les convertir en wav afin qu’on puisse extraire les données avec la librairie python wave.

Une fois que nos données sont pret

# Résultats obtenus

Dans un premier temps, Nous avons utilisé seulement les premiers 10 secondes de chaque enregistrement pour l’apprentissage et le test. Cette stratégie bous faisait perdre beaucoup d’informations et nous donnais une accuracy de 44%.

Cependant, après avoir utilisé le logiciel ffmpeg pour découper chaque audio en segments de 10 secondes et en utiliser la totalité, nous avons observé une amélioration significative de l'accuracy, qui est passée à 60%. Cette modification a permis d'inclure une quantité de données plus importante dans l'apprentissage du modèle, ce qui a contribué à améliorer ses performances.

En faisant un premier essaie avec toutes les classe, Nous avons remarqué que le model confondait le plus le bruant jaune avec le bruant zizi.

En écoutant quelques enregistrements, on peut remarquer qu’ils ont un chant similaire

![Untitled](Rendu%20Projet%20microIA%204654df37914c493f8412a7a83eae0ef8/Untitled.png)

Nous avant donc fait un autre essaie sans la classe du bruant jaune et l’accuracy est monté à 70%.

![Untitled](Rendu%20Projet%20microIA%204654df37914c493f8412a7a83eae0ef8/Untitled%201.png)

En effet, un des difficultés de faire apprendre une ia à reconnaitre les chants des oiseaux est la similarité de chants de plusieurs espèces. 

Une autre difficulté est le fait que les audios sont souvent “polluée” par du bruit de font ou de chants d’autres espèces. Il y a aussi le fait que les oiseaux ont souvent plusieurs chants différents ( le “si yut-tee yut-tee” joyeux, le “te tuuiii" gazouillant, le “yun-yun-yun” alarmant, etc).

Le dernier problème que nous avons rencontré est le déséquilibre du nombre d’enregistrements entre les espèces. Cela est du au fait que certaines sont plus populaires que d’autres, et certaines sont plus rare  que d’autre. Pour donner un exemple, nous avons décidé de ne pas inclure le  Faucon crécerelle dans notre base de données car nous avons trouvé seulement une soixantaine d’enregistrements de chants pour cette espèce.

Nous sommes conscients que ces résultats ne sont pas encore à la hauteur de nos attentes. Toutefois, compte tenu du nombre limité de filtres que nous avons utilisés dans les couches de convolution et la baisse de la frequence et des secondes dans l’entrée pour pouvoir déployer le modèle sur la carte, cette performance reste tout à fait satisfaisante.

# Conclusion

Pour conclure, notre projet a été une réussite et nous a permis d'acquérir de nouvelles compétences en traitement de données. Toutefois, pour améliorer encore davantage la précision de notre modèle d'identification des chants d'oiseaux, il serait intéressant de mettre en place un mécanisme de seuil de détection de niveau sonore. Ce dernier permettrait de détecter les parties du signal audio inutiles, telles que les silences entre les enregistrements ou les bruits de fond, et de les supprimer automatiquement. En éliminant ces parties superflues du signal, nous pourrions optimiser notre modèle et ainsi améliorer l'identification des chants d'oiseaux.
