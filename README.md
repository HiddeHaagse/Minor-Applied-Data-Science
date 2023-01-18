# Portfolio Minor 2022-2023 Applied Data Science
### Hidde Franke, 19086504, Toegepaste Wiskunde

## Intro
Ik ben de eerste twee periodes samen met groep 1 bezig geweest met het FoodBoost project en daarna de laatste 2 periodes aan het Cofano project gewerkt.  In dit portfolio zal ik de globaal belangrijkste onderwerpen omschrijven dat ik heb uitgevoerd. Eerst wordt het groepswerk beschreven en daarna mijn individuele toevoeging hieraan.

## Groepswerken
### Project 1: Foodboost
In het eerste project werkte we aan een model dat, aan de hand van een tag van een user, kon voorspellen of de user een gerecht lekker vind of niet. Hiervoor moest de gegeven data aangepast worden. De verschillende excel bladen zijn samengevoegd in één csv. Aangezien er geen data beschikbaar is van users hebben we deze zelf gemaakt; de zogenaamde "simulated users". De user krijgt een random tag mee en een aantal gerechten met deze tag die dan als lekker worden verklaard. Met deze simulated users kan de train matrix worden opgesteld. Hierin worden twee matrixen horizontaal gecombineerd. In de eerste matrix staat de pivottabel van de tags die in de gelieve gerechten van een user zitten. In de tweede matrix staan de tags van een gelieve gerecht (one out) en een random (ofwel niet lekker) gerecht van de user. Respectievelijk worden deze in een y kolom met 1 en 0 geclassificeerd; lekker en niet lekker. Verder zijn er nog twee kolommen voor visualisatie; Randomgerecht en One Out. Omdat de userdata van te voren is gesplit in train, validate en test data worden dan de hyperparameters voorspelt. De K-Nearestneighbor met k=7 komt op de beste validatie accuracy score uit; 0,795. Verder wordt er dan op de testdata een accuracy score van 0.79125.

### Project 2: Uitbreiding Foodboost
In de tweede periode van het Foodboost project zijn we van tags omgegaan naar ingredienten als informatie van de user. Om te focussen om een gerichter onderwerp hebben we alleen gekeken naar de gerechten met de tag "diner" en "hoofdgerecht", ofwel het avondeten. We gaan in deze uitbreiding kijken naar de keukens. Het model wat ontwikkeld is kan aan de hand van de gelieve keuken van de user herkennen aan de gerechten of het uit de gelieve keuken komt of niet. Omdat er 5006 verschillende ingredienten aangezig waren in de dataset hebben we besloten om de 500 meest voorkomende ingredienten in het model te verwerken. Doormiddel van permutaties vergelijken we met meerdere modellen (en dus meerdere validaties van hyperparameters) wat de beste accuracy score is per vergelijking. Door niet meer te kijken naar de combinatie van een keuken en 10 tags maar naar alleen keukens met een 1 op N vergelijking en de "diner" & "hoofdgerechten" tags is de accuracy score van ~55% naar ~99,7% geschoten.

### Project 3: Cofano
In de derde periode zijn we begonnen aan het Cofano project, of zoals wij het noemde: Het container project. Doordat we met de beschikbare dataset van de containers niet tot een geschikt project kwamen hebben we zelf iets gedacht. We zouden namelijk een haven zo efficiënt mogelijk moeten vullen met containers. De efficiëntie staat in dit probleem dan voor het minimale verbruik van tijd en aantal stackers. De stackers kunnen alleen een container via de lange zijde oppakken, dit beschouwen we als een horizontale richting; je kan ze van links en rechts pakken. Het is de bedoeling dat we een simulatie bouwen waar doormiddel van een model een efficiënte manier voor dit probleem bedacht kan worden. ***~Aangezien de volgorde van de containers niet altijd vast staat is het aan het model om de containers op posities te plaatsen waarbij aan het eind de stacker zonder (of met zo min mogelijk) extra verplaatsingen bij alle benodigde containters kan. Elk nummer van de container hoor dan ook bij een schipnummer. Er wordt verwacht dat schip 1 als eerste binnenkomt en schip 5 als laatste.~*** 
We hebben een reinforcement learning model gemaakt waarbij de environment een 3 bij 3 matrix is. Het model pakt uit de lijst met containers een container en plaatst deze zo in de grid (ofwel matrix). We kwam erachter dat het reward systeem erg gevoelig was dus hebben we allemaal een eigen bedacht. Na vergelijken van de modellen kwamen we uiteindelijk uit op één model.

### Project 4: Uitbreiding Cofano
Nadat het vullen van een 3 bij 3 matrix was gelukt gingen we proberen de grid te veranderen. De ene helft van de groep ging proberen het 2D vlak te vergroten door bijvoorbeeld een 5 bij 5 matrix zo efficient mogelijk te vullen en de andere helft ging proberen een 3D model te maken waarbij er boxen op elkaar geplaatst konden worden. Dit bleek een lastige opdracht te zijn.


- Domain Knowledge (Literature, jargon, evaluation, existing data sets, ...)
- Predictive Models
- Data preparation
- Data Visualization
- Data collection
- Evaluation
- Diagnostics of the learning process
- Communication (presentations, summaries, paper, ...)
- Link to the Python Notebooks you have finished (you can dump them to PDF)
- List the tickets from the Scrum backlog that you worked on, linked to deliverables, own experiments, etc.
- Add any other assignment you feel is evidence of your abilities



# Portfolio
- Reflectie contributie groepsprojecten
- Reflectie eigen leerobjecten


## Predictive Analytics
[Notebook](www.google.com)

Aangezien we het Foodboost project als een lekker of niet lekker probleem hebben gedefiniëerd, moet er een classificatie model worden gemaakt. Via literatuur ben ik op een paar classificatiemodellen gekomen [link](https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501). Hierbij heb ik gebruik gemaakt van verschillende modellen, waaronder logistische regressie, k-nearest neighbors, support vector machines, random forest classifiers en gaussian naive bayes. Maar eerst heb ik de data aangepast zodat dit gebruikt kon worden voor het trainen van de modellen ([Zie Data Preprocessing](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/README.md#data-preprocessing))).

[Modellen uitleggen en verschil vertellen]

Vervolgens heb ik maatregelen genomen om over- en underfitting te voorkomen. Dit heb ik gedaan door gebruik te maken van meerdere GridSearchCV's voor het tunen van hyperparameters. Dit helpt om de prestaties van de modellen te verbeteren en te voorkomen dat de modellen te veel aangepast worden aan de training data.

Daarnaast heb ik verschillende modellen met elkaar vergeleken en de verschillen tussen de modellen verklaard. Dit helpt om een beeld te krijgen van       de prestaties van de verschillende modellen en om te bepalen welk model het beste geschikt is voor het specifieke probleem. De accuraatheid score zegt al heel veel in de validatie maar toch maak ik een validation curve. KNearest Neighbors komt het beste uit de test. De parameters 'leaf_size' en 'p' bleven bij alle validaties altijd hetzelfde dus heb ik een extra validation curve gemaakt van de parameter 'n_neighbors'.


## Domain Knowledge


## Data Preprocessing
Maar voordat ik iets kan aanbevelen moest ik eerst de data simuleren, aangezien die er niet is (zie kopstuk Define Users). 


## Communication


- Evalutatie groepsproject als geheel


