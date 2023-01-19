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

## Research Project



## Predictive Analytics
[Notebook](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/Portfolio%20(Predictive%20Analytics).ipynb)

Aangezien we het Foodboost project als een lekker of niet lekker probleem hebben gedefiniëerd, moet er een classificatie model worden gemaakt. Via literatuur ben ik op een paar classificatiemodellen gekomen [link](https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501). Hierbij heb ik gebruik gemaakt van verschillende modellen, waaronder logistische regressie, k-nearest neighbors, support vector machines, random forest classifiers en gaussian naive bayes. Maar eerst heb ik de data aangepast zodat dit gebruikt kon worden voor het trainen van de modellen ([Zie Data Preprocessing](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/README.md#data-preprocessing))).

### Modellen
-Logistic Regression is een klassieke en veelgebruikte methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een lineaire methode die een goede prestatie levert voor kleine datasets.

-K-Nearest Neighbors is een eenvoudige en effectieve methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een niet-parametrische methode die geschikt is voor kleine datasets.

-Support Vector Machine is een veelgebruikte methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een niet-lineaire methode die een goede prestatie levert voor grote datasets.

-Random Forest is een veelgebruikte methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een ensemble methode die een goede prestatie levert voor grote datasets.

-Gaussian Naive Bayes is een eenvoudige en veelgebruikte methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een probabilistische methode die een goede prestatie levert voor kleine datasets.

Daarnaast heb ik verschillende modellen met elkaar vergeleken en de verschillen tussen de modellen verklaard. Dit helpt om een beeld te krijgen van       de prestaties van de verschillende modellen en om te bepalen welk model het beste geschikt is voor het specifieke probleem. De accuraatheid score zegt al heel veel in de validatie maar toch maak ik een validation curve. KNearest Neighbors komt het beste uit de test. De parameters 'leaf_size' en 'p' bleven bij alle validaties altijd hetzelfde dus heb ik een extra validation curve gemaakt van de parameter 'n_neighbors'.


## Domain Knowledge


## Data Preprocessing
[Notebook](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/Portfolio%20(Data%20Preprocessing).ipynb) 

*Tip: U kunt zijwaards scrollen bij de matrixen*

Binnen het onderwerp "Data Preprocessing heb ik bij het Foodboost project creatieve daden verricht. In [Predictive Analytics](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/README.md#predictive-analytics) staat het model van het project omschreven. Maar voordat ik met een model iets kan aanbevelen moest ik eerst de data simuleren, en aangezien die er voor de aanbeveling niet is moest ik deze zelf maken.
De code definieert een functie genaamd "User_Favo_Random_Tags" die een dataset maakt voor een systeem voor het aanbevelen van voedingsrecepten. De functie neemt twee parameters aan: "randomTag", dat een willekeurig geselecteerde tag is uit een "tagsdf" databestand, en "K" dat het minimum aantal gerechten is dat geassocieerd moet zijn met de "randomTag". De functie gebruikt de "random" module om willekeurig een tag te selecteren die ten minste "K" aantal gerelateerde gerechten heeft. De functie selecteert vervolgens "K" aantal willekeurige gerechten die geassocieerd zijn met de geselecteerde tag en splitst deze in een trainingsset en een testset (respectievelijk 80% en 20%). De functie maakt ook een lijst met tags die zijn geassocieerd met de geselecteerde gerechten, en een set van "K" aantal willekeurige gerechten die niet geassocieerd zijn met de geselecteerde tag. Deze gerechten worden ook gesplitst in een trainingsset en een testset (80% en 20%). De train & testset wordt hier gebruikt vanuit de Leave one out.


Voor de trian, validate en test data maak ik een matrix met behulp van de pandas bibliotheek die is gevuld met nullen, waarbij de kolommen de lijst met tags zijn en de rijen de gebruikers zijn. De functie maakt ook een tweede matrix, vergelijkbaar met de eerste, maar met een ander kolomvoorvoegsel. De functie maakt vervolgens een numpy-array van de lijst met tags van de gebruiker en gebruikt de LeaveOneOut methode van sklearn.model_selection om over de train- en testindexen te itereren. De eerste matrix wordt als pivottable gebruikt door de tags van de **gelieve** recepten op te slaan. De tweede matrix gebruikt ook de pivottable van de tags van de recepten, maar de even rijen staan de gelieve recepten van de Leave One Out zijn, en in de oneven rijen is zijn het de tags van een recept dat niet gelieve is.
In de code werkt dit als volgt: de functie splitst de lijst met tags van de gebruiker in train- en testsets, en voor elke iteratie, haalt het het eerste element uit de randomTags en randomRecipes lijsten (De recepten die niet lekker worden gevonden door de user). De functie gebruikt vervolgens de unieke tags om de pivot tabel/matrix te vullen. Tenslotte geeft de functie de train- en test sets van geselecteerde gerechten, hun bijbehorende tags, willekeurige tags, "K", en de test sets van niet-geassocieerde gerechten en hun bijbehorende tags terug.

De Users worden geplitst in TrainUsers (60%), ValidateUsers (20%) en TestUsers (20%) zodat deze op de juiste plekken in het trainen, valideren en testen van het model kunnen worden gebruikt. Op het einde worden de kolommen "Randomgerecht" en "One out" uit de dataset gehaald aangezien deze alleen voor visualisatie van de train dataset was.


## Communication


- Evalutatie groepsproject als geheel


