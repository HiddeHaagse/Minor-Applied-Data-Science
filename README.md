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
In de derde periode zijn we begonnen aan het Cofano project, of zoals wij het noemde: Het container project. Doordat we met de beschikbare dataset van de containers niet tot een geschikt project kwamen hebben we zelf iets gedacht. We zouden namelijk een haven zo efficiënt mogelijk moeten vullen met containers. De efficiëntie staat in dit probleem dan voor het minimale verbruik van tijd en aantal stackers. De stackers kunnen alleen een container via de lange zijde oppakken, dit beschouwen we als een horizontale richting; je kan ze van links en rechts pakken. Het is de bedoeling dat we een simulatie bouwen waar doormiddel van een model een efficiënte manier voor dit probleem bedacht kan worden. Aangezien de volgorde van de containers niet altijd vast staat is het aan het model om de containers op posities te plaatsen waarbij aan het eind de stacker zonder (of met zo min mogelijk) extra verplaatsingen bij alle benodigde containters kan. Elk nummer van de container hoor dan ook bij een schipnummer. Er wordt verwacht dat schip 1 als eerste binnenkomt en schip 3 als laatste. 
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

Aangezien we het Foodboost project als een lekker of niet lekker probleem hebben gedefiniëerd, moet er een classificatie model worden gemaakt. Via het atrikel van *Gong (2022)* ben ik op een paar classificatiemodellen gekomen. Hierbij heb ik gebruik gemaakt van verschillende modellen, waaronder logistische regressie, k-nearest neighbors, support vector machines, random forest classifiers en gaussian naive bayes. Maar eerst heb ik de data aangepast zodat dit gebruikt kon worden voor het trainen van de modellen ([Zie Data Preprocessing](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/README.md#data-preprocessing))).

### Modellen
-Logistic Regression is een klassieke en veelgebruikte methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een lineaire methode die een goede prestatie levert voor kleine datasets.

-K-Nearest Neighbors is een eenvoudige en effectieve methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een niet-parametrische methode die geschikt is voor kleine datasets.

-Support Vector Machine is een veelgebruikte methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een niet-lineaire methode die een goede prestatie levert voor grote datasets.

-Random Forest is een veelgebruikte methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een ensemble methode die een goede prestatie levert voor grote datasets.

-Gaussian Naive Bayes is een eenvoudige en veelgebruikte methode voor het oplossen van problemen met afhankelijke variabelen die categorisch zijn. Het is een probabilistische methode die een goede prestatie levert voor kleine datasets.

Daarnaast heb ik verschillende modellen met elkaar vergeleken en de verschillen tussen de modellen verklaard. Dit helpt om een beeld te krijgen van de prestaties van de verschillende modellen en om te bepalen welk model het beste geschikt is voor het specifieke probleem. De accuraatheid score zegt al heel veel in de validatie maar toch maak ik een validation curve. KNearest Neighbors komt het beste uit de test. De parameters 'leaf_size' en 'p' bleven bij alle validaties altijd hetzelfde dus heb ik een extra validation curve gemaakt van de parameter 'n_neighbors'.


## Domain Knowledge
[Notebook](www.google.com)

### Introductie Cofano Project
Optimalisatie van containerplaatsing is een cruciaal aspect van logistiek en supply chain management, met name in de context van haventerminals. De efficiënte plaatsing en organisatie van containers is essentieel voor een soepele en tijdige bedrijfsvoering in deze faciliteiten. Maar het vinden van de optimale plaatsing van containers in een haventerminal kan een uitdaging zijn, omdat er rekening moet worden gehouden met factoren zoals toegankelijkheid, stapeltoepassing, tijdsbestek en ruimtebenutting.

Een bedrijf dat voor in de rij staat van het aanpakken van deze uitdaging is Cofano. Cofano is een toonaangevende leverancier van oplossingen voor containerbeheer voor haventerminals. Nu is er met een simulatie een werkend prototype ontwikkeld voor optimalisatie van containerplaatsing met behulp van Reinforcement Learning (RL).

RL is een type machine learning-algoritme dat agents in staat stelt om te leren van hun interacties met de omgeving. In de context van containerplaatsing kan een RL-agent een optimale plaatsingsstrategie leren door middel van vallen en opstaan (trail and error), door beloningen of straffen te ontvangen op basis van de uitkomst van zijn acties.

Cofano's oplossing maakt gebruik van een RL-agent om de plaatsing van containers in een haventerminal te optimaliseren. De agent is getraind om factoren zoals toegankelijkheid, stapeltoepassing, tijdsbestek en ruimtebenutting in acht te nemen, om de meest optimale plaatsing van containers te vinden, waar de stacker elke container kan pakken die hij wil zonder extra stappen te doen. De stacker pakt containers op de horizontale as. Wanneer het model een container op een plek plaatst waar al een container staat of een gelieve container inboxt, krijgt het een negatieve beloning. Zodra de agent een container goed plaatst, bijvoorbeeld naast of op een andere gelieve container, krijgt het een positieve belonging. Nu heb ik het model nog twee extra manieren gegeven om een hogere score te behalen. Dit is wanneer de agent een volledige rij vult met dezelfde containers en het krijgt een minpunt als de rij met meer dan 2 verschillende containers is gevuld. Dit zal de variatie per rij laag houden en dus dezelfde containers dicht bij elkaar zetten. Zie voor verdere uitwerking de [Notebook](www.google.com). Zo leert het model om de haventerminal zo efficiënt mogelijk in te richten.

***~~Aan het model wordt ook een observation space meegegeven, deze observation space bestaat uit de environment waarin wordt gewerkt, voor een drie bij drie matrix gaat het om een box van drie bij drie, en het nummer van de container die geplaatst gaat worden.***
-Action Space
-Inboxen vindt plaats als er aan beide lange zijdes (ook bij een lagere stapel), van een container, een container staat van een ander nummer. De stacker kan niet meteen bij de geïnitieerde container, maar heeft daar een extra stap voor nodig.
Daarnaast is het belangrijk dat er geen gaten ontstaan tussen containers bij het plaatsen van de containers.
~~***


### Literatuur
In dit literatuuronderzoek zullen we de keuzes die zijn gemaakt in het Cofano project onderbouwen met behulp van relevante literatuur. Zie [bibliografie](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/README.md#bibliografie).

Eerst en vooral, heb ik gekozen voor het gebruik van reinforcement learning als techniek voor het oplossen van het container plaatsingsprobleem. Dit komt overeen met de aanbevelingen van *Jiang, Wang, Yan, and Zeng (2021)* die een nieuwe heuristische reinforcement learning methode introduceren voor het containerrelocatieprobleem, en *Hu, Wang, Xiao, and Yang (2023)* die een multi-agent reinforcement learning benadering gebruiken voor anti-conflict AGV padplanning in geautomatiseerde containerterminals.

Ten tweede, heb ik ervoor gekozen om een beloning te geven voor het plaatsen van containers op lege plekken en een negatieve beloning voor het plaatsen van containers op plekken waar al een container staat. Dit is consistent met de aanbevelingen van *Hu, Yang, and Zeng (2011)* die een methode integreren van simulatie en reinforcement learning voor de planning van operationele inzet in containerterminals.

Daarnaast, heb ik extra beloningen en straffen geïmplementeerd voor het plaatsen van containers naast containers van dezelfde soort, of het plaatsen van containers tussen twee ongelijke containers. Dit is consistent met de aanbevelingen van *Shi, W. (2021)* die een nieuw algoritme voorgesteld voor het oplossen van het container pre-marshalling probleem en *Ghedira, Kefi, Korbaa, and Yim (2010)* die een heuristische gebaseerde model introduceren voor het containerstapelingsprobleem.

Om deze "steps" te evalueren heb ik voor een Proximal Policy Optimization (PPO) agent gekozen voor de stabiliteit van de clip functie bij het trainen. Op aanbeveling van het experiment van *~~Duan, Chen, Houthooft, Schulman, Abbeel (2016)~~* heb ik geen Advantage Actor-Critic (A2C) agent gebruikt omdat PPO simpelweg beter presteert. Het verschil is dat het A2C-model agressiever zoekt naar een verbetering. Dit zien we dan ook terug in de value loss van het A2C-model vergeleken met het PPO-model. Uiteindelijk heb ik beide modellen uitgeprobeerd en is er voor dit project gekozen voor het PPO-model omdat hier de beste resultaten uit voort kwamen.

Tot slot, heb ik een random containerlijst geïmplementeerd met de soorten containers die beschikbaar zijn voor plaatsing. Dit is consistent met de aanbevelingen van *Euchi, Moussi, Ndiaye, and Yassine (2016)* die een ant colony optimization benadering gebruiken voor het oplossen van het containerstapelingsprobleem in Le Havre Seaport Terminal.


## Data Preprocessing
[Notebook](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/Portfolio%20(Data%20Preprocessing).ipynb) 

*Tip: U kunt zijwaards scrollen bij de matrixen*

Binnen het onderwerp "Data Preprocessing heb ik bij het Foodboost project creatieve daden verricht. In [Predictive Analytics](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/README.md#predictive-analytics) staat het model van het project omschreven. Maar voordat ik met een model iets kan aanbevelen moest ik eerst de data simuleren, en aangezien die er voor de aanbeveling niet is moest ik deze zelf maken.
De code definieert een functie genaamd "User_Favo_Random_Tags" die een dataset maakt voor een systeem voor het aanbevelen van voedingsrecepten. De functie neemt twee parameters aan: "randomTag", dat een willekeurig geselecteerde tag is uit een "tagsdf" databestand, en "K" dat het minimum aantal gerechten is dat geassocieerd moet zijn met de "randomTag". De functie gebruikt de "random" module om willekeurig een tag te selecteren die ten minste "K" aantal gerelateerde gerechten heeft. De functie selecteert vervolgens "K" aantal willekeurige gerechten die geassocieerd zijn met de geselecteerde tag en splitst deze in een trainingsset en een testset (respectievelijk 80% en 20%). De functie maakt ook een lijst met tags die zijn geassocieerd met de geselecteerde gerechten, en een set van "K" aantal willekeurige gerechten die niet geassocieerd zijn met de geselecteerde tag. Deze gerechten worden ook gesplitst in een trainingsset en een testset (80% en 20%). De train & testset wordt hier gebruikt vanuit de Leave one out.


Voor de trian, validate en test data maak ik een matrix met behulp van de pandas bibliotheek die is gevuld met nullen, waarbij de kolommen de lijst met tags zijn en de rijen de gebruikers zijn. De functie maakt ook een tweede matrix, vergelijkbaar met de eerste, maar met een ander kolomvoorvoegsel. De functie maakt vervolgens een numpy-array van de lijst met tags van de gebruiker en gebruikt de LeaveOneOut methode van sklearn.model_selection om over de train- en testindexen te itereren. De eerste matrix wordt als pivottable gebruikt door de tags van de **gelieve** recepten op te slaan. De tweede matrix gebruikt ook de pivottable van de tags van de recepten, maar de even rijen staan de gelieve recepten van de Leave One Out zijn, en in de oneven rijen is zijn het de tags van een recept dat niet gelieve is.
In de code werkt dit als volgt: de functie splitst de lijst met tags van de gebruiker in train- en testsets, en voor elke iteratie, haalt het het eerste element uit de randomTags en randomRecipes lijsten (De recepten die niet lekker worden gevonden door de user). De functie gebruikt vervolgens de unieke tags om de pivot tabel/matrix te vullen. Tenslotte geeft de functie de train- en test sets van geselecteerde gerechten, hun bijbehorende tags, willekeurige tags, "K", en de test sets van niet-geassocieerde gerechten en hun bijbehorende tags terug.

De Users worden geplitst in TrainUsers (60%), ValidateUsers (20%) en TestUsers (20%) zodat deze op de juiste plekken in het trainen, valideren en testen van het model kunnen worden gebruikt. Op het einde worden de kolommen "Randomgerecht" en "One out" uit de dataset gehaald aangezien deze alleen voor visualisatie van de train dataset was.


## Communication
[Link Scrum Board](https://trello.com/invite/b/KkauUOva/ATTIa5bb61e1949ad898fb9c1cc8f9fc3b3f095EF43D/applied-data-science)

- Evalutatie groepsproject als geheel


# Bibliografie
*Gong, D. (2022, July 12). Top 6 Machine Learning Algorithms for Classification. Medium. [Link](https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501).*

*Jiang, T., Zeng, B., Wang, Y., & Yan, W. (2021, April). A New Heuristic Reinforcement Learning for Container Relocation Problem. Journal of Physics: Conference Series (Vol. 1873, No. 1, p. 012050). IOP Publishing. doi: [10.1088/1742-6596/1873/1/012050](https://iopscience.iop.org/article/10.1088/1742-6596/1873/1/012050)*

*Hu, H., Wang, F., Xiao, S., Yang, X. (2023). Anti-conflict AGV Path Planning in Automated Container Terminals Based on Multi-agent Reinforcement Learning. International Journal of Production Research, 61(1), 65-80. doi: [10.1080/00207543.2021.1998695](https://www.researchgate.net/publication/356272643_Anti-conflict_AGV_path_planning_in_automated_container_terminals_based_on_multi-agent_reinforcement_learning)*

*Hu, X., Yang, Z., Zeng, Q. (2011) A Method Integrating Simulation and Reinforcement Learning for Operation Scheduling in Container Terminals. Transport, 26(4), 383-393. doi: [10.3846/16484142.2011.638022](https://doi.org/10.3846/16484142.2011.638022)*

*Shi, W. (2021). A New Algorithm for the Container Pre-marshalling Problem. International Core Journal of Engineering, 7(8), 20-24, doi: [10.6919/ICJE.202108_7(8).0004](https://doi.org/10.6919/ICJE.202108_7(8).0004)* 

*Kefi, M., Korbaa, O., Ghedira, K., & Yim, P. (2007). Heuristic-based model for container stacking problem. In 19th International Conference on Production Research-ICPR (Vol. 7). [Link](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=38de975bf0b50cea4daa83606bee945994c91f34).*

*~~Duan, Y., Chen, X., Houthooft, R., Schulman, J., & Abbeel, P. (2016, June). Benchmarking deep reinforcement learning for continuous control. In International conference on machine learning (pp. 1329-1338). PMLR.~~*

*Euchi, J., Moussi. R., Ndiaye, F., Yassine, A (2016). Ant Colony Optimization for Solving the Container Stacking Problem: Case of Le Havre (France) Seaport Terminal. International Journal of Applied Logistics, 6(2), 81-101. doi: [10.4018/IJAL.2016070104](https://www.researchgate.net/publication/308969102_Ant_Colony_Optimization_for_Solving_the_Container_Stacking_Problem_Case_of_Le_Havre_France_Seaport_Terminal)*





Krishna, V., Sudhir, Y. (2020). Comparison of Reinforcement Learning Algorithms [Powerpoint-slides]. Departure of Computere Science and Engeneering, University at Buffalo. Geraadpleegd op 28 november 2022, van https://cse.buffalo.edu/~avereshc/rl_fall20/  



