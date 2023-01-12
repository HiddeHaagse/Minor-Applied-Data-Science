# Portfolio Minor 2022-2023 Applied Data Science
### Hidde Franke, 19086504, Toegepaste Wiskunde

## Intro
Ik ben de eerste twee periodes samen met groep 1 bezig geweest met het FoodBoost project en daarna de laatste 2 periodes aan het Cofano project gewerkt.  In dit portfolio zal ik de globaal belangrijkste onderwerpen omschrijven dat ik heb uitgevoerd. Eerst wordt het groepswerk beschreven en daarna mijn individuele toevoeging hieraan.

## Groepswerken
### Project 1: Foodboost
In het eerste project werkte we aan een model dat, aan de hand van een tag van een user, kon voorspellen of de user een gerecht lekker vind of niet. Hiervoor moest de gegeven data aangepast worden. De verschillende excel bladen zijn samengevoegd in één csv. Aangezien er geen data beschikbaar is van users hebben we deze zelf gemaakt; de zogenaamde "simulated users". De user krijgt een random tag mee en een aantal gerechten met deze tag die dan als lekker worden verklaard. Met deze simulated users kan de train matrix worden opgesteld. Hierin worden twee matrixen horizontaal gecombineerd. In de eerste matrix staat de pivottabel van de tags die in de gelieve gerechten van een user zitten. In de tweede matrix staan de tags van een gelieve gerecht (one out) en een random (ofwel niet lekker) gerecht van de user. Respectievelijk worden deze in een y kolom met 1 en 0 geclassificeerd; lekker en niet lekker. Verder zijn er nog twee kolommen voor visualisatie; Randomgerecht en One Out. Omdat de userdata van te voren is gesplit in train, validate en test data worden dan de hyperparameters voorspelt. De K-Nearestneighbor met k=7 komt op de beste validatie accuray score uit; 0,795. Verder wordt er dan op de testdata een accuray score van 0.79125.

### Project 2: Uitbreiding Foodboost
In de tweede periode van het Foodboost project zijn we van tags omgegaan naar ingredienten als informatie van de user.

### Project 3: Cofano

### Project 4: Uitbreiding Cofano



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



## Portfolio
- Reflectie contributie groepsprojecten
- Reflectie eigen leerobjecten
  - Predictive Analytics

    Met het lezen van het csv bestand begonnen we ieder met het uitwerken van een simpel model om te begrijpen hoe een train, (validate,) test model in       elkaar zit. Een voorbeeld hiervan is een [lineair regressie model voor het voorspellen van het aantal gram verzadigd vet in een gerecht op basis van     de hoeveelheid gram vet in het gerecht](https://github.com/HiddeHaagse/Minor-Applied-Data-Science/blob/main/Lineair%20regressie%20model%20vet%20en%20verzadigd%20vet.png).
  - Domain Knowledge
  - Data Preprocessing
  - Communication
- Evalutatie groepsproject als geheel


