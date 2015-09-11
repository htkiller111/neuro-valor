# Wizja #

System będzie używał algorytmu ewolucyjnego do znalezienia optymalnej struktury sieci neuronowej, która po wytrenowaniu będzie potrafiła przewidzieć najbliższe zmiany notowań walorów giełdowych.


# Założenia #

Wstępne badanie dziedziny problemu pozwoliło ustalić, że:
  * Użytym modelem sieci neuronowej będzie najprawdopodobniej feed-forward z uczeniem back-propagation (samodzielnie implementowana).
  * Ewolucja sieci będzie polegała na modyfikacji ilości ukrytych warstw oraz obecnych na nich neuronów (oraz ewentualnie na zmianie funkcji aktywacji niektórych neuronów).
  * Wejściem sieci będzie N ostatnich wycen waloru (np. w ujęciu dziennym, tygodniowym lub miesięcznym) poddane odpowiedniemu pre-processingowi. Wartość N również może podlegać ewolucji (jeszcze o tym nie zdecydowano).
  * Pre-processing będzie polegał na znalezieniu pochodnej przebiegu wycen, a następnie przeskalowaniu jej i przesunięciu w obszar 0.0 - 1.0.
  * Sieć będzie udzielała odpowiedzi reprezentującej poziom najbliższej zmiany kursu waloru (0.0 to największy spadek, 0.5 to brak zmian, a 1.0 to największy wzrost - identycznie jak dla danych wejściowych).
  * Sieć może być trenowana z użyciem K historycznych wycen konkretnego waloru (wtedy wyspecjalizuje się w danym walorze), lub z użyciem wycen wielu walorów z konkretnego przedziału czasu (wtedy pozyska informacje o aktualnej ogólnej sytuacji rynku).