# PR #3: Feature/activation function tuning

Questa PR introduce l'infrastruttura C++ / MPI per eseguire cross-validazione sulla scelta della funzione di attivazione e misura le performance delle diverse non linearità (ReLU, LeakyReLU, Tanh, Sigmoid) per l'architettura base.

## Modifiche
- Allineato il `kGlobalBatchSize` a 64 e impostate 100 epoche.
- **Bug Alpha Corretto**: La funzione `make_blueprint` precedentemente ometteva il passaggio del parametro `alpha` a `Recipes::activation`, fissando `leaky_alpha` sempre a 0.05. Il parametro è stato esposto, permettendo di testare varianti.
- Estesa la griglia dei candidati in `main.cpp` testando i valori `alpha` per la LeakyReLU: `{0.01, 0.05, 0.1, 0.2, 0.3}`.
- Aggiornato ai nuovi campioni verificati generati con `seed 42`.
- Implementata la logica ad orchestratore esterno per processare i candidati indipendentemente e prevenire i crash da timeout hardware lunghi.

## Risultati
Tutti gli esperimenti sono stati eseguiti con successo sull'hardware locale. 
Le valutazioni appaiate rispetto alla base (`leakyrelu-0.05`) hanno dimostrato che **nessun'altra funzione di attivazione o parametro $\alpha$ riesce a battere in modo robusto il baseline**.
Sebbene `leakyrelu-0.01` mostri 4 fold migliorativi e un MSE leggermente inferiore in media, la differenza appaiata media (-0.000112) non supera rigorosamente la sua deviazione standard (0.000116).
Di conseguenza, il vincitore resta `leakyrelu-0.05`.

**Stato**: Il codice, l'infrastruttura e i README sono pronti. Risultati definitivi raccolti e documentati! Pronta per la review.
