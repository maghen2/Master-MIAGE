#markov_drh.py
import numpy as np
import numpy.linalg

# Définition de la matrice de transition initiale (avec état "Extérieur" absorbant)

P_absorbant = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.6, 0.1, 0.0, 0.0, 0.3],
    [0.0, 0.0, 0.5, 0.2, 0.0, 0.3],
    [0.0, 0.0, 0.0, 0.5, 0.2, 0.3],
    [0.0, 0.0, 0.0, 0.0, 0.6, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
])

etats_absorbant = ['Niveau A', 'Niveau B', 'Niveau C', 'Niveau D', 'Niveau E', 'Extérieur']

print("Matrice de transition avec état absorbant 'Extérieur':")
print(P_absorbant)
print("\nÉtats:", etats_absorbant)

# 1. Calcul des états transients, récurrents et absorbants (théorique, déjà discuté précédemment)
# Code pour identifier automatiquement (non trivial pour cet exemple simple, mais conceptuellement possible)
# Dans cet exemple, nous les avons déjà identifiés théoriquement.


# 2. Calcul de la distribution stationnaire pour la chaîne avec état absorbant
#   Pour une chaîne avec état absorbant, la distribution stationnaire est concentrée
#   sur l'état absorbant. Nous pouvons vérifier cela numériquement (mais trivial ici).

def distribution_stationnaire_absorbante(P):
    """
    Calcule la distribution stationnaire pour une chaîne de Markov avec état absorbant.
    Pour ce type de chaîne, la distribution stationnaire est souvent concentrée sur les états absorbants.
    """
    n_etats = P.shape[0]
    # Pour un état absorbant, la distribution stationnaire tend vers [0, 0, ..., 1, ..., 0] où le 1 est à la position de l'état absorbant.
    # Pour notre cas, l'état absorbant est le dernier (index 5, état 'Extérieur').
    pi_stationnaire = np.zeros(n_etats)
    pi_stationnaire[-1] = 1.0
    return pi_stationnaire

pi_absorbant_stationnaire = distribution_stationnaire_absorbante(P_absorbant)
print("\nDistribution stationnaire pour la chaîne avec état absorbant:")
print(pi_absorbant_stationnaire)
print(dict(zip(etats_absorbant, pi_absorbant_stationnaire)))


# Définition de la matrice de transition modifiée (tous états récurrents)

P_recurrent = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0],
    [0.3, 0.6, 0.1, 0.0, 0.0],
    [0.3, 0.0, 0.5, 0.2, 0.0],
    [0.3, 0.0, 0.0, 0.5, 0.2],
    [0.4, 0.0, 0.0, 0.0, 0.6]
])

etats_recurrent = ['Niveau A', 'Niveau B', 'Niveau C', 'Niveau D', 'Niveau E']

print("\nMatrice de transition modifiée (tous états récurrents):")
print(P_recurrent)
print("\nÉtats:", etats_recurrent)


# 3. Calcul de la distribution stationnaire pour la chaîne modifiée (tous états récurrents)

def distribution_stationnaire_recurrent(P):
    """
    Calcule la distribution stationnaire pour une chaîne de Markov irréductible et apériodique.
    Utilise la méthode de résolution du système d'équations linéaires: pi * P = pi  et somme(pi) = 1.
    """
    n_etats = P.shape[0]
    # Construction de la matrice pour résoudre (P^T - I) * pi^T = 0
    A = P.T - np.eye(n_etats)
    # Ajout de la contrainte de normalisation: somme des pi_i = 1 (on remplace une ligne par [1, 1, ..., 1])
    A[-1, :] = np.ones(n_etats)
    b = np.zeros(n_etats)
    b[-1] = 1  # Le dernier élément de b est 1 pour la contrainte de normalisation

    # Résolution du système linéaire A * pi_T = b
    pi_T = numpy.linalg.solve(A, b)
    return pi_T

pi_recurrent_stationnaire = distribution_stationnaire_recurrent(P_recurrent)

print("\nDistribution stationnaire pour la chaîne modifiée (tous états récurrents):")
print(pi_recurrent_stationnaire)
print(dict(zip(etats_recurrent, pi_recurrent_stationnaire)))


# 4. Simulation de la chaîne de Markov modifiée pour visualiser la convergence vers la distribution stationnaire

def simulation_chaine_markov(P, n_steps, etats_noms, etat_initial_index=0):
    """
    Simule une chaîne de Markov pour n_steps étapes et retourne la séquence des états visités
    et l'évolution de la distribution des états au cours du temps.
    """
    n_etats = P.shape[0]
    etat_actuel = etat_initial_index # Commence à l'état initial (par défaut Niveau A = index 0)
    sequence_etats = [etats_noms[etat_actuel]]
    distributions_etats = [np.zeros(n_etats)]
    distributions_etats[0][etat_initial_index] = 1 # Distribution initiale concentrée sur l'état initial

    for _ in range(n_steps):
        # Tirage aléatoire selon les probabilités de transition de l'état actuel
        etat_suivant = np.random.choice(n_etats, p=P[etat_actuel, :])
        sequence_etats.append(etats_noms[etat_suivant])
        etat_actuel = etat_suivant

        # Calcul de la distribution empirique des états à chaque étape
        distribution_courante = np.zeros(n_etats)
        for etat_index in range(n_etats):
            distribution_courante[etat_index] = sequence_etats.count(etats_noms[etat_index]) / len(sequence_etats)
        distributions_etats.append(distribution_courante)

    return sequence_etats, distributions_etats


# Paramètres de la simulation
n_steps_simulation = 500
etat_initial = 0 # Commence au niveau A

# Simulation de la chaîne récurrente
sequence_recurrent, distributions_recurrent = simulation_chaine_markov(P_recurrent, n_steps_simulation, etats_recurrent, etat_initial)

print("\n--- Simulation de la chaîne récurrente ---")
print(f"Séquence des 20 premiers états simulés: {sequence_recurrent[:20]}...")
print(f"Distribution des états après {n_steps_simulation} étapes (empirique):")
print(dict(zip(etats_recurrent, distributions_recurrent[-1])))
print(f"Distribution stationnaire théorique: ")
print(dict(zip(etats_recurrent, pi_recurrent_stationnaire)))


# Resultas de la simulation de la chaîne récurrente
"""
Matrice de transition avec état absorbant 'Extérieur':
[[0.9 0.1 0.  0.  0.  0. ]
 [0.  0.6 0.1 0.  0.  0.3]
 [0.  0.  0.5 0.2 0.  0.3]
 [0.  0.  0.  0.5 0.2 0.3]
 [0.  0.  0.  0.  0.6 0.4]
 [0.  0.  0.  0.  0.  1. ]]

États: ['Niveau A', 'Niveau B', 'Niveau C', 'Niveau D', 'Niveau E', 'Extérieur']

Distribution stationnaire pour la chaîne avec état absorbant:
[0. 0. 0. 0. 0. 1.]
{'Niveau A': 0.0, 'Niveau B': 0.0, 'Niveau C': 0.0, 'Niveau D': 0.0, 'Niveau E': 0.0, 'Extérieur': 1.0}

Matrice de transition modifiée (tous états récurrents):
[[0.9 0.1 0.  0.  0. ]
 [0.3 0.6 0.1 0.  0. ]
 [0.3 0.  0.5 0.2 0. ]
 [0.3 0.  0.  0.5 0.2]
 [0.4 0.  0.  0.  0.6]]

États: ['Niveau A', 'Niveau B', 'Niveau C', 'Niveau D', 'Niveau E']

Distribution stationnaire pour la chaîne modifiée (tous états récurrents):
[0.7518797  0.18796992 0.03759398 0.01503759 0.0075188 ]
{'Niveau A': 0.7518796992481203, 'Niveau B': 0.1879699248120301, 'Niveau C': 0.03759398496240602, 'Niveau D': 0.015037593984962409, 'Niveau E': 0.0075187969924811505}

--- Simulation de la chaîne récurrente ---
Séquence des 20 premiers états simulés: ['Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau B', 'Niveau B', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A', 'Niveau A']...
Distribution des états après 500 étapes (empirique):
{'Niveau A': 0.7285429141716567, 'Niveau B': 0.23353293413173654, 'Niveau C': 0.031936127744510975, 'Niveau D': 0.003992015968063872, 'Niveau E': 0.001996007984031936}
Distribution stationnaire théorique: 
{'Niveau A': 0.7518796992481203, 'Niveau B': 0.1879699248120301, 'Niveau C': 0.03759398496240602, 'Niveau D': 0.015037593984962409, 'Niveau E': 0.0075187969924811505}


** Process exited - Return Code: 0 **
Press Enter to exit terminal
"""
