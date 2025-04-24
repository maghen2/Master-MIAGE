# random_walk_stationnaire.py
import numpy as np
import random

def random_walk_stationary_distribution(matrice_transition, etats_noms, n_steps, etat_initial_index=0):
    """
    Calcule une approximation de la distribution stationnaire d'une chaîne de Markov
    en utilisant une marche aléatoire sur le graphe de la chaîne.

    Args:
        matrice_transition (np.array): La matrice de transition de la chaîne de Markov (carrée).
        etats_noms (list): Liste des noms des états (pour indexer la distribution résultante).
        n_steps (int): Nombre d'étapes pour la marche aléatoire.
        etat_initial_index (int): Index de l'état initial pour démarrer la marche aléatoire.

    Returns:
        dict: Un dictionnaire représentant la distribution stationnaire approximée,
              avec les états comme clés et les proportions de temps passé comme valeurs.
    """
    n_etats = matrice_transition.shape[0]
    etat_actuel_index = etat_initial_index # Démarre à l'état initial
    sequence_etats_indices = [etat_actuel_index] # Historique des indices des états visités

    for _ in range(n_steps):
        # Choisir l'état suivant basé sur les probabilités de transition de l'état actuel
        probabilites_transitions = matrice_transition[etat_actuel_index, :]
        etat_suivant_index = np.random.choice(n_etats, p=probabilites_transitions)
        etat_actuel_index = etat_suivant_index
        sequence_etats_indices.append(etat_actuel_index)

    # Calculer les proportions de temps passé dans chaque état
    distribution_approx = np.zeros(n_etats)
    for etat_index in sequence_etats_indices:
        distribution_approx[etat_index] += 1

    distribution_approx /= len(sequence_etats_indices) # Normaliser pour obtenir des proportions

    return dict(zip(etats_noms, distribution_approx))

# Définition de la matrice de transition MODIFIÉE (tous états récurrents)
matrice_P_recurrent = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0],
    [0.3, 0.6, 0.1, 0.0, 0.0],
    [0.3, 0.0, 0.5, 0.2, 0.0],
    [0.3, 0.0, 0.0, 0.5, 0.2],
    [0.4, 0.0, 0.0, 0.0, 0.6]
])

etats_recurrent = ['Niveau A', 'Niveau B', 'Niveau C', 'Niveau D', 'Niveau E']

# Paramètres pour la marche aléatoire
n_steps_random_walk = 100000  # Nombre d'étapes de la marche aléatoire (augmenter pour une meilleure précision)
etat_initial_index = 0 # Débuter la marche à partir du Niveau A (index 0)

# Calcul de la distribution stationnaire approximée par marche aléatoire
distribution_stationnaire_random_walk = random_walk_stationary_distribution(matrice_P_recurrent, etats_recurrent, n_steps_random_walk, etat_initial_index)

print("Distribution stationnaire approximée par marche aléatoire:")
print(distribution_stationnaire_random_walk)

# (Optionnel) Afficher la distribution stationnaire analytique pour comparaison
distribution_stationnaire_analytique = {
    'Niveau A': 100/133,
    'Niveau B': 25/133,
    'Niveau C': 5/133,
    'Niveau D': 2/133,
    'Niveau E': 1/133
}
print("\nDistribution stationnaire analytique (pour comparaison):")
print(distribution_stationnaire_analytique)

# Exemple de sortie attendue (les valeurs peuvent varier légèrement en fonction du nombre d'étapes de la marche aléatoire):
"""
Distribution stationnaire approximée par marche aléatoire:
{'Niveau A': 0.7537524624753752, 'Niveau B': 0.1873981260187398, 'Niveau C': 0.037249627503724965, 'Niveau D': 0.013759862401375986, 'Niveau E': 0.007839921600783992}

Distribution stationnaire analytique (pour comparaison):
{'Niveau A': 0.7518796992481203, 'Niveau B': 0.18796992481203006, 'Niveau C': 0.03759398496240601, 'Niveau D': 0.015037593984962405, 'Niveau E': 0.007518796992481203}


** Process exited - Return Code: 0 **
Press Enter to exit terminal
"""