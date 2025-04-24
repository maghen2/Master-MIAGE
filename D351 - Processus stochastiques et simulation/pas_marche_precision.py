#pas_marche_precision.py
import numpy as np

def random_walk_stationary_distribution(matrice_transition, etats_noms, n_steps, etat_initial_index=0):
    """
    Calcule une approximation de la distribution stationnaire d'une chaîne de Markov
    en utilisant une marche aléatoire sur le graphe de la chaîne.

    Args:
        matrice_transition (np.array): La matrice de transition de la chaîne de Markov (carrée).
        etats_noms (list): Liste des noms des états.
        n_steps (int): Nombre d'étapes pour la marche aléatoire.
        etat_initial_index (int): Index de l'état initial.

    Returns:
        dict: Distribution stationnaire approximée.
    """
    n_etats = matrice_transition.shape[0]
    etat_actuel_index = etat_initial_index
    sequence_etats_indices = [etat_actuel_index]

    for _ in range(n_steps):
        probabilites_transitions = matrice_transition[etat_actuel_index, :]
        etat_suivant_index = np.random.choice(n_etats, p=probabilites_transitions)
        etat_actuel_index = etat_suivant_index
        sequence_etats_indices.append(etat_actuel_index)

    distribution_approx = np.zeros(n_etats)
    for etat_index in sequence_etats_indices:
        distribution_approx[etat_index] += 1

    distribution_approx /= len(sequence_etats_indices)
    return dict(zip(etats_noms, distribution_approx))

def calculer_nombre_pas_pour_precision(matrice_transition, etats_noms, distribution_analytique, precision_decimales=2):
    """
    Détermine le nombre minimal de pas de la marche aléatoire pour atteindre
    une approximation de la distribution stationnaire avec une précision donnée (nombre de décimales exactes).

    Args:
        matrice_transition (np.array): La matrice de transition de la chaîne de Markov.
        etats_noms (list): Liste des noms des états.
        distribution_analytique (dict): Distribution stationnaire analytique de référence.
        precision_decimales (int): Nombre de décimales exactes souhaité.

    Returns:
        int: Nombre de pas minimal requis, ou None si non trouvé après un nombre maximal d'essais.
    """
    nombre_pas = 1000  # Nombre de pas de départ
    pas_increment = 1000  # Incrément pour augmenter le nombre de pas à chaque itération
    tolerance = 0.5 * (10**(-precision_decimales)) # Tolérance pour garantir la précision des décimales

    while True:
        distribution_approx_random_walk = random_walk_stationary_distribution(
            matrice_transition, etats_noms, nombre_pas
        )

        max_diff = 0
        for etat in etats_noms:
            diff = abs(distribution_approx_random_walk[etat] - distribution_analytique[etat])
            max_diff = max(max_diff, diff)

        if max_diff < tolerance:
            return nombre_pas

        nombre_pas += pas_increment
        if nombre_pas > 1000000: # Limite pour éviter une boucle infinie
            return None # Nombre de pas trop important, précision non atteinte dans les limites fixées


# Définition de la matrice de transition MODIFIÉE (tous états récurrents)
matrice_P_recurrent = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0],
    [0.3, 0.6, 0.1, 0.0, 0.0],
    [0.3, 0.0, 0.5, 0.2, 0.0],
    [0.3, 0.0, 0.0, 0.5, 0.2],
    [0.4, 0.0, 0.0, 0.0, 0.6]
])

etats_recurrent = ['Niveau A', 'Niveau B', 'Niveau C', 'Niveau D', 'Niveau E']

# Distribution stationnaire analytique (calculée précédemment)
distribution_stationnaire_analytique_dict = {
    'Niveau A': 100/133,
    'Niveau B': 25/133,
    'Niveau C': 5/133,
    'Niveau D': 2/133,
    'Niveau E': 1/133
}

# Calcul du nombre de pas nécessaires pour 2 décimales exactes
nombre_pas_requis = calculer_nombre_pas_pour_precision(
    matrice_P_recurrent, etats_recurrent, distribution_stationnaire_analytique_dict, precision_decimales=2
)

if nombre_pas_requis:
    print(f"Nombre de pas de la marche aléatoire nécessaire pour avoir une approximation à 2 décimales exactes : {nombre_pas_requis}")
else:
    print("Nombre de pas requis non trouvé après un million d'itérations. La précision à deux décimales pourrait ne pas être atteignable avec cette méthode dans cette limite.")


# Vérification (optionnelle) de l'approximation avec le nombre de pas trouvé
if nombre_pas_requis:
    distribution_approx_finale = random_walk_stationary_distribution(
        matrice_P_recurrent, etats_recurrent, nombre_pas_requis
    )
    print("\nDistribution stationnaire approximée avec ce nombre de pas :")
    print(distribution_approx_finale)
    print("\nDistribution stationnaire analytique (pour comparaison):")
    print(distribution_stationnaire_analytique_dict)

# Sortie attendue (approximative) :
"""
Nombre de pas de la marche aléatoire nécessaire pour avoir une approximation à 2 décimales exactes : 7000

Distribution stationnaire approximée avec ce nombre de pas :
{'Niveau A': 0.7653192401085559, 'Niveau B': 0.18011712612483932, 'Niveau C': 0.03370947007570347, 'Niveau D': 0.013998000285673474, 'Niveau E': 0.006856163405227825}

Distribution stationnaire analytique (pour comparaison):
{'Niveau A': 0.7518796992481203, 'Niveau B': 0.18796992481203006, 'Niveau C': 0.03759398496240601, 'Niveau D': 0.015037593984962405, 'Niveau E': 0.007518796992481203}


** Process exited - Return Code: 0 **
Press Enter to exit terminal
"""
