#stationnaire_power_method.py
import numpy as np

def calculer_distribution_stationnaire_approximation(matrice_transition, tolerance=1e-7, max_iterations=10000):
    """
    Calcule une approximation de la distribution stationnaire d'une chaîne de Markov
    en utilisant la méthode de la puissance itérée (calcul matriciel).

    Args:
        matrice_transition (np.array): La matrice de transition de la chaîne de Markov (carrée).
        tolerance (float): Seuil de tolérance pour la convergence (différence entre distributions successives).
        max_iterations (int): Nombre maximal d'itérations pour la méthode de la puissance.

    Returns:
        np.array: Une approximation de la distribution stationnaire, ou None si la convergence n'est pas atteinte.
    """
    n_etats = matrice_transition.shape[0]
    # Initialisation : distribution initiale uniforme
    distribution_courante = np.ones(n_etats) / n_etats
    iteration = 0

    for _ in range(max_iterations):
        distribution_suivante = distribution_courante @ matrice_transition  # Multiplication matricielle: pi_(n+1) = pi_n * P
        difference = np.linalg.norm(distribution_suivante - distribution_courante, 1) # Norme L1 de la différence pour convergence

        if difference < tolerance:
            return distribution_suivante
        distribution_courante = distribution_suivante
        iteration += 1

    print(f"La méthode de la puissance n'a pas convergé après {max_iterations} iterations.")
    return None

# Définition de la matrice de transition MODIFIÉE (tous états récurrents) - celle pour laquelle on veut la distribution stationnaire
matrice_P_recurrent = np.array([
    [0.9, 0.1, 0.0, 0.0, 0.0],
    [0.3, 0.6, 0.1, 0.0, 0.0],
    [0.3, 0.0, 0.5, 0.2, 0.0],
    [0.3, 0.0, 0.0, 0.5, 0.2],
    [0.4, 0.0, 0.0, 0.0, 0.6]
])

etats_recurrent = ['Niveau A', 'Niveau B', 'Niveau C', 'Niveau D', 'Niveau E']

# Calcul de la distribution stationnaire approximée
distribution_stationnaire_approx = calculer_distribution_stationnaire_approximation(matrice_P_recurrent)

if distribution_stationnaire_approx is not None:
    print("Distribution stationnaire approximée par calcul matriciel (méthode de la puissance):")
    print(distribution_stationnaire_approx)
    distribution_stationnaire_dict = dict(zip(etats_recurrent, distribution_stationnaire_approx))
    print("\nDistribution stationnaire approximée sous forme de dictionnaire:")
    print(distribution_stationnaire_dict)

    # Comparaison avec la distribution stationnaire calculée analytiquement (si vous l'avez déjà calculée)
    distribution_stationnaire_analytique = np.array([100/133, 25/133, 5/133, 2/133, 1/133])
    print("\nDistribution stationnaire analytique (pour comparaison):")
    print(distribution_stationnaire_analytique)
    distribution_stationnaire_analytique_dict = dict(zip(etats_recurrent, distribution_stationnaire_analytique))
    print("\nDistribution stationnaire analytique sous forme de dictionnaire:")
    print(distribution_stationnaire_analytique_dict)
else:
    print("La distribution stationnaire n'a pas pu être approximée par la méthode de la puissance.")

# Exemple de sortie attendue (les valeurs peuvent varier légèrement en fonction de la tolérance et du nombre d'itérations):
"""
Distribution stationnaire approximée par calcul matriciel (méthode de la puissance):
[0.75187965 0.18796992 0.037594   0.01503763 0.0075188 ]

Distribution stationnaire approximée sous forme de dictionnaire:
{'Niveau A': 0.7518796491981825, 'Niveau B': 0.18796991854329975, 'Niveau C': 0.03759400246042534, 'Niveau D': 0.01503762868510195, 'Niveau E': 0.007518801112991094}

Distribution stationnaire analytique (pour comparaison):
[0.7518797  0.18796992 0.03759398 0.01503759 0.0075188 ]

Distribution stationnaire analytique sous forme de dictionnaire:
{'Niveau A': 0.7518796992481203, 'Niveau B': 0.18796992481203006, 'Niveau C': 0.03759398496240601, 'Niveau D': 0.015037593984962405, 'Niveau E': 0.007518796992481203}


** Process exited - Return Code: 0 **
Press Enter to exit terminal
"""
