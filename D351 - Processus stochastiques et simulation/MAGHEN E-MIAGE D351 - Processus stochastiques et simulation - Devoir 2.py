import random
import copy # Pour s'assurer de copier les listes d'indices

# 0. Tableau des livres avec les données fournies
# Nous stockons les données dans une liste de dictionnaires pour un accès facile.
books_data = [
    {"Titre": "L'heure des prédateurs", "Prix": 19.00, "Nombre de pages": 160},
    {"Titre": "Intérieur nuit", "Prix": 18.00, "Nombre de pages": 112},
    {"Titre": "La Femme De Ménage - : Les secrets de la femme de ménage", "Prix": 8.60, "Nombre de pages": 416},
    {"Titre": "The Hunger Games - : Hunger Games - Lever de soleil sur la moisson", "Prix": 19.90, "Nombre de pages": 480},
    {"Titre": "On m'appelle Casquette verte", "Prix": 20.00, "Nombre de pages": 240},
    {"Titre": "La Très Catastrophique Visite du Zoo", "Prix": 19.00, "Nombre de pages": 256},
    {"Titre": "La psy", "Prix": 8.60, "Nombre de pages": 416},
    {"Titre": "Louison et Monsieur Molière", "Prix": 4.90, "Nombre de pages": 128},
    {"Titre": "La prof", "Prix": 22.00, "Nombre de pages": 384},
    {"Titre": "Ma mère, Dieu et Sylvie Vartan", "Prix": 8.40, "Nombre de pages": 240},
    {"Titre": "On ne badine pas avec l'amour", "Prix": 2.95, "Nombre de pages": 192},
    {"Titre": "2666", "Prix": 13.55, "Nombre de pages": 1176},
    {"Titre": "Des mots sur nos maux", "Prix": 15.00, "Nombre de pages": 257},
    {"Titre": "Personne ne doit savoir", "Prix": 8.95, "Nombre de pages": 384},
    {"Titre": "Les ingénieurs du chaos", "Prix": 8.50, "Nombre de pages": 240},
    {"Titre": "Imbalance", "Prix": 19.90, "Nombre de pages": 600},
    {"Titre": "Résister", "Prix": 5.00, "Nombre de pages": 144},
    {"Titre": "Les #Gueux - On fait quoi ?", "Prix": 4.90, "Nombre de pages": 28},
    {"Titre": "Ty carnage", "Prix": 19.95, "Nombre de pages": 208}
]

# Paramètres de l'exercice
BUDGET = 70.0
NUM_BOOKS = len(books_data)
NUM_ITERATIONS = random.randint(100000, 1000000) # Génère un grand nombre aléatoire pour les étapes de la simulation

# --- Fonctions d'aide ---

def calculate_price(selection_indices, book_list):
    """Calcule le prix total pour une sélection de livres donnée par leurs indices."""
    total_price = 0
    for idx in selection_indices:
        total_price += book_list[idx]["Prix"]
    return round(total_price, 2) # Arrondi pour éviter les problèmes de virgule flottante

def calculate_pages(selection_indices, book_list):
    """Calcule le nombre total de pages pour une sélection de livres donnée par leurs indices."""
    total_pages = 0
    for idx in selection_indices:
        total_pages += book_list[idx]["Nombre de pages"]
    return total_pages

def is_valid_selection(selection_indices, book_list, budget):
    """Vérifie si une sélection de livres respecte le budget."""
    return calculate_price(selection_indices, book_list) <= budget

# --- Fonction de transition de la chaîne de Markov ---

def markov_transition(current_selection_indices, book_list, budget):
    """
    Applique une étape de transition symétrique :
    Propose d'ajouter ou retirer un livre aléatoire.
    Accepte la proposition si elle reste dans le budget, sinon reste dans l'état actuel.
    """
    num_books = len(book_list)
    # Choisissez aléatoirement un indice de livre parmi tous les livres possibles
    random_book_idx = random.randrange(num_books)

    # Créez une copie de la sélection actuelle pour proposer une modification
    proposed_selection_indices = list(current_selection_indices) # Utilisation de list() pour copier

    # Proposer l'ajout ou le retrait du livre choisi
    if random_book_idx in proposed_selection_indices:
        # Le livre est déjà dans la sélection actuelle, proposer de le retirer
        proposed_selection_indices.remove(random_book_idx)
    else:
        # Le livre n'est pas dans la sélection actuelle, proposer de l'ajouter
        proposed_selection_indices.append(random_book_idx)

    # Vérifier si la sélection proposée est valide (respecte le budget)
    if is_valid_selection(proposed_selection_indices, book_list, budget):
        # La proposition est valide, la transition est acceptée
        return proposed_selection_indices
    else:
        # La proposition est invalide, la chaîne reste dans l'état actuel
        return current_selection_indices # Retourne l'état actuel inchangé

# --- Simulation MCMC pour trouver la meilleure sélection ---

def find_best_selection_mcmc(book_list, budget, num_iterations):
    """
    Simule la chaîne de Markov pour explorer l'espace des sélections valides
    et trouver la sélection valide avec le maximum de pages rencontrée.
    """
    # Initialisation : Commencer avec une sélection valide simple (l'ensemble vide)
    current_selection_indices = []
    best_selection_indices = list(current_selection_indices) # Copie initiale
    max_pages = calculate_pages(best_selection_indices, book_list)

    print(f"Démarrage de la simulation MCMC avec un budget de {budget} € pour {num_iterations} itérations.")
    print(f"État initial : Sélection vide ({max_pages} pages, {calculate_price(best_selection_indices, book_list):.2f} €)")

    # Boucle de simulation
    for i in range(num_iterations):
        # Appliquer une étape de la chaîne de Markov
        next_selection_indices = markov_transition(current_selection_indices, book_list, budget)

        # Calculer les pages pour le nouvel état
        next_pages = calculate_pages(next_selection_indices, book_list)

        # Vérifier si ce nouvel état est le meilleur trouvé jusqu'à présent
        if next_pages > max_pages:
            max_pages = next_pages
            best_selection_indices = list(next_selection_indices) # STOCKER UNE COPIE du meilleur état!
            best_price = calculate_price(best_selection_indices, book_list)
            # Afficher une mise à jour quand un meilleur état est trouvé
            print(f"--- Nouvelle meilleure sélection trouvée à l'itération {i+1} ---")
            print(f"Pages : {max_pages}, Prix : {best_price:.2f} €, Nombre de livres : {len(best_selection_indices)}")


        # Mettre à jour l'état actuel pour la prochaine itération
        current_selection_indices = next_selection_indices

        # Afficher la progression de temps en temps (optionnel)
        # if (i + 1) % (num_iterations // 10) == 0:
        #     print(f"Progression : {i + 1}/{num_iterations} itérations...")


    print("\nSimulation terminée.")
    return best_selection_indices, max_pages

# --- Exécuter la simulation ---

best_indices, best_total_pages = find_best_selection_mcmc(books_data, BUDGET, NUM_ITERATIONS)

# --- Afficher les résultats ---

best_selection_details = [books_data[i] for i in best_indices]
best_total_price = calculate_price(best_indices, books_data)

print("\n--- Meilleure sélection trouvée ---")
print(f"Budget : {BUDGET} €")
print(f"Nombre total de pages : {best_total_pages}")
print(f"Prix total : {best_total_price:.2f} €")
print(f"Nombre de livres : {len(best_indices)}")
print(f"Nombre d'itérations : {NUM_ITERATIONS}")
print("\nLivres sélectionnés :")
for book in best_selection_details:
    print(f"- {book['Titre']} ({book['Nombre de pages']} pages, {book['Prix']:.2f} €)")

# Optional: Verify the best found selection is actually valid within the budget
print(f"\nVérification du budget pour la meilleure sélection : {is_valid_selection(best_indices, books_data, BUDGET)}")

# --- Exemple de résultats ---
Démarrage de la simulation MCMC avec un budget de 70.0 € pour 310413 itérations.
État initial : Sélection vide (0 pages, 0.00 €)
--- Nouvelle meilleure sélection trouvée à l'itération 1 ---
Pages : 256, Prix : 19.00 €, Nombre de livres : 1
--- Nouvelle meilleure sélection trouvée à l'itération 2 ---
Pages : 448, Prix : 21.95 €, Nombre de livres : 2
--- Nouvelle meilleure sélection trouvée à l'itération 3 ---
Pages : 928, Prix : 41.85 €, Nombre de livres : 3
--- Nouvelle meilleure sélection trouvée à l'itération 4 ---
Pages : 1312, Prix : 50.80 €, Nombre de livres : 4
--- Nouvelle meilleure sélection trouvée à l'itération 7 ---
Pages : 1712, Prix : 60.40 €, Nombre de livres : 5
--- Nouvelle meilleure sélection trouvée à l'itération 10 ---
Pages : 1952, Prix : 68.80 €, Nombre de livres : 6
--- Nouvelle meilleure sélection trouvée à l'itération 56 ---
Pages : 2128, Prix : 57.40 €, Nombre de livres : 6
--- Nouvelle meilleure sélection trouvée à l'itération 261 ---
Pages : 2249, Prix : 68.35 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 274 ---
Pages : 2504, Prix : 63.35 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 319 ---
Pages : 2664, Prix : 65.45 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 338 ---
Pages : 2772, Prix : 66.55 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 342 ---
Pages : 2872, Prix : 66.55 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 372 ---
Pages : 2924, Prix : 68.30 €, Nombre de livres : 8
--- Nouvelle meilleure sélection trouvée à l'itération 2129 ---
Pages : 3016, Prix : 67.25 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 2152 ---
Pages : 3112, Prix : 69.65 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 5802 ---
Pages : 3136, Prix : 67.35 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 6765 ---
Pages : 3152, Prix : 67.45 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 13822 ---
Pages : 3208, Prix : 67.55 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 19474 ---
Pages : 3220, Prix : 69.35 €, Nombre de livres : 9
--- Nouvelle meilleure sélection trouvée à l'itération 24186 ---
Pages : 3328, Prix : 67.55 €, Nombre de livres : 7
--- Nouvelle meilleure sélection trouvée à l'itération 117119 ---
Pages : 3336, Prix : 69.45 €, Nombre de livres : 9

Simulation terminée.

--- Meilleure sélection trouvée ---
Budget : 70.0 €
Nombre total de pages : 3336
Prix total : 69.45 €
Nombre de livres : 9
Nombre d'itérations : 310413

Livres sélectionnés :
- Louison et Monsieur Molière (128 pages, 4.90 €)
- Résister (144 pages, 5.00 €)
- 2666 (1176 pages, 13.55 €)
- On ne badine pas avec l'amour (192 pages, 2.95 €)
- La psy (416 pages, 8.60 €)
- Personne ne doit savoir (384 pages, 8.95 €)
- La Femme De Ménage - : Les secrets de la femme de ménage (416 pages, 8.60 €)
- Les ingénieurs du chaos (240 pages, 8.50 €)
- Ma mère, Dieu et Sylvie Vartan (240 pages, 8.40 €)

Vérification du budget pour la meilleure sélection : True


** Process exited - Return Code: 0 **
Press Enter to exit terminal
 """