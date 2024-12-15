from datetime import datetime

import streamlit as st
import matplotlib.pyplot as plt
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime, timezone



# Charger les fichiers CSV
try:
    items_df = pd.read_csv("csv/items.csv")
    recommendations_df = pd.read_csv("csv/submission-56.csv")
    interaction_df = pd.read_csv("csv/interactions_train.csv")
except FileNotFoundError as e:
    st.error(f"Error loading data: {e}")

interaction_df.rename(columns={'u': 'user_id', 'i':'item_id'}, inplace=True)  # Exemple de renommage

interaction_df['t'] = interaction_df['t'].apply(lambda x: datetime.fromtimestamp(x, timezone.utc).strftime('%Y-%m-%d'))
# Fonction pour récupérer l'URL de l'image du livre
def fetch_book_image(isbn, title, api_key):
    """
    Récupère l'image d'un livre en priorité via ISBN.
    Si aucune image n'est trouvée, recherche via le titre.
    Si toujours rien, retourne une image par défaut.
    """
    # Recherche par ISBN
    if isbn:
        url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}&key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                volume_info = data["items"][0]["volumeInfo"]
                image_url = volume_info.get("imageLinks", {}).get("thumbnail", None)
                if image_url:
                    return image_url

    # Recherche par titre
    if title:
        url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title}&key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                volume_info = data["items"][0]["volumeInfo"]
                image_url = volume_info.get("imageLinks", {}).get("thumbnail", None)
                if image_url:
                    return image_url

    # Image par défaut
    return "https://via.placeholder.com/150?text=No+Image+For+This+Book"



def show_books_with_images(book_details, api_key):
    """
    Affiche un tableau où chaque ligne contient l'image d'un livre,
    son titre et ses détails.
    """
    st.write("### 📚 Recommended Books")

    # Créer une liste pour stocker les données à afficher dans le tableau
    table_data = []

    for _, row in book_details.iterrows():
        isbn = row["ISBN Valid"]
        title = row["Title"]
        author = row["Author"]
        publisher = row["Publisher"]
        subjects = row["Subjects"]

        # Obtenir l'image via l'API ou afficher une image par défaut
        image_url = fetch_book_image(isbn.strip() if pd.notna(isbn) else None, title, api_key)
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_html = f'<img src="{image_url}" width="60">'
        except Exception:
            image_html = '<img src="https://via.placeholder.com/150?text=No+Image+For+This+Book" width="60">'

        # Ajouter les données dans la liste sous forme HTML
        row_data = {
            "Image": image_html,
            "Title": title,
            "Author": author if pd.notna(author) else "Unknown",
            "Publisher": publisher if pd.notna(publisher) else "Unknown",
            "Subjects": subjects if pd.notna(subjects) else "Unknown"
        }
        table_data.append(row_data)

    # Convertir la liste en DataFrame
    df = pd.DataFrame(table_data)

    # Afficher les données en utilisant st.markdown pour inclure des images
    st.markdown(
        df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

def get_predictions(user_id, api_key):
    try:
        user_id = int(user_id)  # Convertir l'entrée en entier
    except ValueError:
        st.error("Invalid User ID. Please enter a numeric User ID.")
        return

    # Vérifier si l'User ID existe dans les recommandations
    if user_id not in recommendations_df['user_id'].values:
        st.error(f"User ID {user_id} is not in the dataset. Please try another ID.")
        return

    # Récupérer les recommandations
    user_recommendations = recommendations_df[recommendations_df['user_id'] == user_id]
    if user_recommendations.empty:
        st.error("No recommendation available for this user.")
        return

    # Formater les recommandations
    recommended_items = user_recommendations['recommendation'].iloc[0].split()
    recommended_items = list(map(int, recommended_items))

    # Récupérer les informations sur les livres
    book_details = items_df[items_df['i'].isin(recommended_items)][['Title', 'Author', 'Publisher', 'Subjects', 'ISBN Valid']]
    if book_details.empty:
        st.error("No books found for the recommendations.")
        return

    # Nettoyer les titres
    book_details['Title'] = book_details['Title'].str.rstrip('/')

    # Afficher les images et le tableau côte à côte
    show_books_with_images(book_details, api_key)


def get_interactions_with_message(user_id):
    try:
        user_id = int(user_id)  # Convertir l'entrée en entier
    except ValueError:
        return "Invalid User ID. Please enter a numeric User ID.", None

    # Vérifier si l'User ID existe dans les interactions
    # Vérifier si l'User ID existe dans les interactions
    if user_id not in interaction_df['user_id'].values:
        return f"User ID {user_id} is not in the dataset. Please try another ID.", pd.DataFrame()

    user_interactions = interaction_df[interaction_df['user_id'] == user_id]
    if user_interactions.empty:
        return "No interactions found for this user.", None



    # Compter le nombre de livres lus en 2024
    books_read_2024 = user_interactions[user_interactions['t'].str.startswith('2024')].shape[0]
    # Générer un message en fonction du nombre de livres lus
    if books_read_2024 > 5:
        message = f"🎉 Congratulations! In 2024, you made {books_read_2024} interactions with books."
    else:
        message = f"📚 Keep it up! You've interacted with {books_read_2024} books in 2024. Try to read more to reach your goals!"

    # Récupérer les informations sur les livres interactés
    interacted_items = items_df[items_df['i'].isin(user_interactions['item_id'])]
    if interacted_items.empty:
        return message, "No details found for the interacted books."

    # Joindre les dates aux détails des livres
    interacted_items = interacted_items.merge(user_interactions[['item_id', 't']], left_on='i', right_on='item_id')

    # Réorganiser les colonnes pour mettre la date en premier
    interacted_items = interacted_items[['t', 'Title', 'Author', 'Publisher', 'Subjects']]
    interacted_items['Title'] = interacted_items['Title'].str.rstrip('/')
    interacted_items.rename(columns={'t': 'You read this book the:'}, inplace=True)  # Renommer la colonne

    # Supprimer les doublons
    interacted_items = interacted_items.drop_duplicates()

    # Trier par date (les plus récentes en premier)
    interacted_items = interacted_items.sort_values(by='You read this book the:', ascending=False)

    return message, interacted_items


def show_interaction_history_with_images(interacted_books, api_key):
    """
    Affiche un tableau où chaque ligne contient l'image d'un livre,
    son titre, ses détails et la date d'interaction.
    """
    st.write("### 📖 Interaction History")

    # Créer une liste pour stocker les données à afficher dans le tableau
    table_data = []

    for _, row in interacted_books.iterrows():
        isbn = row.get("ISBN Valid", None)
        title = row.get("Title", "Unknown Title")
        author = row.get("Author", "Unknown Author")
        publisher = row.get("Publisher", "Unknown Publisher")
        subjects = row.get("Subjects", "Unknown Subjects")
        date = row.get("You read this book the:", "Unknown Date")

        # Obtenir l'image via l'API ou afficher une image par défaut
        image_url = fetch_book_image(isbn.strip() if pd.notna(isbn) else None, title, api_key)
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_html = f'<img src="{image_url}" width="60">'
        except Exception:
            image_html = '<img src="https://via.placeholder.com/150?text=No+Image+For+This+Book" width="60">'

        # Ajouter les données dans la liste sous forme HTML
        row_data = {
            "Image": image_html,
            "Date": date,
            "Title": title,
            "Author": author,
            "Publisher": publisher,
            "Subjects": subjects,
        }
        table_data.append(row_data)

    # Convertir la liste en DataFrame
    df = pd.DataFrame(table_data)

    # Afficher les données en utilisant st.markdown pour inclure des images
    st.markdown(
        df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
# Interface Streamlit
st.title("📚 Recommendation System with Interaction History")
st.write("Enter a User ID to see their recommendations or interaction history.")

# Saisie de l'utilisateur
user_id = st.text_input("Enter User ID", placeholder="Type a User ID here")
api_key = "AIzaSyAB6qGLJAGdm3-PtTFskZmbDubd0ZOn8ZA"  # Remplacez par votre clé API Google Books

# Boutons pour afficher les recommandations ou l'historique
if st.button("🔍 View Recommendations"):
    if user_id.isnumeric():
        get_predictions(user_id, api_key)
    else:
        st.error("Please enter a valid numeric User ID.")

if st.button("📖 View Interaction History"):
    if user_id.isnumeric():
        # Appeler la fonction et récupérer le message et les interactions
        message, interacted_items = get_interactions_with_message(user_id)

        # Vérifier si des interactions existent
        if isinstance(interacted_items, pd.DataFrame) and not interacted_items.empty:
            # Cas de succès : afficher le message et les interactions
            st.success(message)
            show_interaction_history_with_images(interacted_items, api_key)
        else:
            # Cas d'échec : afficher un message d'erreur ou d'avertissement
            st.error(message)
    else:
        # Si l'ID utilisateur n'est pas valide
        st.error("Please enter a valid numeric User ID.")
