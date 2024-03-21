import cv2
import numpy as np

# Fonction pour dessiner les axes X et Y sur l'image
def dessiner_repere(image):
    # Dessiner l'axe X (horizontal) en rouge
    cv2.line(image, (0, image.shape[0]//2), (image.shape[1], image.shape[0]//2), (0, 0, 255), 2)
    # Dessiner l'axe Y (vertical) en vert
    cv2.line(image, (image.shape[1]//2, 0), (image.shape[1]//2, image.shape[0]), (0, 255, 0), 2)


# Fonction pour détecter la balle de tennis
def detecter_balle(image):
    # Convertir l'image en espace de couleur HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Définir la plage de couleur de la balle de tennis en HSV
    lower_color = np.array([30, 50, 50])
    upper_color = np.array([45, 255, 255])

    # Créer un masque pour filtrer la couleur de la balle de tennis
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Appliquer une série de transformations morphologiques pour nettoyer le masque
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Trouver les contours de la balle de tennis dans le masque
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Vérifier si des contours ont été trouvés
    if len(contours) > 0:
        # Trouver le contour de la plus grande forme (qui devrait être la balle de tennis)
        c = max(contours, key=cv2.contourArea)
        # Obtenir le cercle englobant et le centre de la balle de tennis
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        return (int(x), int(y)), int(radius)
    else:
        return None

# Fonction pour afficher les coordonnées de la balle par rapport à la surface détectée par la caméra
def afficher_coordonnees_balle(surface_detectee, centre_balle):
    # Calculer les coordonnées de la balle par rapport à la surface détectée
    coord_x = centre_balle[0] - surface_detectee[0]
    coord_y = centre_balle[1] - surface_detectee[1]
    print("Coordonnées de la balle par rapport à la surface détectée:")
    print("X:", coord_x)
    print("Y:", coord_y)

# Fonction pour afficher les coordonnées de la balle par rapport au repère
def afficher_position_balle_par_rapport_a_axy(image, centre_balle):
    # Coordonnées du centre de l'image
    centre_image_x = image.shape[1] // 2
    centre_image_y = image.shape[0] // 2

    # Coordonnées du centre du repère
    centre_repere_x = centre_image_x
    centre_repere_y = centre_image_y

    # Calculer les coordonnées de la balle par rapport au repère
    diff_x = centre_balle[0] - centre_repere_x
    diff_y = centre_repere_y - centre_balle[1]  # Inverser pour que les coordonnées positives soient vers le haut

    print("Position de la balle par rapport au repère:")
    print("X:", diff_x)
    print("Y:", diff_y)

    # Afficher le message en fonction de la position de la balle par rapport au centre de l'écran
    if diff_x < -50  :
        print("Balle à gauche du centre")
    elif diff_x > 50:
        print("Balle à droite du centre")
    else:
        print("********************Balle centrée***************************")

# Lecture de l'image à partir de la webcam (remplacez 0 par le numéro de votre webcam si nécessaire)
cap = cv2.VideoCapture("http://192.168.88.213:8080/video")   

while True:
    # Capture d'une image de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Dessiner les axes X et Y sur l'image
    dessiner_repere(frame)
        
    # Détecter la balle de tennis dans l'image
    resultat_detection = detecter_balle(frame)

    # Vérifier si la balle de tennis a été détectée
    if resultat_detection is not None:
        centre, rayon = resultat_detection
        # Dessiner le cercle englobant autour de la balle de tennis
        cv2.circle(frame, centre, rayon, (0, 255, 0), 2)
        # Afficher les coordonnées de la balle par rapport à la surface détectée
        # afficher_coordonnees_balle((frame.shape[1]//2, frame.shape[0]//2), centre)
         # Afficher la position de la balle par rapport au repère
        afficher_position_balle_par_rapport_a_axy(frame, centre)

    # Afficher l'image avec les résultats
    cv2.imshow('Frame', frame)

    # Attendre l'appui sur la touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et détruire toutes les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()
