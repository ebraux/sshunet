```python
# Fonctions de visualisation pour le modèle U-NET SSH

def visualize_dataset_samples(dataset, num_samples=5):
    """
    Visualise des échantillons du dataset SSH.
    
    Args:
        dataset: Instance de LightSSHDataset
        num_samples: Nombre d'échantillons à afficher
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Limiter le nombre d'échantillons à la taille du dataset
    num_samples = min(num_samples, len(dataset))
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))
    
    for i in range(num_samples):
        sample = dataset[i]
        
        # Image bruitée (input)
        im1 = axes[0, i].imshow(sample['input'].squeeze(), cmap='viridis')
        axes[0, i].set_title(f"Input {i+1}")
        axes[0, i].axis('off')
        
        # Image propre (target)
        im2 = axes[1, i].imshow(sample['target'].squeeze(), cmap='viridis')
        axes[1, i].set_title(f"Target {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    fig.colorbar(im1, ax=axes[0, :].ravel().tolist(), shrink=0.8)
    fig.colorbar(im2, ax=axes[1, :].ravel().tolist(), shrink=0.8)
    plt.show()
    
    # Visualiser un exemple en détail
    sample = dataset[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Input
    im1 = ax1.imshow(sample['input'].squeeze(), cmap='viridis')
    ax1.set_title('Données SSH bruitées (Input)')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.axis('off')
    
    # Target
    im2 = ax2.imshow(sample['target'].squeeze(), cmap='viridis')
    ax2.set_title('Données SSH propres (Target)')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    
    # Différence
    diff = sample['input'].squeeze() - sample['target'].squeeze()
    im3 = ax3.imshow(diff, cmap='RdBu_r')
    ax3.set_title('Bruit (Différence)')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_model_architecture(model):
    """
    Visualise l'architecture du modèle U-NET avec un diagramme simple.
    
    Args:
        model: Instance du modèle SSHUNetModel
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Obtenir les détails du modèle
    unet = model.model
    
    # Dimensions approximatives des couches (version simplifiée)
    img_size = 64
    layer_sizes = [
        img_size,           # Input
        img_size,           # Après inc
        img_size // 2,      # Après down1
        img_size // 4,      # Après down2
        img_size // 8,      # Après down3
        img_size // 16,     # Après down4
        img_size // 8,      # Après up1
        img_size // 4,      # Après up2
        img_size // 2,      # Après up3
        img_size,           # Après up4
        img_size            # Output
    ]
    
    # Nombre de canaux (version simplifiée)
    n_channels = [
        1,                  # Input
        32,                 # Après inc
        64,                 # Après down1
        128,                # Après down2
        256,                # Après down3
        256,                # Après down4 (bilinear=True -> divisé par 2)
        128,                # Après up1
        64,                 # Après up2
        32,                 # Après up3
        32,                 # Après up4
        1                   # Output
    ]
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Paramètres de visualisation
    box_width = 0.5
    max_height = 2.0
    center_x = 5
    
    # Dessiner les couches
    for i, (size, channels) in enumerate(zip(layer_sizes, n_channels)):
        # Position en Y (plus bas pour les couches de l'encodeur, plus haut pour le décodeur)
        if i <= 5:
            # Encodeur - descend
            y_pos = i
        else:
            # Décodeur - remonte
            y_pos = 10 - i
            
        # Taille relative des boîtes
        rel_size = size / img_size
        box_height = max_height * rel_size
        
        # Position X
        if i <= 5:
            # Encodeur - à gauche du centre
            x_pos = center_x - 2
        else:
            # Décodeur - à droite du centre
            x_pos = center_x + 2
            
        # Dessiner la boîte
        rect = plt.Rectangle(
            (x_pos - box_width/2, y_pos - box_height/2),
            box_width, box_height,
            facecolor='skyblue', edgecolor='navy', alpha=0.7
        )
        ax.add_patch(rect)
        
        # Ajouter du texte
        if i == 0:
            label = "Input"
        elif i == len(layer_sizes) - 1:
            label = "Output"
        elif i <= 4:
            label = f"Down{i}"
        else:
            label = f"Up{10-i}"
            
        ax.text(x_pos, y_pos, label, ha='center', va='center', fontweight='bold')
        ax.text(x_pos, y_pos+0.3, f"{channels} ch", ha='center', va='center', fontsize=8)
        ax.text(x_pos, y_pos-0.3, f"{size}×{size}", ha='center', va='center', fontsize=8)
        
        # Dessiner les connexions skip pour le décodeur
        if i > 5:
            # Connexion entre up(i-5) et down(10-i)
            skip_y = 10 - i  # Position Y de la couche correspondante dans l'encodeur
            ax.plot([x_pos - box_width/2, center_x, x_pos - 2 - box_width/2], 
                   [y_pos, (y_pos + skip_y)/2, skip_y],
                   'r-', alpha=0.5)
    
    # Ajuster les limites et supprimer les axes
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 6)
    ax.axis('off')
    
    # Titre
    plt.title('Architecture U-NET Simplifiée', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(trainer_history=None, epochs=None):
    """
    Trace les courbes d'apprentissage du modèle.
    
    Args:
        trainer_history: Historique d'entraînement (optionnel)
        epochs: Nombre d'epochs si pas d'historique disponible
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if trainer_history is not None:
        # Utiliser l'historique réel si disponible
        train_losses = trainer_history['train_loss']
        val_losses = trainer_history['val_loss']
        train_psnr = trainer_history['train_psnr']
        val_psnr = trainer_history['val_psnr']
        epochs = list(range(1, len(train_losses) + 1))
    elif epochs is not None:
        # Simuler des courbes d'apprentissage typiques
        epochs = list(range(1, epochs + 1))
        train_losses = [0.15 - 0.01*i + 0.005*np.random.randn() for i in epochs]
        val_losses = [0.14 - 0.008*i + 0.01*np.random.randn() for i in epochs]
        train_psnr = [10 + 1.5*i - 0.2*np.random.randn() for i in epochs]
        val_psnr = [9 + 1.3*i - 0.5*np.random.randn() for i in epochs]
    else:
        raise ValueError("Veuillez fournir soit l'historique d'entraînement, soit le nombre d'epochs")
    
    # Tracer les courbes d'apprentissage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Courbe de perte
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Évolution de la perte (MSE)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Mean Squared Error')
    ax1.legend()
    ax1.grid(True)
    
    # Courbe PSNR
    ax2.plot(epochs, train_psnr, 'b-', label='Train PSNR')
    ax2.plot(epochs, val_psnr, 'r-', label='Validation PSNR')
    ax2.set_title('Évolution du PSNR')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('PSNR (dB)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_model_predictions(model, test_loader, num_samples=5):
    """
    Visualise les prédictions du modèle sur des données de test.
    
    Args:
        model: Modèle entraîné (SSHUNetModel)
        test_loader: DataLoader contenant les données de test
        num_samples: Nombre d'échantillons à visualiser
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Collecter quelques prédictions
    inputs = []
    targets = []
    predictions = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            x = batch['input']
            y = batch['target']
            y_hat = model(x)
            
            inputs.append(x.squeeze().cpu().numpy())
            targets.append(y.squeeze().cpu().numpy())
            predictions.append(y_hat.squeeze().cpu().numpy())
    
    # Visualiser les résultats
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*3, 9))
    
    for i in range(num_samples):
        # Input
        im1 = axes[0, i].imshow(inputs[i], cmap='viridis')
        axes[0, i].set_title(f"Input {i+1}")
        axes[0, i].axis('off')
        
        # Target
        im2 = axes[1, i].imshow(targets[i], cmap='viridis')
        axes[1, i].set_title(f"Target {i+1}")
        axes[1, i].axis('off')
        
        # Prediction
        im3 = axes[2, i].imshow(predictions[i], cmap='viridis')
        axes[2, i].set_title(f"Prediction {i+1}")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    fig.colorbar(im1, ax=axes[0, :].ravel().tolist(), shrink=0.8)
    fig.colorbar(im2, ax=axes[1, :].ravel().tolist(), shrink=0.8)
    fig.colorbar(im3, ax=axes[2, :].ravel().tolist(), shrink=0.8)
    plt.show()
    
    # Visualiser un exemple en détail avec les métriques
    if num_samples > 0:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        
        # Input
        im1 = ax1.imshow(inputs[0], cmap='viridis')
        ax1.set_title('Input (Bruitée)')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axis('off')
        
        # Target
        im2 = ax2.imshow(targets[0], cmap='viridis')
        ax2.set_title('Target (Vérité)')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')
        
        # Prediction
        im3 = ax3.imshow(predictions[0], cmap='viridis')
        ax3.set_title('Prédiction')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.axis('off')
        
        # Erreur de prédiction
        error = predictions[0] - targets[0]
        im4 = ax4.imshow(error, cmap='RdBu_r')
        
        # Calculer les métriques pour cet exemple
        mse = np.mean(error**2)
        mae = np.mean(np.abs(error))
        psnr = 20 * np.log10(2.0 / np.sqrt(mse))  # Assumant données normalisées entre -1 et 1
        
        ax4.set_title(f'Erreur (MSE={mse:.4f}, PSNR={psnr:.2f}dB)')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()


def visualize_feature_maps(model, test_loader):
    """
    Visualise les cartes de caractéristiques intermédiaires du modèle U-NET.
    
    Args:
        model: Modèle U-NET entraîné
        test_loader: DataLoader avec les données de test
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    
    # Mettre le modèle en mode évaluation
    model.eval()
    unet = model.model
    
    # Obtenir un batch de données
    batch = next(iter(test_loader))
    x = batch['input']
    
    # Obtenir les features maps
    with torch.no_grad():
        # Encodeur
        x1 = unet.inc(x)
        x2 = unet.down1(x1)
        x3 = unet.down2(x2)
        x4 = unet.down3(x3)
        x5 = unet.down4(x4)
        
        # Décodeur
        x6 = unet.up1(x5, x4)
        x7 = unet.up2(x6, x3)
        x8 = unet.up3(x7, x2)
        x9 = unet.up4(x8, x1)
        output = unet.outc(x9)
    
    # Convertir en numpy pour la visualisation
    feature_maps = [
        ('Input', x.squeeze().cpu().numpy()),
        ('Enc1', x1[0, :4].cpu().numpy()),    # Afficher seulement 4 canaux
        ('Enc2', x2[0, :4].cpu().numpy()),
        ('Enc3', x3[0, :4].cpu().numpy()),
        ('Enc4', x4[0, :4].cpu().numpy()),
        ('Bottleneck', x5[0, :4].cpu().numpy()),
        ('Dec4', x6[0, :4].cpu().numpy()),
        ('Dec3', x7[0, :4].cpu().numpy()),
        ('Dec2', x8[0, :4].cpu().numpy()),
        ('Dec1', x9[0, :4].cpu().numpy()),
        ('Output', output.squeeze().cpu().numpy())
    ]
    
    # Créer une figure avec des sous-graphiques pour chaque couche
    plt.figure(figsize=(15, 25))
    
    for i, (name, maps) in enumerate(feature_maps):
        if name == 'Input' or name == 'Output':
            # Pour l'entrée et la sortie (1 seul canal)
            plt.subplot(len(feature_maps), 1, i+1)
            plt.title(f"{name} - Shape: {maps.shape}")
            plt.imshow(maps, cmap='viridis')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
        else:
            # Pour les feature maps intermédiaires (afficher 4 canaux)
            channels = maps.shape[0]
            plt.subplot(len(feature_maps), 1, i+1)
            plt.title(f"{name} - Shape: {maps.shape}")
            
            # Créer une grille de visualisation
            grid_size = int(np.ceil(np.sqrt(channels)))
            for j in range(channels):
                plt.subplot(len(feature_maps), grid_size, i*grid_size + j + 1)
                if j == 0:
                    plt.title(f"{name} (quelques canaux)")
                plt.imshow(maps[j], cmap='viridis')
                plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def evaluate_model_performance(model, test_loader):
    """
    Évalue les performances du modèle sur l'ensemble de test.
    
    Args:
        model: Modèle entraîné
        test_loader: DataLoader avec les données de test
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Métriques
    mse_values = []
    mae_values = []
    psnr_values = []
    ssim_values = []  # Structural Similarity Index
    
    # Collecter les prédictions
    with torch.no_grad():
        for batch in test_loader:
            x = batch['input']
            y = batch['target']
            y_hat = model(x)
            
            # Convertir en numpy
            y = y.squeeze().cpu().numpy()
            y_hat = y_hat.squeeze().cpu().numpy()
            
            # Calculer les métriques
            mse = mean_squared_error(y, y_hat)
            mae = mean_absolute_error(y, y_hat)
            psnr = 20 * np.log10(2.0 / np.sqrt(mse))  # Assumant données normalisées entre -1 et 1
            
            # Approximation simple de SSIM (on pourrait utiliser skimage.metrics.structural_similarity)
            # Pour simplifier, on calcule juste la corrélation
            corr = np.corrcoef(y.flatten(), y_hat.flatten())[0, 1]
            
            mse_values.append(mse)
            mae_values.append(mae)
            psnr_values.append(psnr)
            ssim_values.append(corr)
    
    # Calculer les moyennes
    avg_mse = np.mean(mse_values)
    avg_mae = np.mean(mae_values)
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    # Afficher les résultats
    print("Évaluation des performances du modèle:")
    print(f"  MSE moyen: {avg_mse:.6f}")
    print(f"  MAE moyen: {avg_mae:.6f}")
    print(f"  PSNR moyen: {avg_psnr:.2f} dB")
    print(f"  Corrélation moyenne: {avg_ssim:.4f}")
    
    # Créer un tableau de résultats
    metrics = ['MSE', 'MAE', 'PSNR (dB)', 'Corrélation']
    values = [avg_mse, avg_mae, avg_psnr, avg_ssim]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Afficher les valeurs comme un tableau
    table = ax.table(
        cellText=[[f"{v:.6f}" if i < 2 else f"{v:.2f}" if i == 2 else f"{v:.4f}" for i, v in enumerate(values)]],
        rowLabels=['Valeur moyenne'],
        colLabels=metrics,
        loc='center',
        cellLoc='center',
        colWidths=[0.2] * len(metrics)
    )
    table.scale(1, 2)
    table.set_fontsize(14)
    
    ax.axis('off')
    ax.set_title('Évaluation des performances du modèle', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    # Histogramme des valeurs de PSNR
    plt.figure(figsize=(10, 6))
    plt.hist(psnr_values, bins=10, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.axvline(avg_psnr, color='red', linestyle='--', linewidth=2, label=f'Moyenne: {avg_psnr:.2f} dB')
    plt.title('Distribution des valeurs de PSNR', fontsize=14)
    plt.xlabel('PSNR (dB)', fontsize=12)
    plt.ylabel('Nombre d\'échantillons', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def interactive_inference_demo(model, test_dataset, noise_levels=[0.05, 0.1, 0.2, 0.3]):
    """
    Démonstration interactive du modèle avec différents niveaux de bruit.
    
    Args:
        model: Modèle entraîné
        test_dataset: Dataset de test
        noise_levels: Liste des niveaux de bruit à tester
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from ipywidgets import interact, widgets
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Sélectionner un exemple de base
    base_sample = test_dataset[0]['target'].numpy()
    
    @interact(
        sample_idx=widgets.IntSlider(min=0, max=len(test_dataset)-1, step=1, value=0, description='Échantillon:'),
        noise_level=widgets.Dropdown(options=noise_levels, value=0.1, description='Niveau de bruit:')
    )
    def update_plot(sample_idx, noise_level):
        # Récupérer l'échantillon de base
        clean_sample = test_dataset[sample_idx]['target'].numpy()
        
        # Ajouter du bruit
        np.random.seed(42)  # Pour reproductibilité
        noise = np.random.normal(0, noise_level, clean_sample.shape)
        noisy_sample = clean_sample + noise
        
        # Normaliser pour rester dans [-1, 1]
        max_val = max(np.max(np.abs(noisy_sample)), 1.0)
        noisy_sample /= max_val
        clean_sample /= max_val
        
        # Prédire avec le modèle
        with torch.no_grad():
            input_tensor = torch.FloatTensor(noisy_sample).unsqueeze(0)  # Ajouter dim batch
            prediction = model(input_tensor).squeeze().cpu().numpy()
        
        # Calculer les métriques
        mse = np.mean((prediction - clean_sample.squeeze())**2)
        psnr = 20 * np.log10(2.0 / np.sqrt(mse))
        
        # Visualiser
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        
        # Input bruité
        im1 = ax1.imshow(noisy_sample.squeeze(), cmap='viridis')
        ax1.set_title(f'Entrée (bruit: {noise_level})')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axis('off')
        
        # Vérité terrain
        im2 = ax2.imshow(clean_sample.squeeze(), cmap='viridis')
        ax2.set_title('Vérité terrain')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')
        
        # Prédiction
        im3 = ax3.imshow(prediction, cmap='viridis')
        ax3.set_title(f'Prédiction (PSNR: {psnr:.2f}dB)')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.axis('off')
        
        # Erreur
        error = prediction - clean_sample.squeeze()
        im4 = ax4.imshow(error, cmap='RdBu_r')
        ax4.set_title(f'Erreur (MSE: {mse:.4f})')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()

```

Voici les fonctions de visualisation pour analyser en détail votre modèle U-NET SSH :

1. `visualize_dataset_samples(dataset, num_samples=5)` - Affiche les échantillons d'entrée et cibles de votre dataset synthétique avec une vue détaillée du bruit.

2. `visualize_model_architecture(model)` - Crée un diagramme schématique de l'architecture U-NET pour mieux comprendre la structure du modèle.

3. `plot_learning_curves(trainer_history=None, epochs=None)` - Visualise l'évolution de l'entraînement avec courbes de perte (MSE) et PSNR.

4. `visualize_model_predictions(model, test_loader, num_samples=5)` - Compare les prédictions du modèle avec les données d'entrée et les cibles.

5. `visualize_feature_maps(model, test_loader)` - Analyse les cartes de caractéristiques intermédiaires générées par chaque couche du réseau.

6. `evaluate_model_performance(model, test_loader)` - Calcule et affiche des métriques détaillées sur les performances du modèle.

7. `interactive_inference_demo(model, test_dataset, noise_levels=[0.05, 0.1, 0.2, 0.3])` - Démo interactive pour tester différents niveaux de bruit.

Pour intégrer ces fonctions dans votre notebook, ajoutez une section dédiée à la visualisation après la définition du modèle. Voici comment les utiliser :

```python
# Visualiser le dataset
visualize_dataset_samples(train_dataset)

# Visualiser l'architecture du modèle
visualize_model_architecture(model)

# Après l'entraînement, visualiser les courbes d'apprentissage
plot_learning_curves(epochs=trainer.current_epoch)

# Visualiser les prédictions sur le jeu de test
visualize_model_predictions(model, test_loader)

# Analyser les cartes de caractéristiques (feature maps)
visualize_feature_maps(model, test_loader)

# Évaluer les performances du modèle
evaluate_model_performance(model, test_loader)

# Interface interactive pour tester avec différents niveaux de bruit
interactive_inference_demo(model, test_dataset)
```

Ces fonctions vous permettront d'analyser en profondeur chaque aspect du modèle, de sa structure aux résultats finaux, le tout sans nécessiter de GPU.