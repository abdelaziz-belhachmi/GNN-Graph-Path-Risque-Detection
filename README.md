# GNN-Graph-Path-Risque-Detection | Détection de chemins à risque (Graph + GNN)

## Objectifs pédagogiques

- Identifier des chemins critiques dans un réseau
- Exploiter la propagation de risque dans un graphe
- Utiliser un GNN pour scorer des liens réseau

## Contexte

Un chemin réseau peut devenir critique si :

- il traverse des nœuds chargés
- il utilise des liens dégradés
- il concentre plusieurs flux importants

### Dataset

**Internet Topology Zoo** : https://topology-zoo.org (Topologie)

**Dataset tabulaire synthétique** (charges, latences) fourni ou généré.

Table :
- liens (latence, perte, utilisation)
- nœuds (charge)

## Construction du graphe

### Node features

- charge
- centralité

### Edge features

- latence
- bande passante
- taux d'utilisation

## Travail demandé

### Étape 1 — Graphe

- Construire le graphe à partir des tables

### Étape 2 — GNN

- Apprendre un score de risque par lien
- Propager l'information dans le graphe

### Étape 3 — Analyse

- Identifier chemins critiques
- Comparer avec métriques statiques

## Livrable

- Carte des chemins à risque
- Rapport et présentation
