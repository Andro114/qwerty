from flask import Flask, jsonify,render_template, request
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split







app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

class KMeans:
    def __init__(self, n_cluster=4, random_state=42, max_iters=300, tol=0.0001):
        self.n_cluster = n_cluster
        self.random_state = random_state
        self.max_iters = max_iters
        self.tol = tol  
        self.centroids = None
        self.clusters = None
    def fit(self, dataset):
        self.X = dataset.iloc[:, [0, 1]].values  
        np.random.seed(self.random_state)
        random_indices = np.random.choice(self.X.shape[0], self.n_cluster, replace=False)
        centroids = self.X[random_indices]
        
        for iteration in range(self.max_iters):
            clusters = self.assign_clusters(centroids)
            new_centroids = np.array([self.X[clusters == k].mean(axis=0) for k in range(self.n_cluster)])
            if np.a5ll(np.abs(new_centroids - centroids) < self.tol):
                print(f"Konvergen pada iterasi {iteration + 1}")
                break
            
            centroids = new_centroids
        
        self.centroids = centroids
        self.clusters = clusters
        self.plot_clusters()
        silhouette = self.calculate_silhouette_score()
        print(f"Silhouette Score: {silhouette:.4f}")

        return clusters, centroids
    def predict(self, new_data):
        # Pastikan data baru adalah array numpy
        new_data = np.array(new_data)
        distances = np.linalg.norm(new_data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    def assign_clusters(self, centroids):
        distances = np.linalg.norm(self.X[:, np.newaxis] - centroids, axis=2) 
        return np.argmin(distances, axis=1) 
    
    def plot_clusters(self):

        current_palette = sns.color_palette()
        colors = {i: current_palette[i] for i in range(len(self.centroids))}
        sns.palplot(current_palette[:len(self.centroids)])
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x=self.X[:, 0], y=self.X[:, 1], hue=self.clusters, palette=colors, legend=False, ax=ax)

        sns.scatterplot(x=self.centroids[:, 0], y=self.centroids[:, 1],
                        hue=list(range(len(self.centroids))),
                        palette=colors,
                        s=200, alpha=1, legend=False, ax=ax)

        for i, (x, y) in enumerate(self.centroids):
            ax.text(x, y, f"Cluster {i}", fontsize=12, fontweight='bold',
                    color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

        plt.title('Persebaran Cluster', fontsize=16)
        plt.show()

    def calculate_silhouette_score(self):
        silhouette_scores = []
        for i in range(self.X.shape[0]):
            same_cluster = self.clusters == self.clusters[i]
            other_clusters = self.clusters != self.clusters[i]
            a = np.mean(np.linalg.norm(self.X[same_cluster] - self.X[i], axis=1))  
            b = np.min([np.mean(np.linalg.norm(self.X[other_clusters & (self.clusters == k)] - self.X[i], axis=1)) for k in range(self.n_cluster) if k != self.clusters[i]]) 
            silhouette_scores.append((b - a) / max(a, b))  
        return np.mean(silhouette_scores)




# random_forest_model = load('model\\best_rf_model.joblib')
# kmeans_model80 = load('kmeans_over_80.joblib')
# kmeans_model6080 = load('kmeans_60-80.joblib')
# kmeans_model60 = load('kmeans_under_60.joblib')
# Scalerrandomforest = load("model\scalerrandomforest.joblib")
# scaler80 = load("standardsclae80.joblib")
# scaler6080 = load("standardsclae6080.joblib")
# scaler60 = load("standardsclae60.joblib")
# pca80 = load("pca80.joblib")
# pca6080 = load('pca6080.joblib')
# pca60 = load("pca60.joblib")



kmeans_modelall = load('model\kmeans_all.joblib')
scalerall = load("model\standardsclaeall.joblib")
pcaall = load("model\pcaall.joblib")

df = pd.read_csv('model\players_22.csv')

work_rate_mapping = {
    'Medium/Low': 0,
    'High/Medium': 1,
    'High/Low': 2,
    'High/High': 3,
    'Medium/Medium': 4,
    'Medium/High': 5,
    'Low/High': 6,
    'Low/Medium': 7,
    'Low/Low': 8
}

primary_position_mapping = {
    'RW': 0,
    'ST': 1,
    'LW': 2,
    'CM': 3,
    'GK': 4,
    'CDM': 5,
    'CF': 6,
    'LM': 7,
    'CB': 8,
    'CAM': 9,
    'LB': 10,
    'RB': 11,
    'RM': 12,
    'LWB': 13,
    'RWB': 14
}
encoding_map = {
    'Unique': 0,
    'Normal (170-185)': 1,
    'Lean (170-185)': 2,
    'Normal (185+)': 3,
    'Lean (185+)': 4,
    'Normal (170-)': 5,
    'Stocky (185+)': 6,
    'Lean (170-)': 7,
    'Stocky (170-185)': 8,
    'Stocky (170-)': 9
}
scale_map = {0: 0, 1: 20, 2: 40, 3: 60, 4: 80, 5: 100}

df['body_type'] = df['body_type'].map(encoding_map)
df['skill_moves'] = df['skill_moves'].map(scale_map)
df['weak_foot'] = df['weak_foot'].map(scale_map)

df['work_rate'] = df['work_rate'].map(work_rate_mapping)
df['primary_position'] = df['primary_position'].map(primary_position_mapping)

featuress = [
    "skill_moves",  "pace", "shooting", "passing", "dribbling",
    "defending", "physic", "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy",
    "skill_long_passing", "skill_ball_control", "movement_acceleration", "movement_sprint_speed",
    "movement_agility", "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
    "power_stamina", "power_strength", "power_long_shots", "mentality_aggression", "mentality_interceptions",
    "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning",
    "goalkeeping_reflexes",'weak_foot','work_rate','primary_position',"body_type"]

Xrf = df[featuress] 
Yrf = df['overall']  
Yrf = Yrf.values.ravel()
X_train, X_test, y_train, y_test = train_test_split(Xrf,Yrf, random_state=42, test_size=0.20)

Scalerrandomforest = MinMaxScaler(feature_range=(47, 93))
X_train = Scalerrandomforest.fit_transform(X_train)
X_test = Scalerrandomforest.transform(X_test)

random_forest_model = RandomForestRegressor(n_estimators= 100, max_depth= None, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt',random_state=42)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)


@app.route('/')
def index():
    return render_template('posisi.html')
# Fungsi untuk prediksi
def predik_overall(input_data):
    # Mapping kategori ke numerik
    input_data['body_type'] = encoding_map[input_data['body_type']]
    input_data['skill_moves'] = scale_map[int(input_data['skill_moves'])]
    input_data['weak_foot'] = scale_map[int(input_data['weak_foot'])]
    input_data['work_rate'] = work_rate_mapping[input_data['work_rate']]
    input_data['primary_position'] = primary_position_mapping[input_data['primary_position']]
    # Buat DataFrame dari input
    test_data = pd.DataFrame([input_data])

    # Pilih fitur yang diperlukan untuk model
    fitur_random_forest = [
        "skill_moves", "pace", "shooting", "passing", "dribbling", 
        "defending", "physic", "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
        "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy",
        "skill_long_passing", "skill_ball_control", "movement_acceleration", "movement_sprint_speed",
        "movement_agility", "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
        "power_stamina", "power_strength", "power_long_shots", "mentality_aggression", "mentality_interceptions",
        "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure",
        "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
        "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning",
        "goalkeeping_reflexes", "weak_foot", "work_rate","primary_position","body_type"
    ]

    # Ambil fitur dari input dan lakukan scaling
    X_random_forest = test_data[fitur_random_forest]
    X_random_forest = Scalerrandomforest.transform(X_random_forest)

    # Prediksi menggunakan model Random Forest
    test_data['overall'] = np.ceil(random_forest_model.predict(X_random_forest)).astype(int)

    return test_data

# Endpoint untuk menerima input
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    result = predik_overall(input_data)
    print(input_data)
    logging.debug(f"Input data received: {input_data}")

 
    fitur_random_forest = [
        "skill_moves", "pace", "shooting", "passing", "dribbling", 
        "defending", "physic", "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
        "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy",
        "skill_long_passing", "skill_ball_control", "movement_acceleration", "movement_sprint_speed",
        "movement_agility", "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
        "power_stamina", "power_strength", "power_long_shots", "mentality_aggression", "mentality_interceptions",
        "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure",
        "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
        "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning",
        "goalkeeping_reflexes", "weak_foot", "work_rate","primary_position",'body_type'
    ]
    if not all(key in input_data for key in fitur_random_forest):
        return jsonify({"error": "Invalid input data"}), 400
  
    input_data['overall'] = result['overall'].values[0]
    test_data = pd.DataFrame([input_data])
    fitur_kmeans = [feature for feature in fitur_random_forest if feature not in ["weak_foot"]]
    for feature in fitur_kmeans:
        if  feature not in ["primary_position","work_rate",'body_type']:
            test_data[f'{feature}_normalized'] = test_data[feature] / test_data['overall']

    normalized_features = [f"{feature}_normalized" for feature in fitur_kmeans if  feature not in ["primary_position","work_rate",'body_type']]
    X_kmeans = test_data[normalized_features+["primary_position","work_rate",'body_type']]
    print(X_kmeans)
    cluster_to_position = {
    1: 'Goalkeeper', 
    0: 'Defender',    
    2: 'Balance',  
    3: 'Forward'
    }
    print("Feature names from scalerall:", scalerall.feature_names_in_)
    print("Feature names from X_kmeans:", X_kmeans.columns)

    X_kmeansall = scalerall.transform(X_kmeans)
    X_kmeans_all = pcaall.transform(X_kmeansall)
    U = np.transpose(pcaall.components_)
    C = pd.DataFrame(X_kmeansall.dot(U))
    Datatestall = C.iloc[:, [0, 1]]
    test_data['kmeans_cluster'] = kmeans_modelall.predict(Datatestall)
    test_data['position'] = test_data['kmeans_cluster'].map(cluster_to_position)
 
    columns_to_drop = [f"{feature}_normalized" for feature in fitur_kmeans if feature not in ["primary_position","work_rate","body_type"]]
    test_data = test_data.drop(columns=columns_to_drop)

    df_to_save = test_data.copy()
    df_to_save['overall_predicted'] = int(result['overall'].values[0])  # Tambahkan hasil prediksi overall
    df_to_save.to_csv('predictions_with_clustering.csv', mode='a', index=False,header=False)

    return jsonify({
        "overall": int(result['overall'].values[0]),
        "position": test_data['position'].iloc[0],
        "kmeans_cluster": int(test_data['kmeans_cluster'].iloc[0])
    })

@app.route('/get_leagues', methods=['GET'])
def get_leagues():
    # Membaca file CSV menggunakan pandas
    df = pd.read_csv('model\players_22.csv')

    # Mengonversi data ke format JSON
    leagues = {}
    for _, row in df.iterrows():
        league = str(row['league_name'])  # Pastikan league_name adalah string
        club = str(row['club_name'])     # Pastikan club_name adalah string
        if league not in leagues:
            leagues[league] = set()  # Gunakan set untuk memastikan klub unik
        leagues[league].add(club)  # Menambahkan klub ke dalam set
    
    # Mengonversi kembali set ke list agar bisa diterima oleh frontend
    for league in leagues:
        leagues[league] = list(leagues[league])

    # Mengirimkan data JSON ke frontend
    return jsonify(leagues)



import itertools

@app.route('/get_player_info', methods=['POST'])
def get_player_info():
    data = request.json  # Get data sent from the frontend
    club_name = data.get('club_name')
    formasi_input = data.get('formasi_input')

    # Load the player dataset
    player_df0 = pd.read_csv('model\players_22.csv')

    # Define selected features
    selected_features = [
        "skill_moves", "pace", "shooting", "passing", "dribbling", "skill_moves",
        "defending", "physic", "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
        "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve", "skill_fk_accuracy",
        "skill_long_passing", "skill_ball_control", "movement_acceleration", "movement_sprint_speed",
        "movement_agility", "movement_reactions", "movement_balance", "power_shot_power", "power_jumping",
        "power_stamina", "power_strength", "power_long_shots", "mentality_aggression", "mentality_interceptions",
        "mentality_positioning", "mentality_vision", "mentality_penalties", "mentality_composure",
        "defending_marking_awareness", "defending_standing_tackle", "defending_sliding_tackle",
        "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking", "goalkeeping_positioning",
        "goalkeeping_reflexes", "weak_foot","body_type", "work_rate", "short_name", "club_name", "league_name", "overall","primary_position","sofifa_id"
    ]

    player_df = player_df0[selected_features]
    cluster_df = pd.read_csv('model\player_cluster_all.csv')
    cluster_df = cluster_df.drop(columns=['short_name', 'primary_position'])

    player_df = player_df.merge(cluster_df, on='sofifa_id')
    player_df = player_df.sort_values(by=['overall'], ascending=[False])
    print(player_df)

    # Determine formation based on the user's input
    if formasi_input == "4-3-3":
        formasi = {'Goalkeeper': 1, 'Defender': 4, 'Balance': 3, 'Forward': 3}
    elif formasi_input == "3-5-2":
        formasi = {'Goalkeeper': 1, 'Defender': 3, 'Balance': 5, 'Forward': 2}
    elif formasi_input == "4-4-2":
        formasi = {'Goalkeeper': 1, 'Defender': 4, 'Balance': 4, 'Forward': 2}
    else:
        formasi = {'Goalkeeper': 1, 'Defender': 4, 'Balance': 3, 'Forward': 3}

    # Function to calculate player value based on position
    def calculate_player_value(player, position):
        if position == 'Goalkeeper':
            features = [
                'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                'goalkeeping_positioning', 'goalkeeping_reflexes',
                'power_jumping', 'mentality_composure', 'skill_long_passing'
            ]
        elif position == 'Defender':
            features = [
                'defending', 'defending_marking_awareness', 'defending_standing_tackle',
                'defending_sliding_tackle', 'physic',
                'mentality_aggression', 'mentality_interceptions', 'power_strength',
                'power_jumping', 'power_stamina', 'attacking_short_passing'
            ]
        elif position == 'Balance':
            features = [
                'passing', 'dribbling', 'movement_agility', 'movement_reactions',
                'attacking_short_passing', 'skill_long_passing', 'skill_curve',
                'skill_dribbling', 'skill_ball_control', 'mentality_interceptions',
                'defending_marking_awareness', 'mentality_vision', 'mentality_composure',
                'mentality_aggression', 'movement_balance', 'power_stamina', 'power_strength'
            ]
        elif position == 'Forward':
            features = [
                'shooting', 'attacking_finishing', 'attacking_heading_accuracy',
                'power_shot_power', 'movement_acceleration', 'attacking_volleys',
                'skill_moves', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
                'skill_ball_control', 'movement_sprint_speed', 'movement_agility',
                'movement_reactions', 'power_long_shots', 'power_jumping',
                'mentality_positioning', 'mentality_vision'
            ]

        return player[features].mean()

    # Filter players for the selected club
    club_players = player_df[player_df['club_name'] == club_name]

    # Brute force selection of players based on formation
    def brute_force_selection(eligible_players, count):
        if len(eligible_players) < count:
            return eligible_players

        all_combinations = itertools.combinations(eligible_players, count)
        best_combination = None
        best_value = -float('inf')

        for combination in all_combinations:
            total_value = sum(player[1] for player in combination)
            if total_value > best_value:
                best_value = total_value
                best_combination = combination

        return best_combination

    lineup = {}
    used_players = set()

    for position, count in formasi.items():
        eligible_players = club_players[
            (club_players['Position'] == position) & 
            (~club_players['short_name'].isin(used_players))]

        players_with_value = []
        for pl, player in eligible_players.iterrows():
            value = calculate_player_value(player, position)
            players_with_value.append((player['short_name'], value))

        selected_players = brute_force_selection(players_with_value, count)
        if len(selected_players) < count:
            remaining_count = count - len(selected_players)
            additional_players = club_players[
                (~club_players['short_name'].isin(used_players)) & 
                (~club_players['Position'].isin([position]))]

            additional_players_with_value = []
            for _, player in additional_players.iterrows():
                value = calculate_player_value(player, position)
                additional_players_with_value.append((player['short_name'], value))

            additional_selected = sorted(additional_players_with_value, key=lambda x: x[1], reverse=True)[:remaining_count]
            selected_players += additional_selected

        for player in selected_players:
            used_players.add(player[0])
        lineup[position] = club_players[club_players['short_name'].isin([player[0] for player in selected_players])]

    final_lineup = pd.concat(lineup.values())
    print(final_lineup[['short_name', 'Position', 'overall']].to_dict(orient='records'))
    return jsonify(final_lineup[['short_name', 'Position', 'overall']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
