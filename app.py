import pandas as pd
import json
import requests
from flask import Flask, request, jsonify
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import random
from datetime import datetime

app = Flask(__name__)

# Load the transaction data
file_path = 'mucudu-transactions.csv'
data = pd.read_csv(file_path)

# Load the menu items from JSON file
menu_file_path = 'menu_jimmy.json'
with open(menu_file_path, 'r') as file:
    menu_items = json.load(file)

# Helper function to get posId from menu_jimmy
def get_pos_id(item_name):
    for menu_item in menu_items:
        if menu_item['name'].upper() == item_name.upper():
            return menu_item['posId']
    return None

# Extract item names and posId from the menu
menu_item_names = [item['name'] for item in menu_items]
menu_item_posIds = {item['name']: item.get('posId') for item in menu_items if 'posId' in item}

# Convert 'created_at' from Unix timestamp to datetime
data['created_at'] = pd.to_datetime(data['created_at'], unit='ms')

# Extract relevant columns
transactions = data[['created_at', 'user_id_main', 'venues_id', 'MenuItems', 'amount']].copy()

# Create additional features from 'created_at'
transactions['day_of_week'] = transactions['created_at'].dt.day_name()

# Define a function to safely extract item names from 'MenuItems'
def extract_item_name(menu_items):
    if isinstance(menu_items, str):
        try:
            items = json.loads(menu_items.replace("'", "\""))  # Replace single quotes with double quotes
            if len(items) > 0:
                return items[0]['name']
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
    return None

# Apply the function to 'MenuItems' column
transactions['item'] = transactions['MenuItems'].apply(extract_item_name)

# Define a function to get the mode safely
def get_mode(series):
    mode = series.mode()
    if not mode.empty:
        return mode.iloc[0]
    return None

# Aggregate data to get user-specific features
user_features = transactions.groupby('user_id_main').agg({
    'amount': 'sum',
    'item': get_mode,  # Most frequently purchased item
    'day_of_week': get_mode  # Most frequent day of purchase
}).reset_index()

# Encode categorical variables
le_item = LabelEncoder()
le_day_of_week = LabelEncoder()

user_features['item_encoded'] = le_item.fit_transform(user_features['item'].astype(str))
user_features['day_of_week_encoded'] = le_day_of_week.fit_transform(user_features['day_of_week'].astype(str))

# Standardize the features
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features[['amount', 'item_encoded', 'day_of_week_encoded']])

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
user_features['cluster'] = kmeans.fit_predict(user_features_scaled)

# Prepare the data for association rule mining
basket = transactions.groupby(['user_id_main', 'item'])['amount'].sum().unstack().reset_index().fillna(0)
basket.set_index('user_id_main', inplace=True)

# Convert values to boolean
basket = basket.applymap(lambda x: x > 0)

# Generate frequent itemsets with a lower min_support
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Prepare the data for collaborative filtering
reader = Reader(rating_scale=(1, 5))
data_cf = Dataset.load_from_df(transactions[['user_id_main', 'item', 'amount']], reader)

# Train-test split
trainset, testset = train_test_split(data_cf, test_size=0.25, random_state=42)

# Train SVD algorithm
algo = SVD()
algo.fit(trainset)

# Make predictions
predictions = algo.test(testset)

# Function to generate new offers based on previous transactions and available menu items
def generate_new_offers(user_id, num_offers=1):
    user_cluster = user_features.loc[user_features['user_id_main'] == user_id, 'cluster'].values[0]
    cluster_rules = rules[rules['antecedents'].apply(lambda x: any(item in x for item in user_features[user_features['cluster'] == user_cluster]['item'].unique()))]

    discount_percentages = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    offer_formats = [
        "Buy one {item} and get another {discount}% off!",
        "Buy one {item1} and get a {item2} for free!",
        "Buy {item} and get {discount}% off your bill!",
        "Spend ${amount} and get {discount}% off your bill!",
        "Spend ${amount} and get ${amount_off} off your bill!",
        "Double points when you spend ${amount}!",
        "Double points with purchase of {item}!"
    ]
    offer_names = [
        "Discounted item with purchase",
        "Free item with purchase",
        "Buy item and get % off bill",
        "Spend $ get % off bill",
        "Spend $ get $ off bill",
        "Double points when you spend $",
        "Double point with items purchase"
    ]
    spend_amounts = [20, 25, 30, 35, 40, 45, 50]
    amount_offs = [5, 10, 15, 20, 25, 30]

    offers_to_return = []
    pos_ids_item1 = []
    pos_ids_item2 = []

    if not cluster_rules.empty:
        sorted_rules = cluster_rules.sort_values(by='lift', ascending=False)
        offers = [list(sorted_rules.iloc[i]['consequents'])[0] for i in range(min(num_offers, len(sorted_rules)))]

        for offer in offers:
            pos_id = get_pos_id(offer)
            print(f"Creating offer for item: {offer} (posId: {pos_id})")
            pos_ids_item1.append(pos_id)
            discount = random.choice(discount_percentages)
            amount = random.choice(spend_amounts)
            amount_off = random.choice(amount_offs)
            format_index = random.randint(0, len(offer_formats) - 1)
            offer_format = offer_formats[format_index]
            offer_name = offer_names[format_index]
            if "{item2}" in offer_format:
                item2 = random.choice([item['name'] for item in menu_items])
                item2_pos_id = get_pos_id(item2)
                offer_text = offer_format.format(item1=offer, item2=item2, discount=discount)
                print(f"Including second item: {item2} (posId: {item2_pos_id})")
                pos_ids_item2.append(item2_pos_id)
            elif "{item}" in offer_format:
                offer_text = offer_format.format(item=offer, discount=discount)
            elif "{amount_off}" in offer_format:
                offer_text = offer_format.format(amount=amount, amount_off=amount_off)
            else:
                offer_text = offer_format.format(amount=amount, discount=discount)

            offers_to_return.append(offer_text)
            write_offer_to_db(user_id, offer_text, offer_name, pos_ids_item1, pos_ids_item2)  # Write offer to database

    else:
        default_offers = random.sample([item['name'] for item in menu_items], num_offers)
        for offer in default_offers:
            pos_id = get_pos_id(offer)
            print(f"Creating default offer for item: {offer} (posId: {pos_id})")
            pos_ids_item1.append(pos_id)
            discount = random.choice(discount_percentages)
            amount = random.choice(spend_amounts)
            amount_off = random.choice(amount_offs)
            format_index = random.randint(0, len(offer_formats) - 1)
            offer_format = offer_formats[format_index]
            offer_name = offer_names[format_index]
            if "{item2}" in offer_format:
                item2 = random.choice([item['name'] for item in menu_items])
                item2_pos_id = get_pos_id(item2)
                offer_text = offer_format.format(item1=offer, item2=item2, discount=discount)
                print(f"Including second item: {item2} (posId: {item2_pos_id})")
                pos_ids_item2.append(item2_pos_id)
            elif "{item}" in offer_format:
                offer_text = offer_format.format(item=offer, discount=discount)
            elif "{amount_off}" in offer_format:
                offer_text = offer_format.format(amount=amount, amount_off=amount_off)
            else:
                offer_text = offer_format.format(amount=amount, discount=discount)

            offers_to_return.append(offer_text)
            write_offer_to_db(user_id, offer_text, offer_name, pos_ids_item1, pos_ids_item2)  # Write offer to database

    return offers_to_return, pos_ids_item1, pos_ids_item2

def write_offer_to_db(user_id, offer_text, offer_name, pos_ids_item1, pos_ids_item2):
    payload = {
        "user_id": user_id,
        "offer": offer_text,
        "created_at": datetime.now().isoformat(),
        "details": offer_text,
        "venues_id": 174,  # 174 for Jimmy's
        "image": None,
        "offer_image": "https://423c2ec2ec37482009eff6626a7be741.cdn.bubble.io/f1717049342102x805180131643686700/Mucudu%20Half%20Page%20Ad%20.png",
        "expiry": (datetime.now() + pd.DateOffset(days=30)).isoformat(),
        "loyaltyAmount": 0,
        "recommendation_index": 0,
        "Terms": "Not available with any other offer. Offer only available outside of public holidays. Only while suppliers last",
        "Redemption": "Every Monday",
        "numRedeemedAllowed": 1,
        "requiredPosIDs": pos_ids_item1,
        "rewardPosID": pos_ids_item2,
        "offerType": offer_name,
        "discountPercentage": 0,
        "discountAmount": 0,
        "requiredAmount": 0,
        "total_times_redeemed": 0,
        "Redeem_on_MR_Yum": False,
        "rewards_multiplier": 1,
        "validTimeStart": datetime.now().isoformat(),
        "validTimeEnd": (datetime.now() + pd.DateOffset(hours=24)).isoformat(),
        "validDays": [1, 2, 3, 4, 5, 6, 7],
        "active": True,
        "maxReward": 0,
        "pricePoint": 0,
        "qual_type": "Time spent in the bar",
        "qual_MinimumSpend": 0,
        "qual_QualificationPeriodHours": 0,
        "qual_MinimumPurchaseItems": [],
        "qual_MinimumPurchaseItemQty": 0,
        "qual_RedemptionRestriction": "Cash",
        "qual_title": "",
        "menuItems": [],
        "highestPriceProduct": 0,
        "location": None
    }

    return json.dumps(payload, indent=4)
    
@app.route('/offers/int:user_id', methods=['GET'])
def get_offers(user_id):
    try:
        num_offers = int(request.args.get('num_offers', 3))
        offers = generate_new_offers(user_id, num_offers)
        return jsonify({'user_id': user_id, 'offers': offers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/create_offer/<int:user_id>', methods=['GET'])import pandas as pd
import json
import requests
from flask import Flask, request, jsonify
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import random
from datetime import datetime

app = Flask(__name__)

# Load the transaction data
file_path = 'mucudu-transactions.csv'
data = pd.read_csv(file_path)

# Load the menu items from JSON file
menu_file_path = 'menu_jimmy.json'
with open(menu_file_path, 'r') as file:
    menu_items = json.load(file)

# Helper function to get posId from menu_jimmy
def get_pos_id(item_name):
    for menu_item in menu_items:
        if menu_item['name'].upper() == item_name.upper():
            return menu_item['posId']
    return None

# Extract item names and posId from the menu
menu_item_names = [item['name'] for item in menu_items]
menu_item_posIds = {item['name']: item.get('posId') for item in menu_items if 'posId' in item}

# Convert 'created_at' from Unix timestamp to datetime
data['created_at'] = pd.to_datetime(data['created_at'], unit='ms')

# Extract relevant columns
transactions = data[['created_at', 'user_id_main', 'venues_id', 'MenuItems', 'amount']].copy()

# Create additional features from 'created_at'
transactions['day_of_week'] = transactions['created_at'].dt.day_name()

# Define a function to safely extract item names from 'MenuItems'
def extract_item_name(menu_items):
    if isinstance(menu_items, str):
        try:
            items = json.loads(menu_items.replace("'", "\""))  # Replace single quotes with double quotes
            if len(items) > 0:
                return items[0]['name']
        except (json.JSONDecodeError, TypeError, KeyError):
            return None
    return None

# Apply the function to 'MenuItems' column
transactions['item'] = transactions['MenuItems'].apply(extract_item_name)

# Define a function to get the mode safely
def get_mode(series):
    mode = series.mode()
    if not mode.empty:
        return mode.iloc[0]
    return None

# Aggregate data to get user-specific features
user_features = transactions.groupby('user_id_main').agg({
    'amount': 'sum',
    'item': get_mode,  # Most frequently purchased item
    'day_of_week': get_mode  # Most frequent day of purchase
}).reset_index()

# Encode categorical variables
le_item = LabelEncoder()
le_day_of_week = LabelEncoder()

user_features['item_encoded'] = le_item.fit_transform(user_features['item'].astype(str))
user_features['day_of_week_encoded'] = le_day_of_week.fit_transform(user_features['day_of_week'].astype(str))

# Standardize the features
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features[['amount', 'item_encoded', 'day_of_week_encoded']])

# Perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
user_features['cluster'] = kmeans.fit_predict(user_features_scaled)

# Prepare the data for association rule mining
basket = transactions.groupby(['user_id_main', 'item'])['amount'].sum().unstack().reset_index().fillna(0)
basket.set_index('user_id_main', inplace=True)

# Convert values to boolean
basket = basket.applymap(lambda x: x > 0)

# Generate frequent itemsets with a lower min_support
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

# Prepare the data for collaborative filtering
reader = Reader(rating_scale=(1, 5))
data_cf = Dataset.load_from_df(transactions[['user_id_main', 'item', 'amount']], reader)

# Train-test split
trainset, testset = train_test_split(data_cf, test_size=0.25, random_state=42)

# Train SVD algorithm
algo = SVD()
algo.fit(trainset)

# Make predictions
predictions = algo.test(testset)

# Function to generate new offers based on previous transactions and available menu items
def generate_new_offers(user_id, num_offers=1):
    user_cluster = user_features.loc[user_features['user_id_main'] == user_id, 'cluster'].values[0]
    cluster_rules = rules[rules['antecedents'].apply(lambda x: any(item in x for item in user_features[user_features['cluster'] == user_cluster]['item'].unique()))]

    discount_percentages = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    offer_formats = [
        "Buy one {item} and get another {discount}% off!",
        "Buy one {item1} and get a {item2} for free!",
        "Buy {item} and get {discount}% off your bill!",
        "Spend ${amount} and get {discount}% off your bill!",
        "Spend ${amount} and get ${amount_off} off your bill!",
        "Double points when you spend ${amount}!",
        "Double points with purchase of {item}!"
    ]
    offer_names = [
        "Discounted item with purchase",
        "Free item with purchase",
        "Buy item and get % off bill",
        "Spend $ get % off bill",
        "Spend $ get $ off bill",
        "Double points when you spend $",
        "Double point with items purchase"
    ]
    spend_amounts = [20, 25, 30, 35, 40, 45, 50]
    amount_offs = [5, 10, 15, 20, 25, 30]

    offers_to_return = []
    pos_ids_item1 = []
    pos_ids_item2 = []

    if not cluster_rules.empty:
        sorted_rules = cluster_rules.sort_values(by='lift', ascending=False)
        offers = [list(sorted_rules.iloc[i]['consequents'])[0] for i in range(min(num_offers, len(sorted_rules)))]

        for offer in offers:
            pos_id = get_pos_id(offer)
            print(f"Creating offer for item: {offer} (posId: {pos_id})")
            pos_ids_item1.append(pos_id)
            discount = random.choice(discount_percentages)
            amount = random.choice(spend_amounts)
            amount_off = random.choice(amount_offs)
            format_index = random.randint(0, len(offer_formats) - 1)
            offer_format = offer_formats[format_index]
            offer_name = offer_names[format_index]
            if "{item2}" in offer_format:
                item2 = random.choice([item['name'] for item in menu_items])
                item2_pos_id = get_pos_id(item2)
                offer_text = offer_format.format(item1=offer, item2=item2, discount=discount)
                print(f"Including second item: {item2} (posId: {item2_pos_id})")
                pos_ids_item2.append(item2_pos_id)
            elif "{item}" in offer_format:
                offer_text = offer_format.format(item=offer, discount=discount)
            elif "{amount_off}" in offer_format:
                offer_text = offer_format.format(amount=amount, amount_off=amount_off)
            else:
                offer_text = offer_format.format(amount=amount, discount=discount)

            offers_to_return.append(offer_text)
            write_offer_to_db(user_id, offer_text, offer_name, pos_ids_item1, pos_ids_item2)  # Write offer to database

    else:
        default_offers = random.sample([item['name'] for item in menu_items], num_offers)
        for offer in default_offers:
            pos_id = get_pos_id(offer)
            print(f"Creating default offer for item: {offer} (posId: {pos_id})")
            pos_ids_item1.append(pos_id)
            discount = random.choice(discount_percentages)
            amount = random.choice(spend_amounts)
            amount_off = random.choice(amount_offs)
            format_index = random.randint(0, len(offer_formats) - 1)
            offer_format = offer_formats[format_index]
            offer_name = offer_names[format_index]
            if "{item2}" in offer_format:
                item2 = random.choice([item['name'] for item in menu_items])
                item2_pos_id = get_pos_id(item2)
                offer_text = offer_format.format(item1=offer, item2=item2, discount=discount)
                print(f"Including second item: {item2} (posId: {item2_pos_id})")
                pos_ids_item2.append(item2_pos_id)
            elif "{item}" in offer_format:
                offer_text = offer_format.format(item=offer, discount=discount)
            elif "{amount_off}" in offer_format:
                offer_text = offer_format.format(amount=amount, amount_off=amount_off)
            else:
                offer_text = offer_format.format(amount=amount, discount=discount)

            offers_to_return.append(offer_text)
            write_offer_to_db(user_id, offer_text, offer_name, pos_ids_item1, pos_ids_item2)  # Write offer to database

    return offers_to_return, pos_ids_item1, pos_ids_item2

def write_offer_to_db(user_id, offer_text, offer_name, pos_ids_item1, pos_ids_item2):
    payload = {
        "user_id": user_id,
        "offer": offer_text,
        "created_at": datetime.now().isoformat(),
        "details": offer_text,
        "venues_id": 174,  # 174 for Jimmy's
        "image": None,
        "offer_image": "https://423c2ec2ec37482009eff6626a7be741.cdn.bubble.io/f1717049342102x805180131643686700/Mucudu%20Half%20Page%20Ad%20.png",
        "expiry": (datetime.now() + pd.DateOffset(days=30)).isoformat(),
        "loyaltyAmount": 0,
        "recommendation_index": 0,
        "Terms": "Not available with any other offer. Offer only available outside of public holidays. Only while suppliers last",
        "Redemption": "Every Monday",
        "numRedeemedAllowed": 1,
        "requiredPosIDs": pos_ids_item1,
        "rewardPosID": pos_ids_item2,
        "offerType": offer_name,
        "discountPercentage": 0,
        "discountAmount": 0,
        "requiredAmount": 0,
        "total_times_redeemed": 0,
        "Redeem_on_MR_Yum": False,
        "rewards_multiplier": 1,
        "validTimeStart": datetime.now().isoformat(),
        "validTimeEnd": (datetime.now() + pd.DateOffset(hours=24)).isoformat(),
        "validDays": [1, 2, 3, 4, 5, 6, 7],
        "active": True,
        "maxReward": 0,
        "pricePoint": 0,
        "qual_type": "Time spent in the bar",
        "qual_MinimumSpend": 0,
        "qual_QualificationPeriodHours": 0,
        "qual_MinimumPurchaseItems": [],
        "qual_MinimumPurchaseItemQty": 0,
        "qual_RedemptionRestriction": "Cash",
        "qual_title": "",
        "menuItems": [],
        "highestPriceProduct": 0,
        "location": None
    }

    return json.dumps(payload, indent=4)
    
@app.route('/offers/int:user_id', methods=['GET'])
def get_offers(user_id):
    try:
        num_offers = int(request.args.get('num_offers', 3))
        offers = generate_new_offers(user_id, num_offers)
        return jsonify({'user_id': user_id, 'offers': offers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/create_offer/<int:user_id>', methods=['GET'])
def create_offer(user_id):
    try:
        num_offers = int(request.args.get('num_offers', 1))
        offers_payloads = []

        # Read the CSV file
        data = pd.read_csv('new_top_200_users_with_items.csv')
        
        # Loop through each row in the CSV
        for index, row in data.iterrows():
            user_id = row['user_id_main']
            item_name = row['name']
            
            print(f'Index: {index}, Ite name: {name}')
            # Create an offer for each item
            # create_offer(user_id, item_name)
        
        # Generate the specified number of offers
        offers, pos_ids_item1, pos_ids_item2 = generate_new_offers(user_id, num_offers)

        for i in range(num_offers):
            if i < len(offers):
                offer_text = offers[i]
                offer_name = "Generated Offer"
                payload = write_offer_to_db(user_id, offer_text, offer_name, pos_ids_item1, pos_ids_item2)
                offers_payloads.append(json.loads(payload))
            else:
                break

        return jsonify({'user_id': user_id, 'offers': offers_payloads})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
def create_offer(user_id):
    try:
        num_offers = int(request.args.get('num_offers', 1))
        offers_payloads = []
        
        # Generate the specified number of offers
        offers, pos_ids_item1, pos_ids_item2 = generate_new_offers(user_id, num_offers)

        for i in range(num_offers):
            if i < len(offers):
                offer_text = offers[i]
                offer_name = "Generated Offer"
                payload = write_offer_to_db(user_id, offer_text, offer_name, pos_ids_item1, pos_ids_item2)
                offers_payloads.append(json.loads(payload))
            else:
                break

        return jsonify({'user_id': user_id, 'offers': offers_payloads})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
