# app.py - Complete Fraud Detection with Spark Integration
from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import socket
import time
import webbrowser
import urllib.request
import urllib.error

# Import PySpark with error handling
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, avg, count, sum as spark_sum
    PYSPARK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PySpark not available: {e}")
    print("⚠️ Running without Spark functionality")
    PYSPARK_AVAILABLE = False

app = Flask(__name__)

# Global variables
model = None
scaler = None
label_encoders = {}
transaction_data = None
current_uploaded_data = None
spark = None
SPARK_UI_PORT = None
SPARK_UI_URL = None

# -------------------------
# Utility: find free port
# -------------------------
def is_port_free(port, host='127.0.0.1'):
    """Return True if port is free on host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(0.5)
            s.bind((host, port))
            return True
        except OSError:
            return False

# -------------------------
# Initialize Spark Session
# -------------------------
def initialize_spark():
    """Initialize Spark session for big data processing."""
    global spark, SPARK_UI_PORT, SPARK_UI_URL
    
    if not PYSPARK_AVAILABLE:
        print("❌ PySpark is not installed. Please install with: pip install pyspark")
        return False
    
    try:
        print("\n🚀 Initializing Spark Session...")
        
        # Find a free port for Spark UI
        for port in range(4040, 4100):
            if is_port_free(port):
                SPARK_UI_PORT = port
                print(f"✅ Found free port: {port}")
                break
        
        if SPARK_UI_PORT is None:
            print("❌ No free ports found for Spark UI")
            return False
        
        try:
            # Build SparkSession
            spark_builder = SparkSession.builder \
                .appName("Bank Fraud Detection Analytics") \
                .master("local[*]") \
                .config("spark.ui.port", str(SPARK_UI_PORT)) \
                .config("spark.driver.bindAddress", "127.0.0.1") \
                .config("spark.driver.host", "127.0.0.1") \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "2g") \
                .config("spark.sql.shuffle.partitions", "4") \
                .config("spark.default.parallelism", "4")
            
            spark = spark_builder.getOrCreate()
            
            # Test if Spark is working
            test_df = spark.range(100)
            test_count = test_df.count()
            
            # Get Spark UI URL
            SPARK_UI_URL = f"http://localhost:{SPARK_UI_PORT}"
            
            print(f"✅ Spark initialized successfully!")
            print(f"🌐 Spark Web UI: {SPARK_UI_URL}")
            print(f"📊 Test query processed {test_count} rows")
            print(f"🔧 Application ID: {spark.sparkContext.applicationId}")
            
            # Try to open Spark UI in browser
            try:
                webbrowser.open(SPARK_UI_URL)
                print(f"🌐 Opened Spark UI in browser")
            except:
                print(f"⚠️ Could not open browser automatically")
            
            return True
            
        except Exception as e:
            print(f"❌ Spark initialization failed: {str(e)}")
            traceback.print_exc()
            spark = None
            SPARK_UI_PORT = None
            SPARK_UI_URL = None
            return False
            
    except Exception as e:
        print(f"❌ Error in Spark initialization: {str(e)}")
        traceback.print_exc()
        return False

# -------------------------
# Spark UI Status Checker
# -------------------------
def check_spark_ui_status():
    """Check if Spark UI is accessible"""
    global SPARK_UI_URL
    
    if not SPARK_UI_URL:
        return False
    
    try:
        req = urllib.request.Request(SPARK_UI_URL)
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.getcode() == 200
    except:
        return False

# -------------------------
# Restart Spark if needed
# -------------------------
def restart_spark_session():
    """Restart Spark session if it's not working"""
    global spark, SPARK_UI_PORT, SPARK_UI_URL
    
    print("\n🔄 Attempting to restart Spark session...")
    
    # Stop existing session if exists
    if spark is not None:
        try:
            spark.stop()
            print("✅ Stopped existing Spark session")
        except:
            pass
    
    # Reset globals
    spark = None
    SPARK_UI_PORT = None
    SPARK_UI_URL = None
    
    # Reinitialize
    return initialize_spark()

# -------------------------
# Feature engineering class
# -------------------------
class AdvancedFeatureEngineer:
    def create_features(self, df):
        """Create advanced features for fraud detection"""
        df = df.copy()
        
        # Ensure expected columns exist
        if 'NewBalance' not in df.columns and 'newbalanceOrig' in df.columns:
            df['NewBalance'] = df['newbalanceOrig']
        if 'OldBalance' not in df.columns and 'oldbalanceOrg' in df.columns:
            df['OldBalance'] = df['oldbalanceOrg']
        if 'AmountTaken' not in df.columns and 'Amount' in df.columns:
            df['AmountTaken'] = df['Amount']
        
        # Fill missing values
        numeric_cols = ['OldBalance', 'NewBalance', 'AmountTaken', 'Amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Basic financial features
        if 'OldBalance' in df.columns and 'NewBalance' in df.columns:
            df['BalanceRatio'] = df['NewBalance'] / (df['OldBalance'] + 1)
        
        if 'AmountTaken' in df.columns and 'OldBalance' in df.columns:
            df['AmountToBalanceRatio'] = df['AmountTaken'] / (df['OldBalance'] + 1)
        
        if 'NewBalance' in df.columns and 'OldBalance' in df.columns:
            df['IsNegativeBalance'] = ((df['NewBalance'] < 0) & (df['OldBalance'] > 0)).astype(int)
        
        # Enhanced risk countries
        high_risk_countries = {
            'Russia': 1.0, 'Nigeria': 0.9, 'China': 0.8,
            'Ukraine': 0.9, 'Vietnam': 0.7, 'North Korea': 1.0
        }
        if 'Location' in df.columns:
            df['CountryRiskScore'] = df['Location'].map(lambda x: high_risk_countries.get(x, 0.1))
        else:
            df['CountryRiskScore'] = 0.1
        
        # Merchant risk categories
        high_risk_merchants = ['Casino', 'Digital Goods', 'Cryptocurrency', 'Adult', 'Gambling']
        if 'MerchantCategory' in df.columns:
            df['IsHighRiskMerchant'] = df['MerchantCategory'].isin(high_risk_merchants).astype(int)
        else:
            df['IsHighRiskMerchant'] = 0
        
        # Time-based features
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['Hour'] = df['Timestamp'].dt.hour.fillna(12).astype(int)
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek.fillna(0).astype(int)
            df['IsNight'] = ((df['Hour'] < 6) | (df['Hour'] >= 22)).astype(int)
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        else:
            df['Hour'] = 12
            df['DayOfWeek'] = 0
            df['IsNight'] = 0
            df['IsWeekend'] = 0
        
        # Behavioral features
        if 'AmountTaken' in df.columns and 'OldBalance' in df.columns:
            df['LargeTransaction'] = (df['AmountTaken'] > df['OldBalance'] * 0.5).astype(int)
        
        if 'AmountTaken' in df.columns:
            df['SuspiciousAmount'] = (df['AmountTaken'] % 100 == 0).astype(int)
        
        return df

# -------------------------
# Dataset creation function
# -------------------------
def create_bank_transaction_dataset():
    """Create comprehensive bank transaction dataset"""
    print("📝 Creating bank transaction dataset...")
    
    np.random.seed(42)
    n_transactions = 200
    
    data = {
        'TransactionID': [f'TXN{str(i).zfill(5)}' for i in range(1, n_transactions + 1)],
        'Timestamp': [
            (datetime.now() - timedelta(hours=np.random.randint(0, 720))).strftime('%Y-%m-%d %H:%M:%S')
            for _ in range(n_transactions)
        ],
        'AccountID': [f'ACC{str(i).zfill(3)}' for i in range(1, 21) for _ in range(10)],
        'AccountName': [f'Customer {i}' for i in range(1, 21) for _ in range(10)],
        'OldBalance': np.random.randint(1000, 50000, n_transactions),
        'NewBalance': np.random.randint(-1000, 48000, n_transactions),
        'Amount': np.random.randint(10, 5000, n_transactions),
        'CardUsed': np.random.choice(
            ['Credit Card', 'Debit Card', 'Prepaid Card', 'Corporate Card', 'Virtual Card'],
            n_transactions,
            p=[0.4, 0.3, 0.1, 0.15, 0.05]
        ),
        'TransactionMode': np.random.choice(
            ['ATM Withdrawal', 'POS Swipe', 'Online Transfer', 'Mobile Payment', 'Bank Transfer', 'Contactless Payment'],
            n_transactions,
            p=[0.2, 0.25, 0.15, 0.2, 0.1, 0.1]
        ),
        'Location': np.random.choice(
            ['New York', 'California', 'Texas', 'Florida', 'Illinois', 'Russia', 'Nigeria', 'China', 'UK', 'Germany'],
            n_transactions,
            p=[0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.15]
        ),
        'MerchantCategory': np.random.choice(
            ['Retail', 'ATM', 'E-commerce', 'Restaurant', 'Travel', 'Entertainment', 'Utilities', 'Healthcare'],
            n_transactions,
            p=[0.25, 0.15, 0.2, 0.1, 0.1, 0.05, 0.1, 0.05]
        ),
        'DeviceType': np.random.choice(['Mobile', 'Desktop', 'Tablet', 'ATM', 'POS Terminal'], n_transactions),
        'UserBehaviorScore': np.random.uniform(0.1, 0.99, n_transactions),
        'TransactionStatus': np.random.choice(['Completed', 'Failed', 'Pending'], n_transactions, p=[0.85, 0.1, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # Create fraud patterns
    df['AmountTaken'] = df['Amount']
    df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour.fillna(12).astype(int)
    
    fraud_conditions = (
        (df['Location'].isin(['Russia', 'Nigeria', 'China'])) |
        (df['CardUsed'] == 'Virtual Card') |
        (df['TransactionMode'] == 'Online Transfer') |
        (df['Amount'] > df['OldBalance'] * 0.7) |
        (df['NewBalance'] < -500) |
        (df['DeviceType'].isin(['ATM', 'POS Terminal'])) |
        (df['UserBehaviorScore'] < 0.3) |
        ((df['CardUsed'] == 'Credit Card') & (df['Amount'] > 3000)) |
        ((df['TransactionMode'] == 'ATM Withdrawal') & (df['Hour'] < 6))
    )
    
    df['IsFraud'] = (fraud_conditions & (np.random.random(n_transactions) > 0.4)).astype(int)
    
    # Add risk levels
    df['Risk'] = np.where(df['IsFraud'] == 1, 'HIGH',
                         np.where(df['UserBehaviorScore'] < 0.5, 'MEDIUM', 'LOW'))
    
    # Add probability scores
    df['Probability'] = np.where(df['IsFraud'] == 1,
                                np.random.uniform(70, 95, n_transactions),
                                np.random.uniform(5, 40, n_transactions))
    
    # Save to file
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/bank_transactions_dataset.csv', index=False)
    
    print("✅ Bank transaction dataset created successfully!")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Fraud rate: {df['IsFraud'].mean():.2%}")
    
    return df

# -------------------------
# Initialize system
# -------------------------
def initialize_system():
    """Initialize the fraud detection system"""
    global model, scaler, label_encoders, transaction_data
    
    try:
        print("\n" + "="*60)
        print("🚀 INITIALIZING FRAUD DETECTION SYSTEM")
        print("="*60)
        
        # 1. Initialize Spark
        spark_initialized = initialize_spark()
        
        if not spark_initialized:
            print("⚠️ Spark initialization failed. Running without Spark analytics.")
        
        # 2. Create directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        # 3. Load or create transaction data
        if not os.path.exists('data/bank_transactions_dataset.csv'):
            print("\n📝 Creating bank transaction dataset...")
            transaction_data = create_bank_transaction_dataset()
        else:
            print("\n📂 Loading bank transaction data...")
            transaction_data = pd.read_csv('data/bank_transactions_dataset.csv')
            print(f"✅ Loaded {len(transaction_data)} transactions")
        
        print(f"📊 Transaction data shape: {transaction_data.shape}")
        print(f"👥 Unique accounts: {len(transaction_data['AccountID'].unique())}")
        print(f"🎯 Fraud rate: {transaction_data['IsFraud'].mean():.2%}")
        
        # 4. Train or load model
        if not os.path.exists('models/fraud_model.pkl'):
            print("\n🤖 Training fraud detection model...")
            train_enhanced_model()
        else:
            print("\n📚 Loading pre-trained model...")
            try:
                model = joblib.load('models/fraud_model.pkl')
                scaler = joblib.load('models/scaler.pkl')
                label_encoders = joblib.load('models/label_encoders.pkl')
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                train_enhanced_model()
        
        print("\n" + "="*60)
        print("✅ SYSTEM INITIALIZATION COMPLETE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error initializing system: {e}")
        traceback.print_exc()
        return False

# -------------------------
# Train model
# -------------------------
def train_enhanced_model():
    """Train the fraud detection model"""
    global model, scaler, label_encoders, transaction_data
    
    print("🎯 Starting model training...")
    
    # Create advanced features
    feature_engineer = AdvancedFeatureEngineer()
    transaction_enhanced = feature_engineer.create_features(transaction_data.copy())
    
    # Encode categorical variables
    categorical_cols = ['CardUsed', 'TransactionMode', 'Location', 'MerchantCategory', 'DeviceType']
    for col in categorical_cols:
        if col in transaction_enhanced.columns:
            le = LabelEncoder()
            transaction_enhanced[col + '_encoded'] = le.fit_transform(transaction_enhanced[col].astype(str))
            label_encoders[col] = le
    
    # Select features for training
    features = [
        'OldBalance', 'NewBalance', 'Amount', 'BalanceRatio',
        'AmountToBalanceRatio', 'IsNegativeBalance', 'CountryRiskScore',
        'UserBehaviorScore', 'IsNight', 'IsWeekend', 'LargeTransaction',
        'SuspiciousAmount', 'IsHighRiskMerchant'
    ]
    
    # Add encoded columns
    for col in categorical_cols:
        encoded_col = col + '_encoded'
        if encoded_col in transaction_enhanced.columns:
            features.append(encoded_col)
    
    # Keep only features that exist
    features = [f for f in features if f in transaction_enhanced.columns]
    
    X = transaction_enhanced[features].fillna(0)
    y = transaction_enhanced['IsFraud']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_scaled, y)
    
    # Save models
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    accuracy = model.score(X_scaled, y)
    print(f"✅ Model trained! Accuracy: {accuracy:.3f}")
    
    return model, scaler, label_encoders

# -------------------------
# Spark analytics helpers
# -------------------------
def create_spark_analytics(df):
    """Create advanced analytics using Spark"""
    global spark
    
    if spark is None:
        print("⚠️ Spark not available for analytics")
        return None
    
    try:
        # Convert pandas DataFrame to Spark DataFrame
        spark_df = spark.createDataFrame(df)
        spark_df.createOrReplaceTempView("transactions")
        
        analytics_results = {}
        
        # Card fraud analysis
        card_fraud = spark.sql("""
            SELECT CardUsed,
                   COUNT(*) as total_transactions,
                   SUM(IsFraud) as fraud_count,
                   ROUND(SUM(IsFraud) * 100.0 / COUNT(*), 2) as fraud_rate
            FROM transactions
            GROUP BY CardUsed
            ORDER BY fraud_rate DESC
        """).toPandas()
        analytics_results['card_fraud_analysis'] = card_fraud.to_dict('records')
        
        # Transaction mode risk analysis
        mode_risk = spark.sql("""
            SELECT TransactionMode,
                   COUNT(*) as total_count,
                   SUM(IsFraud) as fraud_count,
                   AVG(Amount) as avg_amount,
                   ROUND(SUM(IsFraud) * 100.0 / COUNT(*), 2) as fraud_rate
            FROM transactions
            GROUP BY TransactionMode
            ORDER BY fraud_rate DESC
        """).toPandas()
        analytics_results['mode_risk_analysis'] = mode_risk.to_dict('records')
        
        print("✅ Spark analytics completed")
        return analytics_results
        
    except Exception as e:
        print(f"❌ Spark analytics error: {e}")
        return None

# -------------------------
# Visualization helpers
# -------------------------
def create_visualization(df, viz_type='risk_distribution'):
    """Create various visualizations for the dashboard"""
    plt.figure(figsize=(10, 6))
    
    if viz_type == 'risk_distribution':
        if 'Risk' in df.columns:
            risk_counts = df['Risk'].value_counts()
        elif 'IsFraud' in df.columns:
            df['Risk'] = df['IsFraud'].map({1: 'HIGH', 0: 'LOW'})
            risk_counts = df['Risk'].value_counts()
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['Risk'] = pd.cut(df[numeric_cols[0]], bins=3, labels=['LOW', 'MEDIUM', 'HIGH'])
                risk_counts = df['Risk'].value_counts()
            else:
                risk_counts = pd.Series([len(df)], index=['UNKNOWN'])
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=colors[:len(risk_counts)], startangle=90)
        plt.title('Transaction Risk Distribution', fontweight='bold')
    
    elif viz_type == 'card_usage':
        if 'CardUsed' in df.columns:
            card_usage = df['CardUsed'].value_counts()
            bars = plt.bar(card_usage.index, card_usage.values, alpha=0.7, color='#3498db')
            plt.title('Card Usage Distribution', fontweight='bold')
            plt.ylabel('Number of Transactions')
            plt.xticks(rotation=45, ha='right')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
                        ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No Card Usage Data Available', ha='center', va='center')
    
    elif viz_type == 'transaction_modes':
        if 'TransactionMode' in df.columns:
            transaction_modes = df['TransactionMode'].value_counts()
            bars = plt.bar(transaction_modes.index, transaction_modes.values, alpha=0.7, color='#9b59b6')
            plt.title('Transaction Mode Distribution', fontweight='bold')
            plt.ylabel('Number of Transactions')
            plt.xticks(rotation=45, ha='right')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
                        ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No Transaction Mode Data Available', ha='center', va='center')
    
    elif viz_type == 'fraud_by_card_type':
        if 'CardUsed' in df.columns and 'IsFraud' in df.columns:
            fraud_by_card = df.groupby('CardUsed')['IsFraud'].mean() * 100
            colors = ['#e74c3c' if rate > 20 else '#f39c12' if rate > 10 else '#2ecc71' 
                     for rate in fraud_by_card.values]
            bars = plt.bar(fraud_by_card.index, fraud_by_card.values, alpha=0.7, color=colors)
            plt.title('Fraud Rate by Card Type', fontweight='bold')
            plt.ylabel('Fraud Rate (%)')
            plt.xticks(rotation=45, ha='right')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No Fraud/Card Data Available', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return f"data:image/png;base64,{image_base64}"

# -------------------------
# Data analysis helper
# -------------------------
def analyze_uploaded_data(df):
    """Analyze uploaded CSV data and return insights"""
    analysis = {
        'total_transactions': len(df),
        'columns': list(df.columns),
        'data_types': df.dtypes.astype(str).to_dict(),
        'summary_stats': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        analysis['summary_stats'][col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    
    if 'Risk' in df.columns:
        risk_counts = df['Risk'].value_counts()
        analysis['risk_distribution'] = risk_counts.to_dict()
        analysis['high_risk_count'] = int(risk_counts.get('HIGH', 0))
        if 'Amount' in df.columns:
            analysis['high_risk_amount'] = float(df[df['Risk'] == 'HIGH']['Amount'].sum())
        elif 'AmountTaken' in df.columns:
            analysis['high_risk_amount'] = float(df[df['Risk'] == 'HIGH']['AmountTaken'].sum())
        else:
            analysis['high_risk_amount'] = 0
    else:
        analysis['risk_distribution'] = {'LOW': len(df)}
        analysis['high_risk_count'] = 0
        analysis['high_risk_amount'] = 0
    
    if 'CardUsed' in df.columns:
        analysis['card_usage'] = df['CardUsed'].value_counts().to_dict()
    
    if 'TransactionMode' in df.columns:
        analysis['transaction_modes'] = df['TransactionMode'].value_counts().to_dict()
    
    return analysis

# -------------------------
# Flask routes
# -------------------------
@app.route('/')
def home():
    """Main dashboard"""
    global transaction_data, current_uploaded_data, spark, SPARK_UI_URL
    
    # Prepare account data
    accounts_data = []
    if transaction_data is not None:
        for account_id in transaction_data['AccountID'].unique()[:10]:  # Limit to 10 accounts
            account_data = transaction_data[transaction_data['AccountID'] == account_id]
            if len(account_data) > 0:
                account_info = account_data.iloc[0]
                fraud_count = account_data['IsFraud'].sum() if 'IsFraud' in account_data.columns else 0
                
                accounts_data.append({
                    'AccountID': account_id,
                    'AccountName': account_info.get('AccountName', 'Unknown'),
                    'NewBalance': float(account_info.get('NewBalance', 0)),
                    'TransactionCount': len(account_data),
                    'FraudCount': int(fraud_count),
                    'RiskLevel': 'HIGH' if fraud_count > 0 else 'LOW'
                })
    
    display_data = current_uploaded_data if current_uploaded_data is not None else transaction_data
    
    # Create visualizations
    if display_data is not None:
        risk_chart = create_visualization(display_data, 'risk_distribution')
        card_usage_chart = create_visualization(display_data, 'card_usage')
        transaction_modes_chart = create_visualization(display_data, 'transaction_modes')
        fraud_by_card_chart = create_visualization(display_data, 'fraud_by_card_type')
        
        total_transactions = len(display_data)
        
        if current_uploaded_data is not None:
            analysis = analyze_uploaded_data(current_uploaded_data)
            fraud_count = analysis['high_risk_count']
            fraud_rate = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0
            
            if 'Amount' in display_data.columns:
                avg_transaction = display_data['Amount'].mean()
                high_risk_amount = analysis['high_risk_amount']
            elif 'AmountTaken' in display_data.columns:
                avg_transaction = display_data['AmountTaken'].mean()
                high_risk_amount = analysis['high_risk_amount']
            else:
                numeric_cols = display_data.select_dtypes(include=[np.number]).columns
                avg_transaction = display_data[numeric_cols[0]].mean() if len(numeric_cols) > 0 else 0
                high_risk_amount = 0
        else:
            fraud_count = transaction_data['IsFraud'].sum() if 'IsFraud' in transaction_data.columns else 0
            fraud_rate = (fraud_count / total_transactions) * 100 if total_transactions > 0 else 0
            avg_transaction = transaction_data['Amount'].mean() if 'Amount' in transaction_data.columns else 0
            high_risk_amount = transaction_data[transaction_data['IsFraud'] == 1]['Amount'].sum() if 'Amount' in transaction_data.columns and 'IsFraud' in transaction_data.columns else 0
    else:
        risk_chart = card_usage_chart = transaction_modes_chart = fraud_by_card_chart = ""
        total_transactions = fraud_count = fraud_rate = avg_transaction = high_risk_amount = 0
    
    # Get Spark analytics if available
    spark_analytics = None
    if display_data is not None and spark is not None:
        try:
            spark_analytics = create_spark_analytics(display_data)
        except Exception as e:
            print(f"Spark analytics failed: {e}")
            spark_analytics = None
    
    # Check Spark UI status
    spark_ui_accessible = check_spark_ui_status() if SPARK_UI_URL else False
    
    # Prepare Spark status
    spark_status = "RUNNING" if spark is not None else "NOT RUNNING"
    spark_ui_link = SPARK_UI_URL if SPARK_UI_URL else "#"
    
    # JSON serializable accounts data
    accounts_json = json.dumps(accounts_data)
    
    # Create HTML template using f-strings properly
    html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Bank Fraud Detection System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --light: #ecf0f1;
            --dark: #2c3e50;
        }}
        
        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }}
        
        .dashboard-header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid var(--secondary);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-card.high-risk {{ border-left-color: var(--danger); }}
        .stat-card.medium-risk {{ border-left-color: var(--warning); }}
        .stat-card.low-risk {{ border-left-color: var(--success); }}
        
        .stat-number {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--dark);
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .nav-tabs .nav-link {{
            border: none;
            color: var(--dark);
            font-weight: 500;
            padding: 1rem 1.5rem;
        }}
        
        .nav-tabs .nav-link.active {{
            background: white;
            border-bottom: 3px solid var(--secondary);
        }}
        
        .tab-content {{
            background: white;
            border-radius: 0 15px 15px 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            min-height: 500px;
        }}
        
        .upload-area {{
            border: 3px dashed #bdc3c7;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            transition: all 0.3s ease;
            background: #f8f9fa;
            cursor: pointer;
        }}
        
        .upload-area:hover {{
            border-color: var(--secondary);
            background: #e8f4fc;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .account-list {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 10px;
        }}
        
        .account-item {{
            padding: 12px 15px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
        }}
        
        .account-item:hover {{
            background: #f8f9fa;
            border-color: var(--secondary);
        }}
        
        .account-item.selected {{
            background: var(--secondary);
            color: white;
            border-color: var(--secondary);
        }}
        
        .risk-badge {{
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-weight: 600;
            font-size: 0.75rem;
        }}
        
        .risk-high {{ background: var(--danger); color: white; }}
        .risk-medium {{ background: var(--warning); color: white; }}
        .risk-low {{ background: var(--success); color: white; }}
        
        .spark-status-badge {{
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .spark-running {{ 
            background: #d4edda; 
            color: #155724;
            border: 2px solid #28a745;
        }}
        
        .spark-stopped {{ 
            background: #f8d7da; 
            color: #721c24;
            border: 2px solid #dc3545;
        }}
        
        .spark-ui-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .spark-ui-btn:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <!-- Header -->
        <div class="dashboard-header p-4 mb-4">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1 class="display-4 fw-bold text-dark mb-2">
                        <i class="fas fa-university text-primary"></i>
                        Bank Fraud Detection System
                    </h1>
                    <p class="lead text-muted mb-0">
                        Advanced analytics with Spark integration
                    </p>
                </div>
                <div class="col-md-4 text-end">
                    <div class="d-flex align-items-center justify-content-end gap-3">
                        <div class="spark-status-badge {'spark-running' if spark is not None else 'spark-stopped'}">
                            <i class="fas fa-bolt"></i> Spark: {spark_status}
                        </div>
                        {f'<a href="{spark_ui_link}" target="_blank" class="spark-ui-btn"><i class="fas fa-external-link-alt"></i> Spark UI</a>' if spark_ui_link != "#" else '<button class="spark-ui-btn" disabled><i class="fas fa-bolt"></i> Spark Not Available</button>'}
                        <button class="btn btn-outline-primary" onclick="resetAnalysis()">
                            <i class="fas fa-sync-alt"></i> Reset
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Quick Stats -->
        <div class="row mb-4">
            <div class="col-xl-3 col-md-6">
                <div class="stat-card">
                    <div class="stat-number">{total_transactions}</div>
                    <div class="stat-label">Total Transactions</div>
                    <small class="text-muted"><i class="fas fa-chart-line"></i> Dataset</small>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="stat-card high-risk">
                    <div class="stat-number">{fraud_count}</div>
                    <div class="stat-label">High Risk</div>
                    <small class="text-muted"><i class="fas fa-exclamation-triangle"></i> {fraud_rate:.2f}% rate</small>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="stat-card medium-risk">
                    <div class="stat-number">${avg_transaction:.0f}</div>
                    <div class="stat-label">Avg Transaction</div>
                    <small class="text-muted"><i class="fas fa-money-bill-wave"></i> Average amount</small>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="stat-card low-risk">
                    <div class="stat-number">${high_risk_amount:.0f}</div>
                    <div class="stat-label">High Risk Amount</div>
                    <small class="text-muted"><i class="fas fa-shield-alt"></i> Protected</small>
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="row">
            <!-- Left Sidebar - Account Selection -->
            <div class="col-md-4">
                <!-- Account Selection Panel -->
                <div class="chart-container">
                    <h4 class="mb-3">
                        <i class="fas fa-users"></i> Select Account to Analyze
                    </h4>
                    
                    <div class="account-list" id="accountList">
'''
    
    # Add account items
    for account in accounts_data:
        html_content += f'''
                        <div class="account-item" onclick="selectAccount('{account["AccountID"]}')" 
                             id="account-{account["AccountID"]}">
                            <div class="account-name">
                                {account["AccountID"]} - {account["AccountName"]}
                            </div>
                            <div class="account-details">
                                Balance: ${account["NewBalance"]:.0f} | 
                                Transactions: {account["TransactionCount"]} | 
                                Risk: <span class="risk-badge {'risk-high' if account['RiskLevel'] == 'HIGH' else 'risk-low'}">
                                    {account["RiskLevel"]}
                                </span>
                            </div>
                        </div>
'''
    
    html_content += f'''
                    </div>
                    
                    <button class="btn btn-outline-primary w-100 mt-3" onclick="loadAccounts()">
                        <i class="fas fa-sync-alt"></i> Refresh Accounts
                    </button>
                </div>

                <!-- Account Details -->
                <div class="chart-container mt-3">
                    <h5 class="mb-3">
                        <i class="fas fa-info-circle"></i> Account Details
                    </h5>
                    <div id="accountDetails">
                        <p class="text-muted">Select an account to view details</p>
                    </div>
                </div>
                
                <!-- Spark UI Status -->
                <div class="chart-container mt-3">
                    <h5 class="mb-3">
                        <i class="fas fa-bolt"></i> Spark UI Status
                    </h5>
                    <div class="alert {'alert-success' if spark_ui_accessible else 'alert-warning'}">
                        <p><strong>URL:</strong> {spark_ui_link if spark_ui_link != "#" else "Not available"}</p>
                        <p><strong>Status:</strong> 
                            <span class="{'text-success' if spark_ui_accessible else 'text-danger'}">
                                {'✓ Accessible' if spark_ui_accessible else '✗ Not accessible'}
                            </span>
                        </p>
                        <p><strong>Test Spark:</strong> 
                            <button class="btn btn-sm btn-primary" onclick="testSparkConnection()">
                                <i class="fas fa-plug"></i> Test Connection
                            </button>
                        </p>
                    </div>
                </div>
            </div>

            <!-- Right Side - Analytics -->
            <div class="col-md-8">
                <!-- Navigation Tabs -->
                <ul class="nav nav-tabs" id="mainTabs" role="tablist">
                    <li class="nav-item">
                        <a class="nav-link active" id="analytics-tab" data-bs-toggle="tab" href="#analytics">
                            <i class="fas fa-chart-bar"></i> Banking Analytics
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" id="csv-tab" data-bs-toggle="tab" href="#csv">
                            <i class="fas fa-file-csv"></i> CSV Analysis
                        </a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" id="spark-tab" data-bs-toggle="tab" href="#spark">
                            <i class="fas fa-bolt"></i> Spark Analytics
                        </a>
                    </li>
                </ul>

                <div class="tab-content p-4" id="mainTabsContent">
                    <!-- Analytics Tab -->
                    <div class="tab-pane fade show active" id="analytics">
                        <h4 class="mb-4"><i class="fas fa-chart-bar"></i> Banking Analytics</h4>
                        
                        <!-- Charts Row 1 -->
                        <div class="row">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <div class="chart-title">Risk Distribution</div>
                                    <img src="{risk_chart}" alt="Risk Distribution" class="img-fluid">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <div class="chart-title">Card Usage Distribution</div>
                                    <img src="{card_usage_chart}" alt="Card Usage" class="img-fluid">
                                </div>
                            </div>
                        </div>
                        
                        <!-- Charts Row 2 -->
                        <div class="row mt-4">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <div class="chart-title">Transaction Mode Distribution</div>
                                    <img src="{transaction_modes_chart}" alt="Transaction Modes" class="img-fluid">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <div class="chart-title">Fraud Rate by Card Type</div>
                                    <img src="{fraud_by_card_chart}" alt="Fraud by Card Type" class="img-fluid">
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- CSV Analysis Tab -->
                    <div class="tab-pane fade" id="csv">
                        <h4 class="mb-4"><i class="fas fa-file-csv"></i> CSV File Analysis</h4>
                        
                        <!-- File Upload Area -->
                        <div class="upload-area mb-4" id="uploadArea">
                            <i class="fas fa-cloud-upload-alt fa-3x text-primary mb-3"></i>
                            <h5>Drag & Drop your CSV file here</h5>
                            <p class="text-muted">or click to browse files</p>
                            <form id="uploadForm" enctype="multipart/form-data">
                                <input type="file" class="form-control d-none" id="csvFile" accept=".csv" required>
                                <button type="button" class="btn btn-outline-primary btn-lg mt-3" onclick="document.getElementById('csvFile').click()">
                                    <i class="fas fa-folder-open"></i> Choose File
                                </button>
                                <div class="mt-2" id="fileName">
                                    <small class="text-muted">No file chosen</small>
                                </div>
                            </form>
                        </div>

                        <!-- Analyze Button -->
                        <div class="text-center mb-4">
                            <button type="button" class="btn btn-primary btn-lg" id="analyzeBtn" onclick="analyzeCSV()" disabled>
                                <i class="fas fa-chart-pie"></i> Analyze CSV File
                            </button>
                        </div>

                        <!-- Analysis Results -->
                        <div id="analysisResults">
                            <p class="text-muted">Upload a CSV file to see analysis results</p>
                        </div>
                    </div>

                    <!-- Spark Analytics Tab -->
                    <div class="tab-pane fade" id="spark">
                        <h4 class="mb-4"><i class="fas fa-bolt"></i> Spark Analytics</h4>
'''
    
    if spark_analytics:
        html_content += '''
                        <!-- Card Fraud Analysis -->
                        <div class="chart-container mb-4">
                            <h5><i class="fas fa-credit-card"></i> Fraud Analysis by Card Type</h5>
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Card Type</th>
                                            <th>Total Transactions</th>
                                            <th>Fraud Count</th>
                                            <th>Fraud Rate (%)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
'''
        
        for item in spark_analytics.get('card_fraud_analysis', []):
            badge_class = 'bg-danger' if item.get('fraud_rate', 0) > 20 else 'bg-warning' if item.get('fraud_rate', 0) > 10 else 'bg-success'
            html_content += f'''
                                        <tr>
                                            <td>{item.get('CardUsed', 'N/A')}</td>
                                            <td>{item.get('total_transactions', 0)}</td>
                                            <td>{item.get('fraud_count', 0)}</td>
                                            <td><span class="badge {badge_class}">{item.get('fraud_rate', 0)}%</span></td>
                                        </tr>
'''
        
        html_content += '''
                                    </tbody>
                                </table>
                            </div>
                        </div>
'''
    else:
        html_content += f'''
                        <div class="alert {'alert-info' if spark is not None else 'alert-warning'}">
                            <h5><i class="fas fa-info-circle"></i> Spark Analytics</h5>
                            <p class="mb-0">{
                                'Spark analytics will be available when transaction data is loaded and Spark is running.'
                                if spark is not None else 
                                'Spark is not available. Please check Spark initialization.'
                            }</p>
                            <p class="mb-0 mt-2"><small>Try uploading a CSV file or ensure Spark is properly initialized.</small></p>
                            {'<button class="btn btn-sm btn-primary mt-2" onclick="restartSpark()"><i class="fas fa-redo"></i> Restart Spark</button>' if spark is None else ''}
                        </div>
'''
    
    html_content += '''
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentFile = null;
        let selectedAccount = null;
        let accounts = ''' + accounts_json + ''';
        
        // File input change handler
        document.getElementById('csvFile').addEventListener('change', function(e) {
            if (this.files.length > 0) {
                currentFile = this.files[0];
                document.getElementById('fileName').innerHTML = 
                    `<strong><i class="fas fa-file-csv"></i> ${currentFile.name}</strong>`;
                document.getElementById('analyzeBtn').disabled = false;
            } else {
                currentFile = null;
                document.getElementById('fileName').innerHTML = 
                    '<small class="text-muted">No file chosen</small>';
                document.getElementById('analyzeBtn').disabled = true;
            }
        });

        // Drag and drop functionality
        const uploadArea = document.getElementById('uploadArea');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#3498db';
            uploadArea.style.background = '#e8f4fc';
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#bdc3c7';
            uploadArea.style.background = '#f8f9fa';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#bdc3c7';
            uploadArea.style.background = '#f8f9fa';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('csvFile').files = files;
                currentFile = files[0];
                document.getElementById('fileName').innerHTML = 
                    `<strong><i class="fas fa-file-csv"></i> ${currentFile.name}</strong>`;
                document.getElementById('analyzeBtn').disabled = false;
            }
        });

        // Click to upload
        uploadArea.addEventListener('click', () => {
            document.getElementById('csvFile').click();
        });

        // Analyze CSV function
        function analyzeCSV() {
            if (!currentFile) {
                alert('Please select a CSV file first.');
                return;
            }

            const formData = new FormData();
            formData.append('file', currentFile);

            const analyzeBtn = document.getElementById('analyzeBtn');
            const originalText = analyzeBtn.innerHTML;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            analyzeBtn.disabled = true;

            fetch('/api/upload-csv', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('analysisResults').innerHTML = `
                        <div class="alert alert-success">
                            <h5><i class="fas fa-check-circle"></i> Analysis Complete</h5>
                            <p><strong>File:</strong> ${currentFile.name}</p>
                            <p><strong>Transactions:</strong> ${data.analysis.total_transactions}</p>
                            <p><strong>High Risk:</strong> ${data.analysis.high_risk_count}</p>
                            <p><strong>High Risk Amount:</strong> $${data.analysis.high_risk_amount.toLocaleString()}</p>
                        </div>
                    `;
                    // Reload page after 2 seconds
                    setTimeout(() => {
                        window.location.reload();
                    }, 2000);
                } else {
                    alert('Error analyzing file: ' + data.error);
                    analyzeBtn.innerHTML = originalText;
                    analyzeBtn.disabled = false;
                }
            })
            .catch(error => {
                alert('Error analyzing file. Please try again.');
                analyzeBtn.innerHTML = originalText;
                analyzeBtn.disabled = false;
            });
        }

        // Account selection functions
        function selectAccount(accountId) {
            // Update UI
            document.querySelectorAll('.account-item').forEach(item => {
                item.classList.remove('selected');
            });
            document.getElementById('account-' + accountId).classList.add('selected');
            
            selectedAccount = accountId;
            
            // Load account details
            loadAccountDetails(accountId);
        }

        async function loadAccountDetails(accountId) {
            try {
                const response = await fetch('/api/accounts/' + accountId);
                const data = await response.json();
                
                if (data.account) {
                    const acc = data.account;
                    document.getElementById('accountDetails').innerHTML = `
                        <h5>${acc.AccountName} (${acc.AccountID})</h5>
                        <p><strong>Current Balance:</strong> $${acc.NewBalance.toLocaleString()}</p>
                        <p><strong>Total Transactions:</strong> ${acc.TransactionCount}</p>
                        <p><strong>Fraudulent Transactions:</strong> ${acc.FraudCount}</p>
                        <p><strong>Risk Level:</strong> <span class="badge ${acc.RiskLevel === 'HIGH' ? 'bg-danger' : 'bg-success'}">${acc.RiskLevel}</span></p>
                    `;
                }
            } catch (error) {
                document.getElementById('accountDetails').innerHTML = '<p class="text-danger">Error loading account details</p>';
            }
        }

        function loadAccounts() {
            window.location.reload();
        }

        function resetAnalysis() {
            if (confirm('Are you sure you want to reset the analysis and clear uploaded data?')) {
                fetch('/api/reset-analysis', {
                    method: 'POST'
                })
                .then(() => {
                    window.location.reload();
                });
            }
        }

        function testSparkConnection() {
            fetch('/api/test-spark')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('✅ Spark connection successful!\\n' + data.message);
                    } else {
                        alert('❌ Spark connection failed:\\n' + data.error);
                    }
                })
                .catch(error => {
                    alert('Error testing Spark connection: ' + error);
                });
        }

        function restartSpark() {
            if (confirm('Are you sure you want to restart Spark? This will temporarily interrupt analytics.')) {
                fetch('/api/restart-spark', {
                    method: 'POST'
                })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Spark restarted successfully!');
                            setTimeout(() => {
                                window.location.reload();
                            }, 2000);
                        } else {
                            alert('Failed to restart Spark: ' + data.error);
                        }
                    })
                    .catch(error => {
                        alert('Error restarting Spark: ' + error);
                    });
            }
        }

        // Auto-select first account on load
        document.addEventListener('DOMContentLoaded', function() {
            if (accounts.length > 0) {
                selectAccount(accounts[0].AccountID);
            }
        });
    </script>
</body>
</html>
'''
    
    return html_content

# -------------------------
# API Routes
# -------------------------
@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload and analysis"""
    global current_uploaded_data
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Read the uploaded CSV file
            df = pd.read_csv(file)
            current_uploaded_data = df
            
            # Analyze the data
            analysis = analyze_uploaded_data(df)
            
            return jsonify({
                'success': True,
                'message': 'File uploaded and analyzed successfully',
                'analysis': analysis
            })
        else:
            return jsonify({'success': False, 'error': 'Please upload a CSV file'}), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing file: {str(e)}'
        }), 500

@app.route('/api/reset-analysis', methods=['POST'])
def reset_analysis():
    """Reset the analysis and clear uploaded data"""
    global current_uploaded_data
    current_uploaded_data = None
    return jsonify({'success': True, 'message': 'Analysis reset successfully'})

@app.route('/api/accounts')
def get_accounts():
    """Get all accounts with summary information"""
    global transaction_data
    try:
        if transaction_data is None:
            return jsonify({'error': 'Transaction data not loaded'}), 500
        
        accounts = []
        for account_id in transaction_data['AccountID'].unique()[:10]:  # Limit to 10 for performance
            account_data = transaction_data[transaction_data['AccountID'] == account_id]
            if len(account_data) > 0:
                account_info = account_data.iloc[0]
                fraud_count = account_data['IsFraud'].sum() if 'IsFraud' in account_data.columns else 0
                
                accounts.append({
                    'AccountID': account_id,
                    'AccountName': account_info.get('AccountName', 'Unknown'),
                    'NewBalance': float(account_info.get('NewBalance', 0)),
                    'TransactionCount': len(account_data),
                    'FraudCount': int(fraud_count),
                    'RiskLevel': 'HIGH' if fraud_count > 0 else 'LOW'
                })
        
        return jsonify({'accounts': accounts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/accounts/<account_id>')
def get_account(account_id):
    """Get specific account details"""
    global transaction_data
    try:
        if transaction_data is None:
            return jsonify({'error': 'Transaction data not loaded'}), 500
        
        account_data = transaction_data[transaction_data['AccountID'] == account_id]
        if account_data.empty:
            return jsonify({'error': 'Account not found'}), 404
        
        account_info = account_data.iloc[0]
        fraud_count = account_data['IsFraud'].sum() if 'IsFraud' in account_data.columns else 0
        
        return jsonify({
            'account': {
                'AccountID': account_id,
                'AccountName': account_info.get('AccountName', 'Unknown'),
                'OldBalance': float(account_info.get('OldBalance', 0)),
                'NewBalance': float(account_info.get('NewBalance', 0)),
                'TransactionCount': len(account_data),
                'FraudCount': int(fraud_count),
                'RiskLevel': 'HIGH' if fraud_count > 0 else 'LOW'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/spark-status')
def spark_status():
    """Check Spark status"""
    global spark, SPARK_UI_URL
    
    if spark is None:
        return jsonify({'status': 'not_running', 'spark_ui': None, 'accessible': False})
    
    try:
        accessible = check_spark_ui_status()
        return jsonify({
            'status': 'running',
            'spark_ui': SPARK_UI_URL,
            'accessible': accessible,
            'app_id': spark.sparkContext.applicationId if spark else None
        })
    except:
        return jsonify({'status': 'error', 'spark_ui': None, 'accessible': False})

@app.route('/api/test-spark')
def test_spark():
    """Test Spark connection"""
    global spark
    
    if spark is None:
        return jsonify({'success': False, 'error': 'Spark not initialized'})
    
    try:
        # Run a simple Spark query
        df = spark.range(100)
        count = df.count()
        
        # Check if Spark UI is accessible
        accessible = check_spark_ui_status()
        
        return jsonify({
            'success': True,
            'message': f'Spark is working! Processed {count} rows.',
            'spark_ui': SPARK_UI_URL,
            'accessible': accessible,
            'app_id': spark.sparkContext.applicationId
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/restart-spark', methods=['POST'])
def api_restart_spark():
    """API endpoint to restart Spark"""
    global spark
    
    try:
        success = restart_spark_session()
        if success:
            return jsonify({
                'success': True,
                'message': 'Spark restarted successfully',
                'spark_ui': SPARK_UI_URL
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to restart Spark'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error restarting Spark: {str(e)}'
        })

@app.route('/api/open-spark-ui')
def open_spark_ui():
    """Open Spark UI in browser"""
    global SPARK_UI_URL
    
    if not SPARK_UI_URL:
        return jsonify({'success': False, 'error': 'Spark UI not available'})
    
    try:
        webbrowser.open(SPARK_UI_URL)
        return jsonify({'success': True, 'message': f'Opened {SPARK_UI_URL}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# -------------------------
# Main entrypoint
# -------------------------
if __name__ == '__main__':
    try:
        # Initialize system
        if initialize_system():
            print("\n" + "="*60)
            print("✅ SYSTEM READY")
            print("="*60)
            print(f"🌐 Dashboard: http://localhost:5000")
            print(f"⚡ Spark UI: {SPARK_UI_URL if SPARK_UI_URL else 'Not available'}")
            print("="*60)
            print("\n📊 Press Ctrl+C to stop the application")
            print("="*60)
            
            # Run Flask app
            app.run(debug=True, host='0.0.0.0', port=5000, threaded=False, use_reloader=False)
        else:
            print("❌ Failed to initialize system")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping application...")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
    finally:
        # Clean up Spark session
        if spark is not None:
            try:
                print("🧹 Stopping Spark session...")
                spark.stop()
                print("✅ Spark session stopped")
            except:
                pass