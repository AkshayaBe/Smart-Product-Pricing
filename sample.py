# %%
# =============================================================================
# Cell 1: Import Libraries and Load Data
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("--- Cell 1: Loading Data ---")
# Use a relative path to make the code portable
df_train = pd.read_csv('student_resource/dataset/train.csv')

print("Data loaded successfully. First 5 rows:")
# In a .py file, you use print() to see DataFrames
print(df_train.head())

print("\nData Info:")
df_train.info()
print("--- Cell 1: Complete ---")


# %%
# =============================================================================
# Cell 2: Analyze the Price Distribution
# =============================================================================
print("\n--- Cell 2: Analyzing Price ---")
print("Price Statistics:")
print(df_train['price'].describe())

# The plot will appear in the Interactive Window
plt.figure(figsize=(12, 6))
sns.histplot(df_train['price'], bins=100, kde=True)
plt.title('Distribution of Product Prices')
plt.xlabel('Price')
plt.show()
print("--- Cell 2: Complete ---")

# %%
# =============================================================================
# NEW CELL: Image Feature Extraction
# =============================================================================
# %%
# =============================================================================
# NEW CELL: Image Feature Extraction (Corrected)
# =============================================================================
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import os

print("\n--- Image Feature Extraction ---")

# === THIS IS THE FIX ===
# Load df_test *before* the try block so it always exists.
df_test = pd.read_csv('student_resource/dataset/test.csv')

# Check if pre-computed features exist to save time
try:
    train_image_features = np.load('train_image_features.npy')
    test_image_features = np.load('test_image_features.npy')
    print("Loaded pre-computed image features from .npy files.")
except FileNotFoundError:
    print("Pre-computed features not found. Starting extraction...")
    # Set up the pre-trained ResNet-18 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)
    model.eval()

    # Define the image transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0).to(device)
            with torch.no_grad():
                features = model(batch_t)
            return features.squeeze().cpu().numpy()
        except Exception as e:
            print(f"Warning: Could not process {image_path}. Error: {e}")
            return np.zeros(512)
    
    train_image_paths = [f"images/train/{sid}.jpg" for sid in df_train['sample_id']]
    test_image_paths = [f"images/test/{sid}.jpg" for sid in df_test['sample_id']]

    # Extract features
    train_image_features = np.array([extract_features(p) for p in tqdm(train_image_paths, desc="Processing Train Images")])
    test_image_features = np.array([extract_features(p) for p in tqdm(test_image_paths, desc="Processing Test Images")])

    # Save the features
    np.save('train_image_features.npy', train_image_features)
    np.save('test_image_features.npy', test_image_features)
    print("Image feature extraction complete and saved to .npy files.")

# Create DataFrames from the features
train_img_df = pd.DataFrame(train_image_features, index=df_train.index)
test_img_df = pd.DataFrame(test_image_features, index=df_test.index)
print("--- Image Feature Extraction: Complete ---")

# %%
# =============================================================================
# Cell 3: Build and Train a Baseline Model
# =============================================================================
# %%
# %%
# =============================================================================
# Cell 3: Combine Features and Train Model (Updated)
# =============================================================================
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix

print("\n--- Cell 3: Combining Features & Training ---")

# Vectorize the text data for the full dataset
vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
print("Fitting TF-IDF on all training data...")
all_X_train_tfidf = vectorizer.fit_transform(df_train['catalog_content'])

# Combine text features and image features
# Use hstack to horizontally stack the sparse text matrix and the dense image matrix
X_train_combined = hstack([all_X_train_tfidf, csr_matrix(train_img_df.values)])

# We will use cross-validation later, for now let's train on all data
y_train_full_log = np.log1p(df_train['price'])

print("Training a LightGBM model on combined text and image data...")
lgbm_final = lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31)
lgbm_final.fit(X_train_combined, y_train_full_log)

print("Model training complete.")
print("--- Cell 3: Complete ---")


# %%
# =============================================================================
# Cell 4: Evaluate the Model
# =============================================================================

# %%
# =============================================================================
# Cell 4: Generate Submission File
# =============================================================================
print("\n--- Cell 4: Generating Submission File ---")

# 1. Prepare the test data features
# IMPORTANT: Use the same vectorizer that was FITTED on the training data
print("Transforming test set text data...")
X_test_tfidf = vectorizer.transform(df_test['catalog_content'])

# Combine the test text features and test image features
print("Combining test set features...")
X_test_combined = hstack([X_test_tfidf, csr_matrix(test_img_df.values)])

# 2. Make predictions
print("Making predictions on the test set...")
# The model predicts in "log-language"
log_predictions_test = lgbm_final.predict(X_test_combined)

# Translate the predictions back to dollar prices
final_predictions = np.expm1(log_predictions_test)

# Ensure prices are not negative
final_predictions[final_predictions < 0] = 0

# 3. Create the submission DataFrame
print("Creating submission file...")
submission_df = pd.DataFrame({
    'sample_id': df_test['sample_id'],
    'price': final_predictions
})

# 4. Save the submission file
# The index=False part is very important for the submission format
submission_df.to_csv('test_out.csv', index=False)

print("\nSubmission file 'test_out.csv' has been created successfully!")
print("--- Cell 4: Complete ---")